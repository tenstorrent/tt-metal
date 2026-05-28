# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm microbenchmark in Python using nanobind-backed ``ttml`` ops.
"""


from __future__ import annotations

import argparse
import sys
import time
import zlib

import numpy as np
import torch
import ttnn

TILE = 32
RMSNORM_EPS = 0.0078125
NUM_WARMUP = 5
NUM_MEASURE = 50

_TTL_WORKER_L1_RESERVE_BYTES = 90112

import ttl_rmsnorm_bw_2pass
import ttml
from ttml.autograd import AutoContext, Tensor, create_tensor


def _open_mesh_for_kernel_bw(ctx: AutoContext, kernel: str) -> None:
    """Open a single-device mesh ``(1, 1)``; TTL reserves worker L1 for large kernel configs."""
    if kernel != "ttl_2pass":
        ctx.open_device((1, 1))
        return
    max_l1 = ttnn.device.get_max_worker_l1_unreserved_size()
    if _TTL_WORKER_L1_RESERVE_BYTES >= max_l1:
        raise RuntimeError(
            f"_TTL_WORKER_L1_RESERVE_BYTES ({_TTL_WORKER_L1_RESERVE_BYTES}) must be less than "
            f"max worker L1 unreserved size ({max_l1}); not enough L1 to reserve headroom for TTL kernel configs."
        )
    worker_l1 = max_l1 - _TTL_WORKER_L1_RESERVE_BYTES
    ctx.open_device((1, 1), None, worker_l1)


SHAPES = (
    ([1, 1, 2048, 5632], "2048x5632"),
    ([2, 1, 2048, 5632], "4096x5632"),
    ([4, 1, 2048, 5632], "8192x5632"),
    ([8, 1, 2048, 5632], "16384x5632"),
    ([16, 1, 2048, 5632], "32768x5632"),
)


def _torch_rmsnorm_rms_numpy(x_np: np.ndarray, eps: float) -> np.ndarray:
    """``sqrt(mean(x^2, dim=C, keepdim) + eps)`` → ``[B,1,S,1]`` float32, same reduction as ``rmsnorm_op`` composite."""
    x = torch.from_numpy(np.asarray(x_np, dtype=np.float32))
    mean_sq = (x * x).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    return np.asarray(rms.numpy(), dtype=np.float32)


def _seed_for_name(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def _ttl_rmsnorm_bw_numpy_to_2d(
    x_np: np.ndarray,
    d_np: np.ndarray,
    g_np: np.ndarray,
    rms_np: np.ndarray,
    rows: int,
    columns: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Change ``[B,1,S,C]`` / ``[B,1,S,1]`` / ``[1,1,1,C]`` arrays to 2D."""
    x2 = torch.from_numpy(x_np).reshape(rows, columns)
    dL2 = torch.from_numpy(d_np).reshape(rows, columns)
    g_row = torch.from_numpy(g_np).reshape(columns).unsqueeze(0)
    r_row = torch.from_numpy(rms_np).squeeze(-1).reshape(rows, 1)
    return x2, dL2, g_row, r_row


def _ttl_rmsnorm_bw_tensors_to_padded_device(
    ttl_mod,
    mesh,
    x2: torch.Tensor,
    g_row: torch.Tensor,
    r_row: torch.Tensor,
    dL2: torch.Tensor,
    rows_p: int,
    cols_p: int,
):
    """Pad 2D torch inputs and ``to_dev`` (gamma ``[1, cols_p]``, rms ``[rows_p, 1]``)."""
    x_p = ttl_mod.to_dev(ttl_mod.pad(x2, rows_p, cols_p), mesh)
    g_p = ttl_mod.to_dev(ttl_mod.pad(g_row, 1, cols_p), mesh)
    rms_p = ttl_mod.to_dev(ttl_mod.pad(r_row, rows_p, 1), mesh)
    dL_p = ttl_mod.to_dev(ttl_mod.pad(dL2, rows_p, cols_p), mesh)
    out_da = ttl_mod.to_dev(torch.zeros(rows_p, cols_p, dtype=torch.bfloat16), mesh)
    out_dg = ttl_mod.to_dev(torch.zeros(rows_p, cols_p, dtype=torch.bfloat16), mesh)
    return x_p, g_p, rms_p, dL_p, out_da, out_dg


def _tile_padded_shape(rows: int, cols: int) -> tuple[int, int]:
    return -(-rows // TILE) * TILE, -(-cols // TILE) * TILE


def _metal_rmsnorm_bw_tensors_from_numpy(
    x_np: np.ndarray,
    g_np: np.ndarray,
    d_np: np.ndarray,
    rms_np: np.ndarray,
):
    """Numpy activations / gamma / upstream grad / RMS → tile-backed autograd tensors for ``rmsnorm_bw``."""
    x_storage = Tensor.from_numpy(x_np, ttnn.Layout.TILE)
    g_storage = Tensor.from_numpy(g_np, ttnn.Layout.TILE)
    d_storage = Tensor.from_numpy(d_np, ttnn.Layout.TILE)
    rms_storage = Tensor.from_numpy(rms_np, ttnn.Layout.TILE)
    x_t = create_tensor(x_storage.get_value(), False)
    g_t = create_tensor(g_storage.get_value(), False)
    dL_t = create_tensor(d_storage.get_value(), False)
    rms_t = create_tensor(rms_storage.get_value(), False)
    return x_t, g_t, dL_t, rms_t


def _run_kernel(kernel: str = "metal") -> None:
    """Benchmark RMSNorm backward kernel"""

    ttl_mod = None
    if kernel == "ttl_2pass":
        ttl_mod = ttl_rmsnorm_bw_2pass

    ctx = AutoContext.get_instance()
    _open_mesh_for_kernel_bw(ctx, kernel)
    try:
        mesh = ctx.get_device()
        mesh.enable_program_cache()

        if kernel == "metal":
            tag = "Metal"
        else:
            tag = "TTL (2-pass)"
        print(f"Mode: KernelBw / {tag} (backward-only, RMS from torch forward on host)\n")

        for shape, name in SHAPES:
            seed = _seed_for_name(name)
            rng = np.random.default_rng(seed)
            b, _, s_len, c = shape
            x_np = rng.uniform(-1.0, 1.0, (b, 1, s_len, c)).astype(np.float32)
            g_np = rng.uniform(-1.0, 1.0, (1, 1, 1, c)).astype(np.float32)
            d_np = rng.uniform(-1.0, 1.0, (b, 1, s_len, c)).astype(np.float32)
            rms_np = _torch_rmsnorm_rms_numpy(x_np, RMSNORM_EPS)

            if kernel == "metal":
                x_t, g_t, dL_t, rms_t = _metal_rmsnorm_bw_tensors_from_numpy(x_np, g_np, d_np, rms_np)

                def run_step() -> None:
                    _, _ = ttml.ops.rmsnorm.rmsnorm_bw(x_t, g_t, rms_t, dL_t)

            elif kernel == "ttl_2pass":
                assert ttl_mod is not None
                rows = b * s_len
                columns = c
                x2, dL2, g_row, r_row = _ttl_rmsnorm_bw_numpy_to_2d(x_np, d_np, g_np, rms_np, rows, columns)
                rows_p, cols_p = _tile_padded_shape(rows, columns)
                x_p, g_p, rms_p, dL_p, out_da, out_dg = _ttl_rmsnorm_bw_tensors_to_padded_device(
                    ttl_mod, mesh, x2, g_row, r_row, dL2, rows_p, cols_p
                )
                ttl_kernel = ttl_mod.make_kernel()

                def run_step() -> None:
                    _, _ = ttl_mod.run_rmsnorm_bw_2pass(mesh, ttl_kernel, x_p, g_p, rms_p, dL_p, out_da, out_dg)

            else:
                raise ValueError(f"unknown kernel: {kernel}")

            for _ in range(NUM_WARMUP):
                run_step()
            ttnn.synchronize_device(mesh)

            t0 = time.perf_counter()
            for _ in range(NUM_MEASURE):
                run_step()
            ttnn.synchronize_device(mesh)
            total = time.perf_counter() - t0
            avg_s = total / NUM_MEASURE
            if kernel == "metal":
                suffix = "Kernel_Metal"
            else:
                suffix = "Kernel_TTL_2Pass"
            row = f"{name}_{suffix}"
            print(f"  {row:28}  Time_us={avg_s * 1e6:10.2f}")
    finally:
        ctx.reset_graph()
        ctx.close_device()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--kernel",
        choices=("metal", "ttl_2pass"),
        default="metal",
        help=("metal: ttml rmsnorm_bw; " "ttl_2pass: ttl_2pass rmsnorm_bw "),
    )
    args = p.parse_args()

    print(f"warmup={NUM_WARMUP}  measure={NUM_MEASURE}  epsilon={RMSNORM_EPS}  --kernel={args.kernel}\n")

    _run_kernel(args.kernel)
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
