# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm microbenchmark in Python using nanobind-backed ``ttml`` ops.
"""


from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import ttnn

TILE = 32
RMSNORM_EPS = 0.0078125
NUM_WARMUP = 5
NUM_MEASURE = 50

_TTL_WORKER_L1_RESERVE_BYTES = 77824

_TTML_SOURCES = Path(__file__).resolve().parents[1]
_TTLANG = _TTML_SOURCES / "ttlang"
for _p in (_TTML_SOURCES, _TTLANG):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _ensure_ttl_package_path() -> None:
    """Put the nanobind ``ttl`` package on ``sys.path`` (same logic as ``ttl_tml_rmsnorm_comp``)."""
    candidates: list[Path] = []
    env = os.environ.get("TTL_PYTHON_PACKAGES", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path("/opt/ttlang-toolchain/python_packages"))
    exe_parent = Path(sys.executable).resolve().parent.parent
    if (exe_parent / "python_packages").is_dir():
        candidates.append(exe_parent / "python_packages")
    for p in candidates:
        if p.is_dir() and (p / "ttl").is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
            return


def _import_ttl_rmsnorm_bw_2pass():
    """Import ``ttl_rmsnorm_bw_2pass`` (2-pass tile-streaming TTL backward; needs ``ttl``)."""
    _ensure_ttl_package_path()
    import ttl_rmsnorm_bw_2pass  # noqa: PLC0415

    return ttl_rmsnorm_bw_2pass


import ttml
from ttml.autograd import AutoContext, Tensor, create_tensor


def _open_mesh_for_kernel_bw(ctx: AutoContext, bw_kernel: str) -> None:
    """Open a single-device mesh ``(1, 1)``; TTL reserves worker L1 for large kernel configs."""
    if bw_kernel != "ttl_2pass":
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
    ([1, 1, 256, 384], "256x384"),
    ([1, 1, 256, 2048], "256x2048"),
    ([1, 1, 2048, 2048], "2048x2048"),
    ([2, 1, 2048, 2048], "4096x2048"),
    ([4, 1, 2048, 2048], "8192x2048"),
    ([8, 1, 2048, 2048], "16384x2048"),
    ([16, 1, 2048, 2048], "32768x2048"),
    ([1, 1, 2048, 4096], "2048x4096"),
    ([2, 1, 2048, 4096], "4096x4096"),
    ([4, 1, 2048, 4096], "8192x4096"),
    ([8, 1, 2048, 4096], "16384x4096"),
    ([16, 1, 2048, 4096], "32768x4096")
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
    g1d = torch.from_numpy(g_np).reshape(columns)
    g2 = g1d.unsqueeze(0).expand(rows, columns).contiguous()
    r_row = torch.from_numpy(rms_np).squeeze(-1).reshape(rows, 1)
    r2 = r_row.expand(rows, columns).contiguous()
    return x2, dL2, g2, r2


def _ttl_rmsnorm_bw_tensors_to_padded_device(
    ttl_mod,
    mesh,
    x2: torch.Tensor,
    g2: torch.Tensor,
    r2: torch.Tensor,
    dL2: torch.Tensor,
    rows_p: int,
    cols_p: int,
):
    """Pad 2D torch inputs to ``(rows_p, cols_p)`` and ``_to_dev``."""
    x_p = ttl_mod._to_dev(ttl_mod._pad(x2, rows_p, cols_p), mesh)
    g_p = ttl_mod._to_dev(ttl_mod._pad(g2, rows_p, cols_p), mesh)
    rms_p = ttl_mod._to_dev(ttl_mod._pad(r2, rows_p, cols_p), mesh)
    dL_p = ttl_mod._to_dev(ttl_mod._pad(dL2, rows_p, cols_p), mesh)
    out_da = ttl_mod._to_dev(torch.zeros(rows_p, cols_p, dtype=torch.bfloat16), mesh)
    out_dg = ttl_mod._to_dev(torch.zeros(rows_p, cols_p, dtype=torch.bfloat16), mesh)
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


def _run_kernel_bw_only(bw_kernel: str = "metal") -> None:
    """Benchmark RMSNorm backward kernel"""

    ttl_mod = None
    twopass_mod = None
    if bw_kernel == "ttl_2pass":
        twopass_mod = _import_ttl_rmsnorm_bw_2pass()
        import ttl_rmsnorm_bw  # noqa: PLC0415

        ttl_mod = ttl_rmsnorm_bw

    ctx = AutoContext.get_instance()
    _open_mesh_for_kernel_bw(ctx, bw_kernel)
    try:
        mesh = ctx.get_device()
        mesh.enable_program_cache()

        if bw_kernel == "metal":
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

            if bw_kernel == "metal":
                x_t, g_t, dL_t, rms_t = _metal_rmsnorm_bw_tensors_from_numpy(x_np, g_np, d_np, rms_np)

                def run_step() -> None:
                    d_in, d_gamma = ttml.ops.rmsnorm.rmsnorm_bw(x_t, g_t, rms_t, dL_t)

            elif bw_kernel == "ttl_2pass":
                assert ttl_mod is not None
                assert twopass_mod is not None
                rows = b * s_len
                columns = c
                x2, dL2, g2, r2 = _ttl_rmsnorm_bw_numpy_to_2d(x_np, d_np, g_np, rms_np, rows, columns)
                rows_p, cols_p = _tile_padded_shape(rows, columns)
                x_p, g_p, rms_p, dL_p, out_da, out_dg = _ttl_rmsnorm_bw_tensors_to_padded_device(
                    ttl_mod, mesh, x2, g2, r2, dL2, rows_p, cols_p
                )
                k_2pass = twopass_mod.make_kernel()

                def run_step() -> None:
                    k_2pass(x_p, g_p, rms_p, dL_p, out_da, out_dg)

            else:
                raise ValueError(f"unknown bw_kernel: {bw_kernel}")

            for _ in range(NUM_WARMUP):
                run_step()

            total = 0.0
            for _ in range(NUM_MEASURE):
                t0 = time.perf_counter()
                run_step()
                ttnn.synchronize_device(mesh)
                total += time.perf_counter() - t0
            avg_s = total / NUM_MEASURE
            if bw_kernel == "metal":
                suffix = "KernelBw_Metal"
            else:
                suffix = "KernelBw_TTL_2Pass"
            row = f"{name}_{suffix}"
            print(f"  {row:28}  Time_us={avg_s * 1e6:10.2f}")
    finally:
        ctx.reset_graph()
        ctx.close_device()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bw-kernel",
        choices=("metal", "ttl_2pass"),
        default="metal",
        help=(
            "metal: ttml rmsnorm_bw; "
            "ttl_2pass: ttl rmsnorm_bw "
        ),
    )
    args = p.parse_args()

    print(f"warmup={NUM_WARMUP}  measure={NUM_MEASURE}  epsilon={RMSNORM_EPS}  --bw-kernel={args.bw_kernel}\n")

    _run_kernel_bw_only(args.bw_kernel)
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
