# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PolyNorm3 backward microbenchmark in Python using nanobind-backed ``ttml`` ops.
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
POLYNORM_EPS = 1e-5
POLYNORM_W0 = 0.2
POLYNORM_W1 = 0.3
POLYNORM_W2 = 0.5
NUM_WARMUP = 5
NUM_MEASURE = 50

_TTL_WORKER_L1_RESERVE_BYTES = 77824


SHAPES = (
    ([1, 1, 256, 384], "256x384"),
    ([1, 1, 256, 2048], "256x2048"),
    ([1, 1, 2048, 5632], "2048x5632"),
    ([2, 1, 2048, 5632], "4096x5632"),
    ([4, 1, 2048, 5632], "8192x5632"),
    ([8, 1, 2048, 5632], "16384x5632"),
    ([16, 1, 2048, 5632], "32768x5632"),
    ([1, 1, 4096, 4096], "4096x4096"),
    ([2, 1, 4096, 4096], "8192x4096"),
    ([4, 1, 4096, 4096], "16384x4096"),
    ([8, 1, 4096, 4096], "32768x4096"),
    ([16, 1, 4096, 4096], "65536x4096"),
)


import ttl_polynorm3_bw  # noqa: PLC0415
import ttml
from ttml.autograd import AutoContext, Tensor, create_tensor


def _open_mesh_for_kernel_bw(ctx: AutoContext, kernel: str) -> None:
    """Open a single-device mesh ``(1, 1)``; TTL reserves worker L1 for large kernel configs."""
    if kernel != "ttl":
        ctx.open_device((1, 1))
        return
    max_l1 = ttnn.device.get_max_worker_l1_unreserved_size()
    if _TTL_WORKER_L1_RESERVE_BYTES >= max_l1:
        raise RuntimeError(
            f"_TTL_WORKER_L1_RESERVE_BYTES ({_TTL_WORKER_L1_RESERVE_BYTES}) must be less than "
            f"max worker L1 unreserved size ({max_l1}); not enough L1 to reserve headroom for TTL kernel configs."
        )
    worker_l1 = max_l1 - _TTL_WORKER_L1_RESERVE_BYTES
    try:
        ctx.open_device((1, 1), None, worker_l1)
    except TypeError:
        ctx.open_device((1, 1))



def _seed_for_name(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def _weight_numpy() -> np.ndarray:
    w = np.zeros((1, 1, 1, 3), dtype=np.float32)
    w[0, 0, 0, 0] = POLYNORM_W0
    w[0, 0, 0, 1] = POLYNORM_W1
    w[0, 0, 0, 2] = POLYNORM_W2
    return w


def _metal_polynorm_bw_tensors_from_numpy(
    x_np: np.ndarray,
    d_np: np.ndarray,
    w_np: np.ndarray,
):
    """Numpy activations / upstream grad / weight → tile-backed autograd tensors for ``polynorm3_bw``."""
    x_storage = Tensor.from_numpy(x_np, ttnn.Layout.TILE)
    d_storage = Tensor.from_numpy(d_np, ttnn.Layout.TILE)
    w_storage = Tensor.from_numpy(w_np, ttnn.Layout.TILE)
    x_t = create_tensor(x_storage.get_value(), False)
    dL_t = create_tensor(d_storage.get_value(), False)
    w_t = create_tensor(w_storage.get_value(), False)
    return x_t, dL_t, w_t


def _ttl_polynorm_bw_tensors_to_device(
    ttl_mod,
    mesh,
    x2: torch.Tensor,
    dL2: torch.Tensor,
    w0: float,
    w1: float,
    w2: float,
    eps: float,
    height: int,
    width: int,
):
    """Build TTL PolyNorm3 backward device tensors from 2D host torch buffers."""
    wstrip = torch.zeros(TILE, 3 * TILE, dtype=torch.float32)
    wstrip[:, 0:TILE] = w2
    wstrip[:, TILE : 2 * TILE] = w1
    wstrip[:, 2 * TILE : 3 * TILE] = w0
    eps_t = torch.full((TILE, TILE), eps, dtype=torch.float32)

    x_tt = ttl_mod._to_dev_f32(mesh, x2.float())
    dout_tt = ttl_mod._to_dev_f32(mesh, dL2.float())
    w_tt = ttl_mod._to_dev_f32(mesh, wstrip)
    ep_tt = ttl_mod._to_dev_f32(mesh, eps_t)
    gx_tt = ttl_mod._to_dev_f32(mesh, torch.zeros(height, width, dtype=torch.float32))
    gp_tt = ttl_mod._to_dev_f32(mesh, torch.zeros(height, 4 * TILE, dtype=torch.float32))
    return x_tt, dout_tt, w_tt, ep_tt, gx_tt, gp_tt


def _run_kernel(kernel: str = "metal") -> None:
    """Benchmark PolyNorm3 backward kernel only (random upstream grad on host)."""

    ttl_mod = None
    if kernel == "ttl":
        ttl_mod = ttl_polynorm3_bw

    ctx = AutoContext.get_instance()
    _open_mesh_for_kernel_bw(ctx, kernel)
    try:
        mesh = ctx.get_device()
        mesh.enable_program_cache()

        tag = "TTL" if kernel == "ttl" else "Metal"
        print(f"Mode: KernelBw / {tag} (backward-only, random dL_dout on host)\n")

        w_np = _weight_numpy()

        for shape, name in SHAPES:
            seed = _seed_for_name(name)
            rng = np.random.default_rng(seed)
            b, _, s_len, c = shape
            x_np = rng.uniform(-1.0, 1.0, (b, 1, s_len, c)).astype(np.float32)
            d_np = rng.uniform(-1.0, 1.0, (b, 1, s_len, c)).astype(np.float32)

            if kernel == "metal":
                x_t, dL_t, w_t = _metal_polynorm_bw_tensors_from_numpy(x_np, d_np, w_np)

                def run_step() -> None:
                    ttml.ops.polynorm.polynorm3_bw(x_t, dL_t, w_t, POLYNORM_EPS)

            elif kernel == "ttl":
                assert ttl_mod is not None
                rows = b * s_len
                columns = c
                x2 = torch.from_numpy(x_np).reshape(rows, columns)
                dL2 = torch.from_numpy(d_np).reshape(rows, columns)
                x_tt, dout_tt, w_tt, ep_tt, gx_tt, gp_tt = _ttl_polynorm_bw_tensors_to_device(
                    ttl_mod,
                    mesh,
                    x2,
                    dL2,
                    POLYNORM_W0,
                    POLYNORM_W1,
                    POLYNORM_W2,
                    POLYNORM_EPS,
                    rows,
                    columns,
                )

                def run_step() -> None:
                    ttl_mod.polynorm3_bw(x_tt, dout_tt, w_tt, ep_tt, gx_tt, gp_tt)

            else:
                raise ValueError(f"unknown kernel: {kernel}")

            for _ in range(NUM_WARMUP):
                run_step()

            total = 0.0
            for _ in range(NUM_MEASURE):
                t0 = time.perf_counter()
                run_step()
                ttnn.synchronize_device(mesh)
                total += time.perf_counter() - t0
            avg_s = total / NUM_MEASURE
            suffix = "Kernel_TTL" if kernel == "ttl" else "Kernel_Metal"
            row = f"{name}_{suffix}"
            print(f"  {row:28}  Time_us={avg_s * 1e6:10.2f}")
    finally:
        ctx.reset_graph()
        ctx.close_device()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--kernel",
        choices=("metal", "ttl"),
        default="metal",
        help=(
            "metal: ttml "
            "ttl: ttlang "
        ),
    )
    args = p.parse_args()

    print(f"warmup={NUM_WARMUP}  measure={NUM_MEASURE}  epsilon={POLYNORM_EPS}  " f"--kernel={args.kernel}\n")

    _run_kernel(args.kernel)
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
