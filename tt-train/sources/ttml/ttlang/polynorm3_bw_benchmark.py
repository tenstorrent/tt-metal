# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PolyNorm3 backward microbenchmark in Python using nanobind-backed ``ttml`` ops.

    To build tt-train on tt-lang IRD image (one shot; adjust ``cd`` if your clone is not under ``/localdev/$(whoami)/tt-metal``):
    cd "/localdev/$(whoami)/tt-metal" && R="$PWD" && export TT_METAL_HOME="$R" TT_METAL_RUNTIME_ROOT="$R" LD_LIBRARY_PATH="${R}/build_Release/tt_metal:${R}/build_Release/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" PYTHONPATH="${R}/ttnn:${R}/tools:$R:${R}/tt-train/sources/ttml:${R}/build_Release/tt-train/sources/ttml${PYTHONPATH:+:$PYTHONPATH}" && ./build_metal.sh -b Release --build-tt-train

    One shot command to run the benchmark on IRD with tt-lang image where tt-metal repo is mounted at /localdev/$(whoami)/tt-metal
    cd /localdev/$(whoami)/tt-metal && R="$PWD" && export TT_METAL_HOME="$R" TT_METAL_LOGGER_LEVEL=FATAL TT_METAL_RUNTIME_ROOT="$R" TT_METAL_DISABLE_PRECOMPILED_FW=1 PYTHONPATH="$R/ttnn:$R/tools:$R:$R/tt-train/sources/ttml:$R/build_Release/tt-train/sources/ttml" LD_LIBRARY_PATH="$R/build_Release/lib" && python3 tt-train/sources/ttml/ttlang/polynorm3_bw_benchmark.py --bw-kernel metal
    cd /localdev/$(whoami)/tt-metal && R="$PWD" && export TT_METAL_HOME="$R" TT_METAL_LOGGER_LEVEL=FATAL TT_METAL_RUNTIME_ROOT="$R" TT_METAL_DISABLE_PRECOMPILED_FW=1 PYTHONPATH="$R/ttnn:$R/tools:$R:$R/tt-train/sources/ttml:$R/build_Release/tt-train/sources/ttml" LD_LIBRARY_PATH="$R/build_Release/lib" && python3 tt-train/sources/ttml/ttlang/polynorm3_bw_benchmark.py --bw-kernel ttl
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

_TTML_SOURCES = Path(__file__).resolve().parents[1]
_TTLANG = _TTML_SOURCES / "ttlang"
for _p in (_TTML_SOURCES, _TTLANG):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _ensure_ttl_package_path() -> None:
    """Put the nanobind ``ttl`` package on ``sys.path``."""
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


def _import_ttl_polynorm3_bw():
    """Import ``ttl_polynorm3_bw`` (needs ``ttl`` + this repo's ``ttlang`` on ``sys.path``)."""
    _ensure_ttl_package_path()
    import ttl_polynorm3_bw  # noqa: PLC0415

    return ttl_polynorm3_bw


import ttml
from ttml.autograd import AutoContext, Tensor, create_tensor


def _open_mesh_for_kernel_bw(ctx: AutoContext, bw_kernel: str) -> None:
    """Open a single-device mesh ``(1, 1)``; TTL reserves worker L1 for large kernel configs."""
    if bw_kernel != "ttl":
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


SHAPES = (
    ([1, 1, 256, 384], "256x384"),
    ([1, 1, 256, 2048], "256x2048"),
    ([1, 1, 2048, 2048], "2048x2048"),
    ([1, 1, 2048, 5632], "2048x5632"),
    ([4, 1, 1024, 4096], "4096x4096"),
    ([8, 1, 1024, 8192], "8192x8192"),
)


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


def _run_kernel_bw_only(bw_kernel: str = "metal") -> None:
    """Benchmark PolyNorm3 backward kernel only (random upstream grad on host)."""

    ttl_mod = _import_ttl_polynorm3_bw() if bw_kernel == "ttl" else None

    ctx = AutoContext.get_instance()
    _open_mesh_for_kernel_bw(ctx, bw_kernel)
    try:
        mesh = ctx.get_device()
        mesh.enable_program_cache()

        tag = "TTL" if bw_kernel == "ttl" else "Metal"
        print(f"Mode: KernelBw / {tag} (backward-only, random dL_dout on host)\n")

        w_np = _weight_numpy()

        for shape, name in SHAPES:
            seed = _seed_for_name(name)
            rng = np.random.default_rng(seed)
            b, _, s_len, c = shape
            x_np = rng.uniform(-1.0, 1.0, (b, 1, s_len, c)).astype(np.float32)
            d_np = rng.uniform(-1.0, 1.0, (b, 1, s_len, c)).astype(np.float32)

            if bw_kernel == "metal":
                x_t, dL_t, w_t = _metal_polynorm_bw_tensors_from_numpy(x_np, d_np, w_np)

                def run_step() -> None:
                    ttml.ops.polynorm.polynorm3_bw(x_t, dL_t, w_t, POLYNORM_EPS)

            elif bw_kernel == "ttl":
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
            suffix = "KernelBw_TTL" if bw_kernel == "ttl" else "KernelBw_Metal"
            row = f"{name}_{suffix}"
            print(f"  {row:28}  Time_us={avg_s * 1e6:10.2f}")
    finally:
        ctx.reset_graph()
        ctx.close_device()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bw-kernel",
        choices=("metal", "ttl"),
        default="metal",
        help=(
            "metal: ttml ``metal/ops/polynorm_bw`` via ``ttml.ops.polynorm.polynorm3_bw``; "
            "ttl: ``ttl_polynorm3_bw.py``. TTL mode needs ``ttl`` (+ ``tt-lang`` on path)."
        ),
    )
    args = p.parse_args()

    print(f"warmup={NUM_WARMUP}  measure={NUM_MEASURE}  epsilon={POLYNORM_EPS}  " f"--bw-kernel={args.bw_kernel}\n")

    _run_kernel_bw_only(args.bw_kernel)
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
