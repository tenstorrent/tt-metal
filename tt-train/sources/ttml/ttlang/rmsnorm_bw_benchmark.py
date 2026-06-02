# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm microbenchmark in Python using nanobind-backed ``ttml`` ops.

    To build tt-train on tt-lang IRD image (one shot; adjust ``cd`` if your clone is not under ``/localdev/$(whoami)/tt-metal``):
    cd "/localdev/$(whoami)/tt-metal" && R="$PWD" && export TT_METAL_HOME="$R" TT_METAL_RUNTIME_ROOT="$R" LD_LIBRARY_PATH="${R}/build_Release/tt_metal:${R}/build_Release/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" PYTHONPATH="${R}/ttnn:${R}/tools:${R}:${R}/tt-train/sources/ttml:${R}/build_Release/tt-train/sources/ttml${PYTHONPATH:+:$PYTHONPATH}" && ./build_metal.sh -b Release --build-tt-train

    One shot command to run the benchmark on IRD with tt-lang image where tt-metal repo is mounted at /localdev/$(whoami)/tt-metal
    cd /localdev/$(whoami)/tt-metal && R="$PWD" && export TT_METAL_HOME="$R" TT_METAL_LOGGER_LEVEL=FATAL TT_METAL_RUNTIME_ROOT="$R" TT_METAL_DISABLE_PRECOMPILED_FW=1 PYTHONPATH="$R/ttnn:$R/tools:$R:$R/tt-train/sources/ttml:$R/build_Release/tt-train/sources/ttml" LD_LIBRARY_PATH="$R/build_Release/lib" && python3 tt-train/sources/ttml/ttlang/rmsnorm_bw_benchmark.py

    Current results:
TTML:
    256x384_KernelBw_Metal        Time_us=    158.08
    256x2048_KernelBw_Metal       Time_us=    246.02
    1024x2048_KernelBw_Metal      Time_us=    283.80
    2048x2048_KernelBw_Metal      Time_us=    373.95
    2048x5632_KernelBw_Metal      Time_us=    825.47
    4096x4096_KernelBw_Metal      Time_us=   1189.35
    4096x11008_KernelBw_Metal     Time_us=   3447.24
    8192x8192_KernelBw_Metal      Time_us=   4465.03

TTLANG:
    256x384_KernelBw_TTL          Time_us=    249.63
    256x2048_KernelBw_TTL         Time_us=    250.13
    1024x2048_KernelBw_TTL        Time_us=    393.97
    2048x2048_KernelBw_TTL        Time_us=    506.14
    2048x5632_KernelBw_TTL        Time_us=    817.26
    4096x4096_KernelBw_TTL        Time_us=   1033.92
    4096x11008_KernelBw_TTL       Time_us=   2338.71
    8192x8192_KernelBw_TTL        Time_us=   3380.75


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


def _import_ttl_rmsnorm_bw():
    """Import ``ttl_rmsnorm_bw`` (needs ``ttl`` + this repo's ``ttlang`` on ``sys.path``)."""
    _ensure_ttl_package_path()
    import ttl_rmsnorm_bw  # noqa: PLC0415

    return ttl_rmsnorm_bw


def _import_ttl_rmsnorm_bw_all_in_l1():
    """Import ``ttl_rmsnorm_bw_all_in_l1`` (DFB / all-in-L1 TTL backward; needs ``ttl``)."""
    _ensure_ttl_package_path()
    import ttl_rmsnorm_bw_all_in_l1  # noqa: PLC0415

    return ttl_rmsnorm_bw_all_in_l1


def _import_ttl_rmsnorm_bw_2pass():
    """Import ``ttl_rmsnorm_bw_2pass`` (2-pass tile-streaming TTL backward; needs ``ttl``)."""
    _ensure_ttl_package_path()
    import ttl_rmsnorm_bw_2pass  # noqa: PLC0415

    return ttl_rmsnorm_bw_2pass


import ttml
from ttml.autograd import AutoContext, Tensor, create_tensor


def _open_mesh_for_kernel_bw(ctx: AutoContext, bw_kernel: str) -> None:
    """Open a single-device mesh ``(1, 1)``; TTL reserves worker L1 for large kernel configs."""
    if bw_kernel not in ("ttl", "ttl_all_in_l1", "ttl_2pass"):
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
    # ([1, 1, 256, 2048], "256x2048"),
    # ([1, 1, 256, 11008], "1024x2048"),
    # ([2, 1, 1024, 2048], "2048x2048"),
    # ([2, 1, 1024, 5632], "2048x5632"),
    # ([4, 1, 1024, 4096], "4096x4096"),
    # ([4, 1, 1024, 11008], "4096x11008"),
    # ([8, 1, 1024, 8192], "8192x8192"),
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

    ttl_mod = _import_ttl_rmsnorm_bw() if bw_kernel in ("ttl", "ttl_all_in_l1", "ttl_2pass") else None
    all_in_l1_mod = _import_ttl_rmsnorm_bw_all_in_l1() if bw_kernel == "ttl_all_in_l1" else None
    twopass_mod = _import_ttl_rmsnorm_bw_2pass() if bw_kernel == "ttl_2pass" else None

    ctx = AutoContext.get_instance()
    _open_mesh_for_kernel_bw(ctx, bw_kernel)
    try:
        mesh = ctx.get_device()
        mesh.enable_program_cache()

        if bw_kernel == "metal":
            tag = "Metal"
        elif bw_kernel == "ttl_all_in_l1":
            tag = "TTL (all-in-L1)"
        elif bw_kernel == "ttl_2pass":
            tag = "TTL (2-pass)"
        else:
            tag = "TTL (col-split)"
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

            elif bw_kernel == "ttl":
                assert ttl_mod is not None
                rows = b * s_len
                columns = c
                x2, dL2, g2, r2 = _ttl_rmsnorm_bw_numpy_to_2d(x_np, d_np, g_np, rms_np, rows, columns)
                k, cfg = ttl_mod.make_kernel_for_shape(rows, columns)
                _, _, _, _, ht_p, wt_p = cfg
                rows_p, cols_p = ht_p * TILE, wt_p * TILE
                x_p, g_p, rms_p, dL_p, out_da, out_dg = _ttl_rmsnorm_bw_tensors_to_padded_device(
                    ttl_mod, mesh, x2, g2, r2, dL2, rows_p, cols_p
                )

                def fn(x, g, r, d):
                    return k(x, g, r, d, out_da, out_dg)

                def run_step() -> None:
                    fn(x_p, g_p, rms_p, dL_p)
                    ttnn.synchronize_device(mesh)

            elif bw_kernel == "ttl_all_in_l1":
                assert ttl_mod is not None
                assert all_in_l1_mod is not None
                rows = b * s_len
                columns = c
                x2, dL2, g2, r2 = _ttl_rmsnorm_bw_numpy_to_2d(x_np, d_np, g_np, rms_np, rows, columns)
                _, cfg = ttl_mod.make_kernel_for_shape(rows, columns)
                _, _, _, _, ht_p, wt_p = cfg
                rows_p, cols_p = ht_p * TILE, wt_p * TILE
                x_p, g_p, rms_p, dL_p, out_da, out_dg = _ttl_rmsnorm_bw_tensors_to_padded_device(
                    ttl_mod, mesh, x2, g2, r2, dL2, rows_p, cols_p
                )
                k_l1 = all_in_l1_mod.make_kernel()

                def run_step() -> None:
                    k_l1(x_p, g_p, rms_p, dL_p, out_da, out_dg)
                    ttnn.synchronize_device(mesh)

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
            elif bw_kernel == "ttl_all_in_l1":
                suffix = "KernelBw_TTL_AllInL1"
            elif bw_kernel == "ttl_2pass":
                suffix = "KernelBw_TTL_2Pass"
            else:
                suffix = "KernelBw_TTL"
            row = f"{name}_{suffix}"
            print(f"  {row:28}  Time_us={avg_s * 1e6:10.2f}")
    finally:
        ctx.reset_graph()
        ctx.close_device()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bw-kernel",
        choices=("metal", "ttl", "ttl_all_in_l1", "ttl_2pass"),
        default="metal",
        help=(
            "metal: ttml rmsnorm_bw; ttl: column-split ``ttl_rmsnorm_bw.py``; "
            "ttl_all_in_l1: DFB all-in-L1 ``ttl_rmsnorm_bw_all_in_l1.py``; "
            "ttl_2pass: tile-streaming ``ttl_rmsnorm_bw_2pass.py``. "
            "TTL modes need ``ttl`` (+ ``tt-lang`` on path)."
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
