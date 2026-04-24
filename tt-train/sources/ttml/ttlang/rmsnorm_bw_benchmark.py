# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm microbenchmark in Python using nanobind-backed ``ttml`` ops.

Metal ``rmsnorm_bw`` is run with ``max_num_cores=1`` to match the TT-Lang kernel
(``ttl.operation(grid=(1, 1))`` in ``ttl_rmsnorm_bw.py``).

    To build tt-train on tt-lang ird image:
    1) B="$(pwd)/build_Release"; export LD_LIBRARY_PATH="$B:$B/tt_metal:$B/lib:$B/tt_stl:$B/tt_metal/third_party/umd/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; cd $B/tt-train && python3 setup.py install
    2) ./build_metal.sh -b Release --build-tt-train

    One shot command to run the benchmark on IRD with tt-lang image where tt-metal repo is mounted at /localdev/$(whoami)/tt-metal
    cd /localdev/$(whoami)/tt-metal && R="$PWD" && export TT_METAL_HOME="$R" TT_METAL_LOGGER_LEVEL=FATAL TT_METAL_RUNTIME_ROOT="$R" TT_METAL_DISABLE_PRECOMPILED_FW=1 PYTHONPATH="$R/ttnn:$R/tools:$R:$R/tt-train/sources/ttml:$R/build_Release/tt-train/sources/ttml" LD_LIBRARY_PATH="$R/build_Release/lib" && python3 tt-train/sources/ttml/ttlang/rmsnorm_bw_benchmark.py

    Current results:
TTML:
    256x384_KernelBw_Metal        Time_us=    158.08  GB/s=   3.744  Elems_M=0.0983
    256x2048_KernelBw_Metal       Time_us=    246.02  GB/s=  12.822  Elems_M=0.5243
    1024x2048_KernelBw_Metal      Time_us=    283.80  GB/s=  44.374  Elems_M=2.0972
    2048x2048_KernelBw_Metal      Time_us=    373.95  GB/s=  67.331  Elems_M=4.1943
    2048x5632_KernelBw_Metal      Time_us=    825.47  GB/s=  83.871  Elems_M=11.5343
    4096x4096_KernelBw_Metal      Time_us=   1189.35  GB/s=  84.658  Elems_M=16.7772
    4096x11008_KernelBw_Metal     Time_us=   3447.24  GB/s=  78.493  Elems_M=45.0888
    8192x8192_KernelBw_Metal      Time_us=   4465.03  GB/s=  90.190  Elems_M=67.1089

TTLANG:
    256x384_KernelBw_TTL          Time_us=    249.63  GB/s=   2.371  Elems_M=0.0983
    256x2048_KernelBw_TTL         Time_us=    250.13  GB/s=  12.611  Elems_M=0.5243
    1024x2048_KernelBw_TTL        Time_us=    393.97  GB/s=  31.965  Elems_M=2.0972
    2048x2048_KernelBw_TTL        Time_us=    506.14  GB/s=  49.745  Elems_M=4.1943
    2048x5632_KernelBw_TTL        Time_us=    817.26  GB/s=  84.713  Elems_M=11.5343
    4096x4096_KernelBw_TTL        Time_us=   1033.92  GB/s=  97.385  Elems_M=16.7772
    4096x11008_KernelBw_TTL       Time_us=   2338.71  GB/s= 115.698  Elems_M=45.0888
    8192x8192_KernelBw_TTL        Time_us=   3380.75  GB/s= 119.116  Elems_M=67.1089


"""


from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import ttnn
TILE = 32

_TTL_WORKER_L1_RESERVE_BYTES = 77824

_TTML = Path(__file__).resolve().parents[2] / "sources" / "ttml"
_TTLANG = _TTML / "ttlang"
if _TTLANG.is_dir() and str(_TTLANG) not in sys.path:
    sys.path.insert(0, str(_TTLANG))


def _ensure_ttl_package_path() -> None:
    """Restore ``ttl`` when PYTHONPATH omits the TT-Lang toolchain."""
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
    """Import ``ttl_rmsnorm_bw`` (needs ``ttl`` + repo ``tt-lang`` on ``sys.path``)."""
    _ensure_ttl_package_path()
    import ttl_rmsnorm_bw  # noqa: PLC0415

    return ttl_rmsnorm_bw


from ttml.autograd import AutoContext, Tensor, create_tensor
from ttml import ops


def _open_mesh_for_kernel_bw(ctx: AutoContext, bw_kernel: str) -> None:
    """Open mesh; TTL paths reserve L1 so the kernel-config ringbuffer fits large TTL programs."""
    if bw_kernel != "ttl":
        ctx.open_device((1, 1))
        return
    max_l1 = ttnn.device.get_max_worker_l1_unreserved_size()
    print(f"max_l1: {max_l1}")
    print(f"_TTL_WORKER_L1_RESERVE_BYTES: {_TTL_WORKER_L1_RESERVE_BYTES}")
    worker_l1 = max_l1 - _TTL_WORKER_L1_RESERVE_BYTES
    try:
        ctx.open_device((1, 1), None, worker_l1)
    except TypeError:
        ctx.open_device((1, 1))


RMSNORM_EPS = 0.0078125
NUM_WARMUP = 5
NUM_MEASURE = 50

# Match ``ttl_rmsnorm_bw`` ``@ttl.operation(grid=(1, 1))`` when benchmarking vs ttml metal path.
KERNEL_BW_TTML_MAX_CORES = 0

SHAPES = (
    # ([1, 1, 256, 384], "256x384"),
    # ([1, 1, 256, 2048], "256x2048"),
    # ([1, 1, 256, 11008], "1024x2048"),
    # ([2, 1, 1024, 2048], "2048x2048"),
    # ([2, 1, 1024, 5632], "2048x5632"),
    # ([4, 1, 1024, 4096], "4096x4096"),
    # ([4, 1, 1024, 11008], "4096x11008"),
    ([8, 1, 1024, 8192], "8192x8192"),
)


def _torch_rmsnorm_rms_numpy(x_np: np.ndarray, eps: float) -> np.ndarray:
    """``sqrt(mean(x^2, dim=C, keepdim) + eps)`` → ``[B,1,S,1]`` float32, same reduction as ``rmsnorm_op`` composite."""
    x = torch.from_numpy(np.asarray(x_np, dtype=np.float32))
    mean_sq = (x * x).mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    return np.asarray(rms.numpy(), dtype=np.float32)


def _seed_for_name(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


def _default_jit_cache_dir() -> Path:
    env = os.environ.get("TT_METAL_CACHE", "").strip()
    if env:
        return Path(env).expanduser()
    home = Path.home()
    if home.exists():
        return home / ".cache" / "tt-metal-cache"
    return Path("/tmp/tt-metal-cache")


def _warn_jit_cache_disk_space() -> None:
    """Best-effort: warn if the volume backing the JIT cache is nearly full."""
    cache = _default_jit_cache_dir()
    try:
        parent = cache if cache.is_dir() else cache.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(str(parent))
    except OSError:
        return
    free_gib = usage.free / (1024**3)
    total_gib = usage.total / (1024**3)
    used_pct = 100.0 * (1.0 - usage.free / max(usage.total, 1))
    if free_gib < 2.0 or used_pct > 92.0:
        print(
            f"Warning: low free space for JIT cache (~{free_gib:.1f} GiB free on volume of "
            f"~{total_gib:.1f} GiB; cache dir `{cache}`). New shapes may fail with ENOSPC; "
            "free disk or set TT_METAL_CACHE. See docstring.\n",
            file=sys.stderr,
        )


def _make_ttml_bw_fn():
    """Return ``(x, g, r, d) -> (d_in, d_gamma)`` via metal ``rmsnorm_bw`` with ``max_num_cores`` cap."""
    m = ops.rmsnorm
    fn = getattr(m, "rmsnorm_bw", None)
    if fn is None:
        raise RuntimeError(
            "ttml.ops.rmsnorm has neither rmsnorm_bw nor metal_bw. Rebuild _ttml (tt-train) after nb_ops.cpp changes."
        )

    def call(x, g, r, d):
        return fn(x, g, r, d, KERNEL_BW_TTML_MAX_CORES)

    return call


def _jit_disk_help_if_enospc(exc: BaseException) -> None:
    text = str(exc).lower()
    if (
        "no space left on device" not in text
        and "enospc" not in text
        and "cannot map elf" not in text
        and "failed to rename temporary file" not in text
    ):
        return
    cache = _default_jit_cache_dir()
    print(
        "\nLikely cause: disk (or inode) exhaustion on the kernel JIT cache volume.\n"
        f"  Default cache: {cache}\n"
        "  Try: free space, then `rm -rf` stale trees under that path, or:\n"
        f"    export TT_METAL_CACHE=/path/with/plenty/of/space\n",
        file=sys.stderr,
    )


def _run_kernel_bw_only(bw_kernel: str = "metal") -> None:
    """RMS from host Torch; each timed step: ``rmsnorm_bw`` or TTL backward kernel + sync."""
    if bw_kernel not in ("metal", "ttl"):
        raise ValueError(f"bw_kernel must be 'metal' or 'ttl', got {bw_kernel!r}")
    ttl_mod = _import_ttl_rmsnorm_bw() if bw_kernel == "ttl" else None
    ttml_bw_fn = _make_ttml_bw_fn() if bw_kernel == "metal" else None

    ctx = AutoContext.get_instance()
    _open_mesh_for_kernel_bw(ctx, bw_kernel)
    try:
        mesh = ctx.get_device()
        mesh.enable_program_cache()

        if bw_kernel == "metal":
            tag = "Metal (rmsnorm_bw, 1c)"
        else:
            tag = "TTL (ttl_rmsnorm_bw.py, 1c)"
        print(f"Mode: KernelBw / {tag} (backward-only, RMS from torch forward on host)\n")
        if bw_kernel == "ttl":
            print(
                "[kernel_bw] ttl: ttl.operation(grid=(1, 1)) — 1 logical grid core per device launch.\n",
                flush=True,
            )

        for shape, name in SHAPES:
            seed = _seed_for_name(name)
            rng = np.random.default_rng(seed)
            b, _, s_len, c = shape
            x_np = rng.uniform(-1.0, 1.0, (b, 1, s_len, c)).astype(np.float32)
            g_np = rng.uniform(-1.0, 1.0, (1, 1, 1, c)).astype(np.float32)
            d_np = rng.uniform(-1.0, 1.0, (b, 1, s_len, c)).astype(np.float32)

            x_storage = Tensor.from_numpy(x_np, ttnn.Layout.TILE)
            g_storage = Tensor.from_numpy(g_np, ttnn.Layout.TILE)
            d_storage = Tensor.from_numpy(d_np, ttnn.Layout.TILE)

            x_t = create_tensor(x_storage.get_value(), False)
            g_t = create_tensor(g_storage.get_value(), False)
            dL_t = create_tensor(d_storage.get_value(), False)

            rms_np = _torch_rmsnorm_rms_numpy(x_np, RMSNORM_EPS)
            rms_storage = Tensor.from_numpy(rms_np, ttnn.Layout.TILE)
            rms_t = create_tensor(rms_storage.get_value(), False)

            elems_in = b * s_len * c
            elems_gamma = c
            elems_rms = int(rms_t.get_value().logical_volume())
            elems_dout = elems_in
            elems_din = elems_in
            elems_dgamma = elems_gamma
            total_elems = elems_in + elems_gamma + elems_rms + elems_dout + elems_din + elems_dgamma
            total_dram_bytes = total_elems * 2  # bf16

            ttl_device_tensors: list[object] = []
            if bw_kernel == "metal":
                assert ttml_bw_fn is not None

                def run_step() -> None:
                    d_in, d_gamma = ttml_bw_fn(x_t, g_t, rms_t, dL_t)
                    ttnn.synchronize_device(mesh)
                    d_in.deallocate_storage()
                    d_gamma.deallocate_storage()

            elif bw_kernel == "ttl":
                assert ttl_mod is not None
                rows = b * s_len
                xv = x_t.get_value()
                gv = g_t.get_value()
                rv = rms_t.get_value()
                dv = dL_t.get_value()
                x2 = ttnn.reshape(xv, (rows, c))
                dL2 = ttnn.reshape(dv, (rows, c))
                g4 = ttnn.reshape(gv, (1, 1, 1, c))
                g_rep = ttnn.repeat(g4, (rows, 1, 1, 1))
                g2 = ttnn.reshape(g_rep, (rows, c))
                r4 = ttnn.repeat(rv, (1, 1, 1, c))
                r2 = ttnn.reshape(r4, (rows, c))
                z = torch.zeros((rows, c), dtype=torch.bfloat16)
                out_da = ttnn.from_torch(
                    z,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                out_dg = ttnn.from_torch(
                    torch.zeros((rows, c), dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttl_device_tensors.extend((x2, dL2, g4, g_rep, r4, g2, r2, out_da, out_dg))
                k, cfg = ttl_mod.make_kernel_for_shape(rows, c)
                cc, rc, cpc, rpc, ht_p, wt_p = cfg
                rows_p, cols_p = ht_p * TILE, wt_p * TILE
                x_p = ttl_mod._to_dev(ttl_mod._pad(ttnn.to_torch(x2), rows_p, cols_p), mesh)
                g_p = ttl_mod._to_dev(ttl_mod._pad(ttnn.to_torch(g2), rows_p, cols_p), mesh)
                r2_t = ttnn.to_torch(r2)
                rms_p = ttl_mod._to_dev(
                    ttl_mod._pad_rms_for_kernel(r2_t, rows_p, cols_p, RMSNORM_EPS), mesh
                )
                dL_p = ttl_mod._to_dev(ttl_mod._pad(ttnn.to_torch(dL2), rows_p, cols_p), mesh)
                out_da = ttl_mod._to_dev(torch.zeros(rows_p, cols_p, dtype=torch.bfloat16), mesh)
                out_dg = ttl_mod._to_dev(torch.zeros(rows_p, cols_p, dtype=torch.bfloat16), mesh)
                ttl_device_tensors.extend((x_p, g_p, rms_p, dL_p, out_da, out_dg))

                def fn(x, g, r, d):
                    return k(x, g, r, d, out_da, out_dg)

                def run_step() -> None:
                    fn(x_p, g_p, rms_p, dL_p)
                    ttnn.synchronize_device(mesh)

            for _ in range(NUM_WARMUP):
                run_step()

            total = 0.0
            for _ in range(NUM_MEASURE):
                t0 = time.perf_counter()
                run_step()
                total += time.perf_counter() - t0
            avg_s = total / NUM_MEASURE
            elems_m = elems_in / 1e6
            gb_s = total_dram_bytes / avg_s / 1e9
            if bw_kernel == "metal":
                suffix = "KernelBw_Metal"
            else:
                suffix = "KernelBw_TTL"
            row = f"{name}_{suffix}"
            print(f"  {row:28}  Time_us={avg_s * 1e6:10.2f}  GB/s={gb_s:8.3f}  Elems_M={elems_m:.4f}")

            for t in ttl_device_tensors:
                ttnn.deallocate(t)
            rms_t.deallocate_storage()
            dL_t.deallocate_storage()
            g_t.deallocate_storage()
            x_t.deallocate_storage()
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
            "metal: ttml rmsnorm_bw; ttl: ``ttl_rmsnorm_bw.py``. "
            "TTL needs ``ttl`` (+ ``tt-lang`` on path for ttl)."
        ),
    )
    args = p.parse_args()

    print(f"warmup={NUM_WARMUP}  measure={NUM_MEASURE}  epsilon={RMSNORM_EPS}  --bw-kernel={args.bw_kernel}\n")

    _warn_jit_cache_disk_space()
    _run_kernel_bw_only(args.bw_kernel)
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ImportError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except BaseException as e:
        _jit_disk_help_if_enospc(e)
        raise
