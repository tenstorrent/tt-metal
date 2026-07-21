# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generalized matmul program-config sweep — a CLI tool.

Give it M, K, N (the GEMM dims, A[M,K] @ B[K,N] = C[M,N]) and it sweeps the
program configuration, memory configuration, math fidelity and dest/accumulation
modes for one or more matmul implementations, timing each on device and checking
PCC against a torch reference. It prints a RESULT line per config, writes a CSV,
and reports the BEST (fastest PCC-passing) config per implementation.

Implementations (choose with --impl, default: all):
  minmatmul  ttnn.experimental.minimal_matmul  (MinimalMatmulConfig: M/K/N block + subblock)
  matmul2d   ttnn.matmul + MatmulMultiCoreReuseMultiCastProgramConfig   (2D mcast)
  matmul1d   ttnn.matmul + MatmulMultiCoreReuseMultiCast1DProgramConfig (1D mcast)

The sweep OWNS config generation: from M,K,N and the core grid it derives valid
per-core / block sizes, enumerates the tunable block + subblock params, and skips
combinations that violate the known hardware constraints (DST-register cap,
divisibility) up front so the run isn't just a wall of failures.

Examples
  # BGE-M3 MLP Wi shape (M=12*8192, K=1024, N=4096), all impls, default sweep
  python tests/generalize/matmul_sweep.py --M 98304 --K 1024 --N 4096

  # Just minimal_matmul, wider block sweep, save CSV
  python tests/generalize/matmul_sweep.py --M 98304 --K 4096 --N 1024 \
      --impl minmatmul --csv tests/generalize/out/mlpwo.csv

  # Fixed fidelity/dtype, cap the number of configs tried
  python tests/generalize/matmul_sweep.py --M 8192 --K 8192 --N 8192 \
      --dtypes bfloat8_b --fidelities LoFi --max-configs 40
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

import ttnn

TILE = 32

DTYPES = {
    "bfloat16": ttnn.bfloat16,
    "bfloat8_b": ttnn.bfloat8_b,
    "bfloat4_b": ttnn.bfloat4_b,
}
FIDELITIES = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}
MEMCFGS = {
    "dram": ttnn.DRAM_MEMORY_CONFIG,
    "l1": ttnn.L1_MEMORY_CONFIG,
}


# ----------------------------------------------------------------------------
# Config generation helpers
# ----------------------------------------------------------------------------
def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def subblock_candidates(block_h: int, block_w: int, dst_cap: int) -> list[tuple[int, int]]:
    """(subblock_h, subblock_w) with h*w <= dst_cap, h | block_h, w | block_w.

    dst_cap is the number of output tiles that fit in the DST register per
    acquire: 8 for bf16/bf8 dest, 4 when fp32_dest_acc_en=True (fp32 halves it).
    Larger subblocks amortize the acquire/release + math-init overhead.
    """
    out = []
    for h in divisors(block_h):
        for w in divisors(block_w):
            if h * w <= dst_cap and h * w >= 1:
                out.append((h, w))
    # Prefer larger subblocks first (usually faster) so early results are good.
    out.sort(key=lambda hw: -(hw[0] * hw[1]))
    return out


def dst_cap_for(fp32_dest_acc_en: bool) -> int:
    return 4 if fp32_dest_acc_en else 8


@dataclass
class Trial:
    impl: str
    label: str
    program_config: object
    out_memcfg_name: str
    dtype_name: str
    fidelity_name: str
    fp32_dest_acc_en: bool
    packer_l1_acc: bool
    est_out_cb_bytes: int = 0
    extra: dict = field(default_factory=dict)


def gen_minmatmul(M_t, K_t, N_t, grid, args, dtype_name, fid_name, fp32_dest, packer_l1):
    """MinimalMatmulConfig: block sizes are in TILES; factory splits M/N across
    the grid and iterates ceil(per_core / block) blocks. Bigger M/K block cuts
    iteration overhead but grows the double-buffered CB L1 footprint."""
    cap = dst_cap_for(fp32_dest)
    m_blocks = [b for b in args.m_blocks if b <= max(M_t, 1)]
    k_blocks = [b for b in args.k_blocks if b <= max(K_t, 1)]
    n_blocks = [b for b in args.n_blocks if b <= max(N_t, 1)]
    trials = []
    for mb, kb, nb in itertools.product(m_blocks, k_blocks, n_blocks):
        for sh, sw in subblock_candidates(mb, nb, cap):
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
            )
            tile_bytes = 2048 if dtype_name == "bfloat16" else (1088 if dtype_name == "bfloat8_b" else 576)
            est = 2 * (mb * kb + kb * nb + mb * nb) * tile_bytes  # double-buffered CBs
            trials.append(
                Trial(
                    "minmatmul",
                    f"m{mb}k{kb}n{nb}_sb{sh}x{sw}",
                    cfg,
                    "?",
                    dtype_name,
                    fid_name,
                    fp32_dest,
                    packer_l1,
                    est_out_cb_bytes=est,
                    extra={"M_block": mb, "K_block": kb, "N_block": nb, "sbh": sh, "sbw": sw},
                )
            )
    return trials


def _percore_and_subblocks(M_t, N_t, grid, cap, transpose_mcast):
    """per_core_M/N from grid + valid out_subblock candidates."""
    gx, gy = grid
    # 2D convention: M parallelised on grid rows (y), N on grid cols (x)
    cores_m = gx if transpose_mcast else gy
    cores_n = gy if transpose_mcast else gx
    per_core_M = max(1, -(-M_t // cores_m))  # ceil
    per_core_N = max(1, -(-N_t // cores_n))
    subs = subblock_candidates(per_core_M, per_core_N, cap)
    return per_core_M, per_core_N, subs


def gen_matmul2d(M_t, K_t, N_t, grid, args, dtype_name, fid_name, fp32_dest, packer_l1):
    cap = dst_cap_for(fp32_dest)
    trials = []
    for transpose_mcast in (False, True):
        per_core_M, per_core_N, subs = _percore_and_subblocks(M_t, N_t, grid, cap, transpose_mcast)
        for in0_block_w in [w for w in divisors(K_t) if w in args.in0_block_w or not args.in0_block_w]:
            for sh, sw in subs:
                if per_core_M % sh or per_core_N % sw:
                    continue
                try:
                    cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
                        in0_block_w=in0_block_w,
                        out_subblock_h=sh,
                        out_subblock_w=sw,
                        per_core_M=per_core_M,
                        per_core_N=per_core_N,
                        transpose_mcast=transpose_mcast,
                        fuse_batch=True,
                    )
                except Exception:
                    continue
                trials.append(
                    Trial(
                        "matmul2d",
                        f"pcM{per_core_M}pcN{per_core_N}_ib{in0_block_w}_sb{sh}x{sw}_t{int(transpose_mcast)}",
                        cfg,
                        "?",
                        dtype_name,
                        fid_name,
                        fp32_dest,
                        packer_l1,
                        extra={
                            "per_core_M": per_core_M,
                            "per_core_N": per_core_N,
                            "in0_block_w": in0_block_w,
                            "sbh": sh,
                            "sbw": sw,
                            "transpose_mcast": transpose_mcast,
                        },
                    )
                )
    return trials


def gen_matmul1d(M_t, K_t, N_t, grid, args, dtype_name, fid_name, fp32_dest, packer_l1):
    cap = dst_cap_for(fp32_dest)
    gx, gy = grid
    num_cores = gx * gy
    trials = []
    for mcast_in0 in (True, False):
        # 1D lays all cores in a line. mcast_in0=True: split N across cores, M
        # replicated per core. mcast_in0=False (gather): split M across cores.
        if mcast_in0:
            per_core_M = max(1, M_t)
            per_core_N = max(1, -(-N_t // num_cores))
        else:
            per_core_M = max(1, -(-M_t // num_cores))
            per_core_N = max(1, N_t)
        subs = subblock_candidates(per_core_M, per_core_N, cap)
        for in0_block_w in [w for w in divisors(K_t) if w in args.in0_block_w or not args.in0_block_w]:
            for sh, sw in subs:
                if per_core_M % sh or per_core_N % sw:
                    continue
                try:
                    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
                        in0_block_w=in0_block_w,
                        out_subblock_h=sh,
                        out_subblock_w=sw,
                        per_core_M=per_core_M,
                        per_core_N=per_core_N,
                        fuse_batch=True,
                        mcast_in0=mcast_in0,
                    )
                except Exception:
                    continue
                trials.append(
                    Trial(
                        "matmul1d",
                        f"pcM{per_core_M}pcN{per_core_N}_ib{in0_block_w}_sb{sh}x{sw}_mc{int(mcast_in0)}",
                        cfg,
                        "?",
                        dtype_name,
                        fid_name,
                        fp32_dest,
                        packer_l1,
                        extra={
                            "per_core_M": per_core_M,
                            "per_core_N": per_core_N,
                            "in0_block_w": in0_block_w,
                            "sbh": sh,
                            "sbw": sw,
                            "mcast_in0": mcast_in0,
                        },
                    )
                )
    return trials


GENERATORS = {"minmatmul": gen_minmatmul, "matmul2d": gen_matmul2d, "matmul1d": gen_matmul1d}


# ----------------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------------
def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else (torch.dot(a, b).item() / denom)


_OOM_MARKERS = (
    "out of memory",
    "oom",
    "clash with l1",
    "beyond max l1",
    "circular buffer",
    "bank_manager",
    "allocator",
    "not enough space",
    "failed to allocate",
)


def _looks_like_oom(exc: BaseException) -> bool:
    """Classify an exception as an L1/DRAM capacity failure for reporting."""
    s = str(exc).lower()
    return any(m in s for m in _OOM_MARKERS)


def _device_alive(dev) -> bool:
    """Cheap probe: round-trip a tiny tensor. Returns False if the device is
    wedged/unresponsive so the sweep can abort cleanly instead of spinning."""
    try:
        x = ttnn.from_torch(
            torch.zeros(TILE, TILE, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.synchronize_device(dev)
        _safe_dealloc(x)
        return True
    except Exception:
        return False


def _safe_dealloc(t):
    """Deallocate a device tensor, swallowing any error (already-freed, bad
    state, etc.) so cleanup never masks the real failure."""
    if t is None:
        return
    try:
        ttnn.deallocate(t)
    except Exception:
        pass


def run_trial(dev, a_t, b_t, ref, trial: Trial, out_memcfg_name, iters):
    dtype = DTYPES[trial.dtype_name]
    memcfg = MEMCFGS[out_memcfg_name]
    a = b = None
    outs = []  # track every output so we can free them even on mid-loop failure
    try:
        a = ttnn.from_torch(
            a_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        b = ttnn.from_torch(
            b_t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ck = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=FIDELITIES[trial.fidelity_name],
            math_approx_mode=False,
            fp32_dest_acc_en=trial.fp32_dest_acc_en,
            packer_l1_acc=trial.packer_l1_acc,
        )

        def once():
            if trial.impl == "minmatmul":
                return ttnn.experimental.minimal_matmul(
                    input_tensor=a,
                    weight_tensor=b,
                    bias_tensor=None,
                    config=trial.program_config,
                    memory_config=memcfg,
                    dtype=dtype,
                    compute_kernel_config=ck,
                )
            return ttnn.matmul(
                a, b, program_config=trial.program_config, memory_config=memcfg, compute_kernel_config=ck
            )

        # correctness (warm + compile). A bad program config / L1 OOM / TT_FATAL
        # throws HERE (at program build, before launch) -> caught by the caller.
        out = once()
        outs.append(out)
        ttnn.synchronize_device(dev)
        got = ttnn.to_torch(out).to(torch.float32).reshape(ref.shape)
        p = pcc(ref, got)
        _safe_dealloc(out)
        outs.clear()

        # timing (median of iters)
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = once()
            outs.append(out)
            ttnn.synchronize_device(dev)
            samples.append((time.perf_counter() - t0) * 1e3)
            _safe_dealloc(out)
            outs.clear()
        samples.sort()
        return samples[len(samples) // 2], samples[0], p
    finally:
        # Always free inputs + any straggler outputs so a failed trial does not
        # leak L1/DRAM and cause phantom OOMs on the configs that follow.
        for o in outs:
            _safe_dealloc(o)
        _safe_dealloc(a)
        _safe_dealloc(b)


def main():
    ap = argparse.ArgumentParser(description="Generalized matmul program-config sweep")
    ap.add_argument("--M", type=int, required=True, help="rows of A / output (elements)")
    ap.add_argument("--K", type=int, required=True, help="contraction dim (elements)")
    ap.add_argument("--N", type=int, required=True, help="cols of B / output (elements)")
    ap.add_argument(
        "--impl",
        nargs="+",
        default=["minmatmul", "matmul2d", "matmul1d"],
        choices=list(GENERATORS.keys()),
        help="which implementations to sweep",
    )
    ap.add_argument("--dtypes", nargs="+", default=["bfloat8_b"], choices=list(DTYPES.keys()))
    ap.add_argument("--fidelities", nargs="+", default=["LoFi", "HiFi2"], choices=list(FIDELITIES.keys()))
    ap.add_argument("--out-memcfgs", nargs="+", default=["dram", "l1"], choices=list(MEMCFGS.keys()))
    ap.add_argument(
        "--fp32-dest", nargs="+", type=int, default=[0], choices=[0, 1], help="fp32_dest_acc_en values to sweep (0/1)"
    )
    ap.add_argument("--packer-l1-acc", nargs="+", type=int, default=[1], choices=[0, 1])
    ap.add_argument("--grid", type=int, nargs=2, default=[8, 8], help="core grid x y")
    ap.add_argument("--iters", type=int, default=3, help="timing samples per config (median reported)")
    ap.add_argument("--max-configs", type=int, default=200, help="cap total trials")
    ap.add_argument("--pcc", type=float, default=0.99, help="PCC threshold to count as passing")
    ap.add_argument("--m-blocks", type=int, nargs="+", default=[8, 16, 24, 32], help="minmatmul M_block tiles")
    ap.add_argument("--k-blocks", type=int, nargs="+", default=[4, 8, 16, 32], help="minmatmul K_block tiles")
    ap.add_argument("--n-blocks", type=int, nargs="+", default=[2, 4, 8], help="minmatmul N_block tiles")
    ap.add_argument(
        "--in0-block-w", type=int, nargs="*", default=[], help="fix matmul1d/2d in0_block_w (tiles); empty=all divisors"
    )
    ap.add_argument("--csv", type=str, default="", help="write results CSV to this path")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.M % TILE or args.K % TILE or args.N % TILE:
        raise SystemExit(f"M,K,N must be multiples of {TILE} (tile dim). Got {args.M},{args.K},{args.N}")
    M_t, K_t, N_t = args.M // TILE, args.K // TILE, args.N // TILE
    print(
        f"# GEMM A[{args.M},{args.K}] @ B[{args.K},{args.N}] = C[{args.M},{args.N}]  "
        f"({M_t}x{K_t}x{N_t} tiles), grid={args.grid}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    a_t = torch.randn(args.M, args.K, dtype=torch.bfloat16)
    b_t = torch.randn(args.K, args.N, dtype=torch.bfloat16)
    ref = a_t.to(torch.float32) @ b_t.to(torch.float32)

    # Build the trial list across all axes.
    trials: list[Trial] = []
    for impl in args.impl:
        for dt, fid, fp32, pl1 in itertools.product(args.dtypes, args.fidelities, args.fp32_dest, args.packer_l1_acc):
            trials += GENERATORS[impl](M_t, K_t, N_t, tuple(args.grid), args, dt, fid, bool(fp32), bool(pl1))
    # Expand over output memcfgs (cheap axis, keep last so program configs cluster).
    expanded = []
    for t in trials:
        for mc in args.out_memcfgs:
            t2 = Trial(**{**t.__dict__})
            t2.out_memcfg_name = mc
            expanded.append(t2)
    trials = expanded
    if len(trials) > args.max_configs:
        print(f"# generated {len(trials)} trials, capping to --max-configs={args.max_configs}", flush=True)
        trials = trials[: args.max_configs]
    else:
        print(f"# generated {len(trials)} trials", flush=True)

    def _base_row(t, status, note="", **num):
        return {
            "impl": t.impl,
            "label": t.label,
            "dtype": t.dtype_name,
            "fidelity": t.fidelity_name,
            "fp32_dest": int(t.fp32_dest_acc_en),
            "packer_l1_acc": int(t.packer_l1_acc),
            "out_memcfg": t.out_memcfg_name,
            "status": status,
            "est_out_cb_bytes": t.est_out_cb_bytes,
            "median_ms": "",
            "min_ms": "",
            "pcc": "",
            "note": note,
            **num,
            **t.extra,
        }

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    rows = []
    best = {}
    n_ok = n_skip = n_fatal = 0
    try:
        for i, t in enumerate(trials):
            full = (
                f"{t.impl}|{t.label}|{t.dtype_name}|{t.fidelity_name}|"
                f"fp32d{int(t.fp32_dest_acc_en)}|pl1{int(t.packer_l1_acc)}|out_{t.out_memcfg_name}"
            )
            try:
                med, mn, p = run_trial(dev, a_t, b_t, ref, t, t.out_memcfg_name, args.iters)
                status = "PASS" if p >= args.pcc else "LOWPCC"
                print(f"RESULT {full} median_ms={med:.3f} min_ms={mn:.3f} pcc={p:.4f} [{status}]", flush=True)
                rows.append(_base_row(t, status, median_ms=round(med, 3), min_ms=round(mn, 3), pcc=round(p, 4)))
                n_ok += 1
                if status == "PASS" and (t.impl not in best or med < best[t.impl][0]):
                    best[t.impl] = (med, full)
            except KeyboardInterrupt:
                print("\n# interrupted by user - writing partial results", flush=True)
                break
            except (RuntimeError, ValueError, MemoryError, RuntimeWarning) as exc:
                # Expected, recoverable per-config failures: L1/DRAM OOM, bad
                # program config, TT_FATAL/TT_THROW at program build, PCC-time
                # reshape mismatch. These throw at COMPILE (before launch), so the
                # device is not wedged - log and keep sweeping.
                msg = f"{type(exc).__name__}: {str(exc)[:180]}"
                kind = "OOM" if _looks_like_oom(exc) else "SKIP"
                print(f"{kind} {full}: {msg}", flush=True)
                rows.append(_base_row(t, kind, note=msg))
                n_skip += 1
            except BaseException as exc:  # noqa: BLE001 - last-resort guard
                # Anything unexpected (e.g. a hard device error). Record it,
                # probe device health, and continue if the device still responds.
                msg = f"{type(exc).__name__}: {str(exc)[:180]}"
                print(f"FATAL {full}: {msg}", flush=True)
                rows.append(_base_row(t, "FATAL", note=msg))
                n_fatal += 1
                if not _device_alive(dev):
                    print("# device no longer responsive - aborting sweep, writing partial results", flush=True)
                    break
    finally:
        try:
            ttnn.close_mesh_device(dev)
        except Exception:
            pass
    print(f"\n# trials: {n_ok} ran, {n_skip} skipped/oom, {n_fatal} fatal", flush=True)

    print("\n# ==== BEST (fastest PCC-passing) per implementation ====")
    for impl in args.impl:
        if impl in best:
            print(f"BEST {impl}: median_ms={best[impl][0]:.3f}  {best[impl][1]}")
        else:
            print(f"BEST {impl}: (no PCC-passing config)")

    if args.csv:
        p = Path(args.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        cols = sorted({k for r in rows for k in r})
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"# wrote {len(rows)} rows to {p}")


if __name__ == "__main__":
    main()
