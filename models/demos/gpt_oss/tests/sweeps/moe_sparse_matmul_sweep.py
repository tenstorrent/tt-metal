# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""MoE sparse_matmul core-grid sweep for gpt-oss-20b decode.

Reference: models/demos/gpt_oss/tests/sweeps/matmul_sweep.py (same RESULT/BEST/CSV
structure). This variant targets the *sparse* matmul used by the MoE experts, whose
dominant tunable is the compute grid (decode_gate_up_cores / decode_down_cores in
expert_configs.py).

The decode expert GEMM is, per the real op (GPTOSS_SHAPE_DBG dump):
    hidden_states [1, 1, M=1, K=2880]   (M padded to a tile = 32 rows on device)
    weight        [1, E=32, K=2880, N=2880]
    sparsity      [1, 1, 1, E=32]       (4 of 32 experts active)
    -> Nt = ceil(N/32) = 90

We sweep (core_x, core_y) grids. For a 1D mcast_in0 sparse matmul the kernel needs
    per_core_N = ceil(Nt / num_cores)   and    ceil(Nt / per_core_N) == num_cores
(rectangularity). Grids that violate this are skipped up front.

WHY SUBPROCESS-PER-TRIAL:
    sparse_matmul (mcast_in0) can *deadlock the device* for some configs (e.g. small
    core counts / large per_core_N, or a wrong nnz). A deadlock is NOT a Python
    exception -- it hangs the process on a device sync forever and wedges the board,
    which would otherwise kill the whole sweep. So each grid is run in an isolated
    worker subprocess with a hard wall-clock timeout. If the worker exceeds it we
    treat the config as HANG, reset the board (tt-smi -r), and continue to the next
    grid. One bad config can no longer take down the run.

Usage:
  python models/demos/gpt_oss/tests/sweeps/moe_sparse_matmul_sweep.py --proj gate_up \
      --csv models/demos/gpt_oss/tests/sweeps/out/gate_up.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import ttnn

try:
    from tqdm import tqdm
except Exception:  # tqdm optional
    tqdm = None

TILE = 32

DTYPES = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "bfloat4_b": ttnn.bfloat4_b}
FIDELITIES = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    return 1.0 if denom == 0 else (torch.dot(a, b).item() / denom)


def rectangular(Nt: int, num_cores: int) -> tuple[bool, int]:
    """Return (ok, per_core_N). ok iff ceil(Nt/per_core_N)==num_cores (kernel assert)."""
    if num_cores <= 0:
        return False, 0
    per_core_N = (Nt + num_cores - 1) // num_cores
    ok = ((Nt + per_core_N - 1) // per_core_N) == num_cores
    return ok, per_core_N


def gen_grids(Nt: int, grid_max_x: int, grid_max_y: int, extra):
    """Enumerate (cx, cy) grids whose num_cores gives a rectangular tiling of Nt."""
    seen = set()
    trials = []
    for cy in range(1, grid_max_y + 1):
        for cx in range(1, grid_max_x + 1):
            nc = cx * cy
            if nc in seen:
                continue
            ok, pcn = rectangular(Nt, nc)
            if not ok:
                continue
            seen.add(nc)
            trials.append((cx, cy, nc, pcn))
    # Always include any explicitly requested core counts (as near-square grids).
    for nc in extra:
        if nc in seen:
            continue
        ok, pcn = rectangular(Nt, nc)
        if not ok:
            continue
        cx = min(grid_max_x, nc)
        cy = max(1, math.ceil(nc / cx))
        if cx * cy != nc:
            for a in range(min(grid_max_x, nc), 0, -1):
                if nc % a == 0 and nc // a <= grid_max_y:
                    cx, cy = a, nc // a
                    break
        trials.append((cx, cy, nc, pcn))
        seen.add(nc)
    trials.sort(key=lambda t: t[2])
    return trials


def build_pc(cx, cy, per_core_N, in0_block_w, Kt):
    # snap in0_block_w to a divisor of Kt (kernel asserts Kt % in0_block_w == 0)
    ibw = in0_block_w
    if Kt % ibw != 0:
        divs = [d for d in range(2, ibw + 1) if Kt % d == 0]
        ibw = max(divs) if divs else Kt
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=ibw,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=1,
        per_core_M=1,
        per_core_N=per_core_N,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=True,
    )


def make_inputs(args):
    """Deterministic torch tensors + reference (shared by worker; seeded).

    Two op shapes, matching models/demos/gpt_oss/tt/experts/decode.py:
      gate_up: act=hidden [1,1,1,K] (shared across experts, is_input_b_sparse),
               weight [1,E,K,N]; K=hidden, N=intermediate.
      down:    act=down_input [1,E,1,K] (per-expert, is_input_a_sparse=True),
               weight [1,E,K,N]; K=intermediate, N=hidden.
    """
    E, K, N = args.experts, args.K, args.N
    torch.manual_seed(args.seed)
    weight = torch.randn(1, E, K, N, dtype=torch.bfloat16) * 0.05
    spars = torch.zeros(1, 1, 1, E, dtype=torch.bfloat16)
    spars[..., : args.active] = 1.0
    ref = torch.zeros(args.active, N, dtype=torch.float32)
    if args.proj == "down":
        # Per-expert activation; only active experts carry data.
        act = torch.zeros(1, E, 1, K, dtype=torch.bfloat16)
        act[0, : args.active, 0, :] = torch.randn(args.active, K, dtype=torch.bfloat16) * 0.05
        for e in range(args.active):
            ref[e, :] = act[0, e, 0].to(torch.float32) @ weight[0, e].to(torch.float32)
    else:
        # Shared activation across all experts.
        act = torch.randn(1, 1, 1, K, dtype=torch.bfloat16) * 0.05
        for e in range(args.active):
            ref[e, :] = act[0, 0, 0].to(torch.float32) @ weight[0, e].to(torch.float32)
    return act, weight, spars, ref


# ---------------------------------------------------------------------------
# WORKER: run exactly one grid config, write a JSON result, exit.
# Invoked as a subprocess by the orchestrator so a device deadlock can be
# killed by wall-clock timeout without taking down the whole sweep.
# ---------------------------------------------------------------------------
def run_worker(args):
    result = {"status": "CRASH", "median_ms": "", "min_ms": "", "pcc": "", "note": "worker did not finish"}
    rf = Path(args.result_file)
    try:
        E, K, N = args.experts, args.K, args.N
        Kt, Nt = K // TILE, N // TILE
        act, weight, spars, ref = make_inputs(args)
        is_a_sparse = args.proj == "down"
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        output_tile = ttnn.Tile([32, 32])
        try:
            h_t = ttnn.from_torch(
                act,
                dtype=DTYPES[args.act_dtype],
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            w_t = ttnn.from_torch(
                weight,
                dtype=DTYPES[args.dtype],
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            s_t = ttnn.from_torch(
                spars,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            pc = build_pc(args.cx, args.cy, args.pcn, args.in0_block_w, Kt)

            # nnz semantics (see ttnn.sparse_matmul docs + decode.py):
            #   None -> inferred at runtime (what production uses; robust when the
            #           real non-zero count varies). Safe here since our synthetic
            #           count is exactly args.active.
            #   int  -> static; MUST equal count_nonzero(sparsity) or the device
            #           deadlocks. We only ever pass args.active (== our count).
            nnz_val = None if args.nnz < 0 else args.nnz

            def once():
                # down projection feeds the per-expert activation as the sparse input.
                return ttnn.sparse_matmul(
                    h_t,
                    w_t,
                    sparsity=s_t,
                    nnz=nnz_val,
                    is_input_a_sparse=is_a_sparse,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    output_tile=output_tile,
                    program_config=pc,
                    dtype=DTYPES[args.act_dtype],
                )

            out = once()
            ttnn.synchronize_device(dev)
            # sparse_matmul returns a 6-D tensor [1,1,1,E,1,N]; the expert axis is dim 3.
            # Only the first `active` expert slots are actually computed. Reshape to
            # [E, N] and compare the active rows against the [active, N] reference.
            got = ttnn.to_torch(out).to(torch.float32).reshape(E, N)
            p = pcc(ref, got[: args.active, :])
            ttnn.deallocate(out)

            samples = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                out = once()
                ttnn.synchronize_device(dev)
                samples.append((time.perf_counter() - t0) * 1e3)
                ttnn.deallocate(out)
            samples.sort()
            med, mn = samples[len(samples) // 2], samples[0]
            status = "PASS" if p >= args.pcc else "LOWPCC"
            result = {
                "status": status,
                "median_ms": round(med, 4),
                "min_ms": round(mn, 4),
                "pcc": round(p, 4),
                "note": "",
            }
            print(f"RESULT worker median_ms={med:.4f} min_ms={mn:.4f} pcc={p:.4f} [{status}]", flush=True)
        finally:
            try:
                ttnn.close_mesh_device(dev)
            except Exception:
                pass
    except (RuntimeError, ValueError, MemoryError) as exc:
        result = {
            "status": "SKIP",
            "median_ms": "",
            "min_ms": "",
            "pcc": "",
            "note": f"{type(exc).__name__}: {str(exc)[:150]}",
        }
        print(f"SKIP worker: {result['note']}", flush=True)
    except BaseException as exc:  # noqa: BLE001
        result = {
            "status": "CRASH",
            "median_ms": "",
            "min_ms": "",
            "pcc": "",
            "note": f"{type(exc).__name__}: {str(exc)[:150]}",
        }
        print(f"CRASH worker: {result['note']}", flush=True)
    finally:
        try:
            rf.parent.mkdir(parents=True, exist_ok=True)
            with open(rf, "w") as f:
                json.dump(result, f)
        except Exception:
            pass
    return 0


def reset_board(tt_smi):
    """Hard-reset the board so a deadlocked config doesn't poison the next trial."""
    try:
        subprocess.run([tt_smi, "-r"], timeout=180, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception as e:
        print(f"# board reset FAILED: {type(e).__name__}: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# ORCHESTRATOR: generate grids, run each in a worker subprocess with a hard
# timeout, reset the board on hang, aggregate results + live progress.
# ---------------------------------------------------------------------------
def run_orchestrator(args):
    E, K, N = args.experts, args.K, args.N
    Kt, Nt = K // TILE, N // TILE
    print(
        f"# MoE sparse_matmul {args.proj}: hidden[1,1,1,{K}] @ w[1,{E},{K},{N}] "
        f"(Kt={Kt} Nt={Nt}), active={args.active}/{E}, grid_max={args.grid}",
        flush=True,
    )

    grids = gen_grids(Nt, args.grid[0], args.grid[1], args.extra_cores)
    total = len(grids)
    print(f"# {total} rectangular grids to try (per-trial timeout={args.trial_timeout}s)", flush=True)

    if args.progress_file:
        progress_path = Path(args.progress_file)
    elif args.csv:
        progress_path = Path(args.csv).with_suffix(".progress.json")
    else:
        progress_path = Path("moe_sparse_sweep.progress.json")
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    result_file = progress_path.with_suffix(".worker.json")
    start_ts = time.perf_counter()
    rows = []
    best = None
    n_pass = n_lowpcc = n_skip = n_hang = n_crash = 0

    def write_progress(done, current_label, finished=False):
        elapsed = time.perf_counter() - start_ts
        rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
        remaining = total - done
        eta = remaining / rate if rate > 0 else None
        snap = {
            "updated": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "sweep": "moe_sparse_matmul",
            "proj": args.proj,
            "gemm": f"hidden[1,1,1,{K}]@w[1,{E},{K},{N}] Nt={Nt}",
            "total": total,
            "done": done,
            "remaining": remaining,
            "percent": round(100.0 * done / total, 2) if total else 100.0,
            "pass": n_pass,
            "lowpcc": n_lowpcc,
            "skipped": n_skip,
            "hang": n_hang,
            "crash": n_crash,
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta, 1) if eta is not None else None,
            "avg_s_per_trial": round(1.0 / rate, 3) if rate > 0 else None,
            "current": current_label,
            "best": ({"median_ms": best[0], "label": best[1], "num_cores": best[2]} if best else None),
            "finished": finished,
            "csv": str(Path(args.csv)) if args.csv else None,
        }
        tmp = progress_path.with_suffix(progress_path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(snap, f, indent=2)
        os.replace(tmp, progress_path)

    write_progress(0, "(starting)")
    print(f"# progress file: {progress_path}", flush=True)
    bar = tqdm(total=total, unit="grid", dynamic_ncols=True) if tqdm is not None else None

    base_cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--worker",
        "--proj",
        args.proj,
        "--experts",
        str(E),
        "--active",
        str(args.active),
        "--K",
        str(K),
        "--N",
        str(N),
        "--dtype",
        args.dtype,
        "--act-dtype",
        args.act_dtype,
        "--fidelity",
        args.fidelity,
        "--in0-block-w",
        str(args.in0_block_w),
        "--iters",
        str(args.iters),
        "--pcc",
        str(args.pcc),
        "--seed",
        str(args.seed),
        "--nnz",
        str(args.nnz),
        "--result-file",
        str(result_file),
    ]

    for gi, (cx, cy, nc, pcn) in enumerate(grids):
        label = f"grid{cx}x{cy}_nc{nc}_pcN{pcn}"
        write_progress(gi, label)  # mark 'current' before launching
        cmd = base_cmd + ["--cx", str(cx), "--cy", str(cy), "--pcn", str(pcn)]
        try:
            if result_file.exists():
                result_file.unlink()
        except Exception:
            pass

        status, med, mn, p, note = "CRASH", "", "", "", ""
        try:
            proc = subprocess.run(
                cmd, timeout=args.trial_timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            # Read the worker's structured result (survives even if stdout is noisy).
            if result_file.exists():
                r = json.loads(result_file.read_text())
                status = r.get("status", "CRASH")
                med, mn, p, note = r.get("median_ms", ""), r.get("min_ms", ""), r.get("pcc", ""), r.get("note", "")
            else:
                status = "CRASH"
                tail = (proc.stdout or "").strip().splitlines()[-1:] if proc.stdout else []
                note = f"no result file (rc={proc.returncode}) {tail}"
        except subprocess.TimeoutExpired:
            status = "HANG"
            note = f"exceeded {args.trial_timeout}s -> device deadlock, board reset"
            print(f"HANG {label}: {note}", flush=True)
            reset_board(args.tt_smi)

        if status == "PASS":
            n_pass += 1
            if best is None or (isinstance(med, (int, float)) and med < best[0]):
                best = (med, label, nc)
        elif status == "LOWPCC":
            n_lowpcc += 1
        elif status == "SKIP":
            n_skip += 1
        elif status == "HANG":
            n_hang += 1
        else:
            n_crash += 1

        if status not in ("HANG",):
            print(f"RESULT {label} median_ms={med} min_ms={mn} pcc={p} [{status}] {note}", flush=True)
        rows.append(
            dict(
                proj=args.proj,
                cx=cx,
                cy=cy,
                num_cores=nc,
                per_core_N=pcn,
                median_ms=med,
                min_ms=mn,
                pcc=p,
                status=status,
                note=note,
            )
        )

        if bar is not None:
            bar.update(1)
            bar.set_postfix_str(f"pass={n_pass} low={n_lowpcc} skip={n_skip} hang={n_hang}", refresh=False)
        write_progress(gi + 1, label)

    if bar is not None:
        bar.close()
    write_progress(total, "(done)", finished=True)

    print(f"\n# trials: {n_pass} pass, {n_lowpcc} lowpcc, {n_skip} skip, {n_hang} hang, {n_crash} crash", flush=True)
    print("\n# ==== BEST (fastest PCC-passing) ====")
    if best:
        print(f"BEST {args.proj}: median_ms={best[0]:.4f}  {best[1]}  (num_cores={best[2]})")
    else:
        print(f"BEST {args.proj}: (no PCC-passing config)")

    if args.csv:
        p = Path(args.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        cols = ["proj", "cx", "cy", "num_cores", "per_core_N", "median_ms", "min_ms", "pcc", "status", "note"]
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"# wrote {len(rows)} rows to {p}")


def main():
    ap = argparse.ArgumentParser(description="MoE sparse_matmul core-grid sweep (subprocess-per-trial)")
    ap.add_argument("--proj", choices=["gate_up", "down"], default="gate_up")
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--active", type=int, default=4, help="non-zero experts in sparsity")
    ap.add_argument(
        "--nnz",
        type=int,
        default=-1,
        help="static nnz for sparse_matmul; -1 (default) => None/runtime-inferred (production behavior)",
    )
    ap.add_argument("--K", type=int, default=2880)
    ap.add_argument("--N", type=int, default=2880)
    ap.add_argument("--dtype", default="bfloat4_b", choices=list(DTYPES.keys()))
    ap.add_argument("--act-dtype", default="bfloat8_b", choices=list(DTYPES.keys()))
    ap.add_argument("--fidelity", default="LoFi", choices=list(FIDELITIES.keys()))
    ap.add_argument("--in0-block-w", type=int, default=2)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--grid", type=int, nargs=2, default=[13, 10], help="max core grid x y")
    ap.add_argument("--pcc", type=float, default=0.99)
    ap.add_argument("--extra-cores", type=int, nargs="*", default=[12, 30, 45, 90])
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument(
        "--progress-file",
        type=str,
        default="",
        help="live JSON progress file (default: <csv>.progress.json, or ./moe_sparse_sweep.progress.json)",
    )
    ap.add_argument("--seed", type=int, default=0)
    # orchestration / worker plumbing
    ap.add_argument(
        "--trial-timeout",
        type=float,
        default=120.0,
        help="per-config wall-clock timeout (s); exceeding it => HANG + board reset",
    )
    ap.add_argument(
        "--tt-smi",
        type=str,
        default="/local/ttuser/.tenstorrent-venv/bin/tt-smi",
        help="tt-smi binary used to reset the board after a hang",
    )
    ap.add_argument("--worker", action="store_true", help="internal: run a single grid config and exit")
    ap.add_argument("--cx", type=int, default=0, help="internal (worker): grid x")
    ap.add_argument("--cy", type=int, default=0, help="internal (worker): grid y")
    ap.add_argument("--pcn", type=int, default=0, help="internal (worker): per_core_N")
    ap.add_argument("--result-file", type=str, default="", help="internal (worker): where to write result JSON")
    args = ap.parse_args()

    if args.worker:
        sys.exit(run_worker(args))
    run_orchestrator(args)


if __name__ == "__main__":
    main()
