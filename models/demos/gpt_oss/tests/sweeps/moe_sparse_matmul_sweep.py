# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generalized MoE sparse_matmul program-config sweep -- a CLI tool.

Sibling of models/demos/gpt_oss/tests/sweeps/matmul_sweep.py (same RESULT/BEST/CSV
structure + GB/s / BW-percent reporting), but for ttnn.sparse_matmul (the MoE expert
GEMM) instead of the dense ttnn.matmul. Give it the sparse GEMM dims and it sweeps the
program configuration (core grid, in0_block_w, out_subblock) x fidelity x dtype,
timing each on device, checking PCC against a torch reference, and reporting the
achieved DRAM bandwidth. It prints a RESULT line per config, writes a CSV, and
reports the BEST (fastest PCC-passing) config.

The sparse GEMM is a batch of per-expert matmuls gated by a sparsity vector:
    activation  [1, 1 or E, M=1, K]        (shared over experts, or per-expert)
    weight      [1, E, K, N]               (one [K,N] slice per expert)
    sparsity    [1, 1, 1, E]               (`active` of E experts non-zero)
    -> only `active` experts are actually computed; Nt = ceil(N/32).

Which operand is sparse is selectable:
    --sparse-input b (default): activation shared [1,1,1,K], weight-sparse (the
        gpt-oss gate/up projection). Alias: --proj gate_up.
    --sparse-input a: activation per-expert [1,E,1,K], is_input_a_sparse=True (the
        gpt-oss down projection). Alias: --proj down.

We sweep (core_x, core_y) grids. For a 1D mcast_in0 sparse matmul the kernel needs
    per_core_N = ceil(Nt / num_cores)   and    ceil(Nt / per_core_N) == num_cores
(rectangularity). Grids that violate this are skipped up front.

AVOIDING DEADLOCKS / HANGS  (applies to ALL MoE models, not just gpt-oss):
    These are properties of the ttnn.sparse_matmul `mcast_in0` KERNEL (the multicast
    sender<->receiver semaphore handshake), NOT of any particular model. Any MoE that
    calls sparse_matmul with these program configs will hit the same deadlocks, so the
    rules below are general. Empirically observed deadlock modes on Blackhole:
      1. TINY CORE GRIDS (num_cores below ~12): the in0 multicast handshake wedges
         when there are too few receiver cores. --min-cores skips these (default 12).
      2. out_subblock_w > 1 that cannot cleanly subdivide per_core_N into >=2 blocks
         (e.g. per_core_N=2 with osw=2 -> a single N-block -> the mcast loop hangs).
         Even when it *can* subdivide, osw>1 is fragile AND never wins for these
         memory-bound decode GEMMs (out_subblock_w=1 was fastest in every sweep).
         => DEFAULT is --out-subblock-ws 1. Only raise it deliberately + expect hangs.
      3. A STATIC nnz that != count_nonzero(sparsity): the receivers loop a fixed
         count while the sender mcasts a different count -> noc_semaphore_wait hang.
         => Use nnz=None (runtime-inferred, the default) unless you know the exact count.
    A deadlock is NOT a Python exception -- it hangs the process on a device sync
    forever and wedges the board. So (defense in depth) each config runs in an isolated
    worker subprocess with a hard wall-clock timeout; on timeout we mark it HANG, reset
    the board (tt-smi -r), and continue. The defaults above avoid known-hang configs up
    front so the subprocess timeout is only a last resort, not the primary mechanism.

WHY SUBPROCESS-PER-TRIAL:
    See mode (1)-(3) above -- a hung config would otherwise wedge the board and kill
    the whole sweep. Subprocess isolation + timeout + board reset contains it.

Examples
  # gpt-oss-20b decode gate/up (fused [gate|up], N=5760), full knob sweep
  python models/demos/gpt_oss/tests/sweeps/moe_sparse_matmul_sweep.py --proj gate_up \
      --K 2880 --N 5760 --experts 32 --active 4 \
      --in0-block-ws 1 2 3 5 --out-subblock-ws 1 --fidelities LoFi HiFi2 \
      --csv models/demos/gpt_oss/tests/sweeps/out/gate_up.csv

  # Arbitrary sparse GEMM: 16 experts, 2 active, K=4096 N=4096, weight sparse
  python models/demos/gpt_oss/tests/sweeps/moe_sparse_matmul_sweep.py \
      --sparse-input b --experts 16 --active 2 --K 4096 --N 4096
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
MEMCFGS = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}

# Blackhole p150 peak DRAM BW (GDDR6). Used to report achieved BW% so sparse
# decode configs can be compared against the ~50% MoE bandwidth target.
PEAK_DRAM_BW = 512e9
DTYPE_BYTES = {"bfloat16": 2.0, "bfloat8_b": 1.0625, "bfloat4_b": 0.5625}


def achieved_bw(active, K, N, weight_dtype, act_dtype, median_ms):
    """GB/s moved by a sparse_matmul: only `active` experts are computed, so
    weights = active*K*N (dominant), activations = active*K, output = active*N.
    Returns (GB/s, pct_of_peak)."""
    wb = DTYPE_BYTES.get(weight_dtype, 2.0)
    ab = DTYPE_BYTES.get(act_dtype, 2.0)
    bytes_moved = active * (K * N * wb) + active * (K * ab) + active * (N * 2.0)
    gbs = bytes_moved / (median_ms / 1e3)
    return gbs, 100.0 * gbs / PEAK_DRAM_BW


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


def gen_grids(Nt: int, grid_max_x: int, grid_max_y: int, extra, min_cores: int = 1):
    """Enumerate (cx, cy) grids whose num_cores gives a rectangular tiling of Nt.

    min_cores skips tiny grids: sparse_matmul (mcast_in0) with a 1x1 / very small
    core grid DEADLOCKS the device (documented; the in0 mcast sender/receiver
    handshake wedges). Those aren't useful configs anyway (decode uses 90 cores),
    so filtering them out avoids burning the per-trial timeout on guaranteed hangs.
    """
    seen = set()
    trials = []
    for cy in range(1, grid_max_y + 1):
        for cx in range(1, grid_max_x + 1):
            nc = cx * cy
            if nc in seen or nc < min_cores:
                continue
            ok, pcn = rectangular(Nt, nc)
            if not ok:
                continue
            seen.add(nc)
            trials.append((cx, cy, nc, pcn))
    # Always include any explicitly requested core counts (as near-square grids).
    for nc in extra:
        if nc in seen or nc < min_cores:
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


def snap_ibw(in0_block_w, Kt):
    """Snap in0_block_w to a divisor of Kt (kernel asserts Kt % in0_block_w == 0)."""
    ibw = in0_block_w
    if Kt % ibw != 0:
        divs = [d for d in range(2, ibw + 1) if Kt % d == 0]
        ibw = max(divs) if divs else Kt
    return ibw


def build_pc(cx, cy, per_core_N, in0_block_w, Kt, out_subblock_h=1, out_subblock_w=1, obw_mult=1):
    ibw = snap_ibw(in0_block_w, Kt)
    # out_subblock_w must divide per_core_N; out_subblock_h must divide per_core_M (=1).
    # Also the DST register caps out_subblock_h*out_subblock_w (<=8 for bf8/bf16 dest).
    osw = out_subblock_w
    if per_core_N % osw != 0:
        divs = [d for d in range(osw, 0, -1) if per_core_N % d == 0]
        osw = divs[0] if divs else 1
    osh = out_subblock_h if (1 % out_subblock_h == 0) else 1
    if osh * osw > 8:
        osw = max(1, 8 // osh)
    # out_block_w axis. in1_num_subblocks = out_block_w / out_subblock_w, so
    # obw_mult is that subblock count. obw_mult=1 is Lucas Chin's obw=osw;
    # obw_mult=per_core_N/osw is obw=per_core_N. Must divide per_core_N.
    obw = osw * max(1, obw_mult)
    if obw > per_core_N or per_core_N % obw != 0:
        cands = [m for m in range(max(1, obw_mult), 0, -1) if per_core_N % (osw * m) == 0]
        obw = osw * (cands[0] if cands else 1)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(cx, cy),
        in0_block_w=ibw,
        out_subblock_h=osh,
        out_subblock_w=osw,
        out_block_h=1,
        # Lucas Chin (Anduril), PR #51514: out_block_w=1 forced in1_num_subblocks=0
        # for osw>1, so no osw>1 config was ever really exercised. Derive from osw.
        out_block_w=obw,
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
    if args.sparse_input == "a":
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
        is_a_sparse = args.sparse_input == "a"
        dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        output_tile = ttnn.Tile([32, 32])
        try:
            # Memory placement is a swept axis. Defaults match the in-model decode
            # path: activation/output in L1, weights in DRAM (expert weights are too
            # large to pin in L1, but a placement sweep confirms this). Pinning the
            # reloaded weight in L1 can raise achieved BW when it fits.
            h_t = ttnn.from_torch(
                act,
                dtype=DTYPES[args.act_dtype],
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=MEMCFGS[args.act_mem],
            )
            w_t = ttnn.from_torch(
                weight,
                dtype=DTYPES[args.dtype],
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=MEMCFGS[args.weight_mem],
            )
            s_t = ttnn.from_torch(
                spars,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            pc = build_pc(
                args.cx,
                args.cy,
                args.pcn,
                args.in0_block_w,
                Kt,
                out_subblock_h=args.out_subblock_h,
                out_subblock_w=args.out_subblock_w,
                obw_mult=args.obw_mult,
            )

            # nnz semantics (see ttnn.sparse_matmul docs + decode.py):
            #   None -> inferred at runtime (what production uses; robust when the
            #           real non-zero count varies). Safe here since our synthetic
            #           count is exactly args.active.
            #   int  -> static; MUST equal count_nonzero(sparsity) or the device
            #           deadlocks. We only ever pass args.active (== our count).
            nnz_val = None if args.nnz < 0 else args.nnz
            ckc = ttnn.init_device_compute_kernel_config(
                dev.arch(),
                math_fidelity=FIDELITIES[args.fidelity],
                math_approx_mode=False,
                fp32_dest_acc_en=bool(args.fp32_dest),
                packer_l1_acc=bool(args.packer_l1_acc),
            )

            def once():
                # down projection feeds the per-expert activation as the sparse input.
                return ttnn.sparse_matmul(
                    h_t,
                    w_t,
                    sparsity=s_t,
                    nnz=nnz_val,
                    is_input_a_sparse=is_a_sparse,
                    memory_config=MEMCFGS[args.out_mem],
                    output_tile=output_tile,
                    program_config=pc,
                    dtype=DTYPES[args.act_dtype],
                    compute_kernel_config=ckc,
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
            gbs, bwpct = achieved_bw(args.active, K, N, args.dtype, args.act_dtype, med)
            result = {
                "status": status,
                "median_ms": round(med, 4),
                "min_ms": round(mn, 4),
                "pcc": round(p, 4),
                "gbs": round(gbs / 1e9, 1),
                "bw_pct": round(bwpct, 1),
                "note": "",
            }
            print(
                f"RESULT worker median_ms={med:.4f} min_ms={mn:.4f} pcc={p:.4f} "
                f"GB/s={gbs/1e9:.0f} BW%={bwpct:.1f} [{status}]",
                flush=True,
            )
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

    import itertools as _it

    grids = gen_grids(Nt, args.grid[0], args.grid[1], args.extra_cores, min_cores=args.min_cores)
    # Full cartesian sweep: grid x in0_block_w x out_subblock_w x out_subblock_h
    #   x fidelity x act_dtype x fp32_dest x packer_l1_acc.
    # Each combo runs in its own worker subprocess (deadlock-isolated).
    Kt = K // TILE
    trials = []
    seen_eff = set()
    for cx, cy, nc, pcn in grids:
        for ibw, osw, obwm, osh, fid, adt, fp32, pl1, wmem, omem, amem in _it.product(
            args.in0_block_ws,
            args.out_subblock_ws,
            args.obw_mults,
            args.out_subblock_hs,
            args.fidelities,
            args.act_dtypes,
            args.fp32_dests,
            args.packer_l1_accs,
            args.weight_mems,
            args.out_mems,
            args.act_mems,
        ):
            # Dedup on the EFFECTIVE (snapped) config: ibw snaps to a divisor of
            # Kt, osw snaps to a divisor of per_core_N (capped by DST). Many raw
            # combos collapse to the same kernel config on high-core grids, so
            # skip the duplicates instead of re-timing them for hours.
            eff_ibw = snap_ibw(ibw, Kt)
            eff_osw = osw if pcn % osw == 0 else max([d for d in range(osw, 0, -1) if pcn % d == 0] or [1])
            if osh * eff_osw > 8:
                eff_osw = max(1, 8 // osh)
            # DEADLOCK GUARD (empirically verified): sparse_matmul mcast_in0 wedges
            # the device when per_core_N cannot be subdivided into >=2 N-blocks of
            # size out_subblock_w (e.g. 90 cores -> pcn=2, osw=2 -> 1 block -> hang).
            # It PASSes when pcn >= 2*osw (e.g. 12 cores -> pcn=15, osw=2 -> ok).
            # Lucas Chin (Anduril), PR #51514: this filter was a workaround for the
            # deadlock that out_block_w=1 caused. With out_block_w=osw it is not
            # needed, so it is disabled to let the full osw axis run.
            # if eff_osw > 1 and pcn < 2 * eff_osw:
            #     continue
            # effective obw_mult after the same clamping build_pc applies
            eff_obwm = max(1, obwm)
            if eff_osw * eff_obwm > pcn or pcn % (eff_osw * eff_obwm) != 0:
                cands = [m for m in range(eff_obwm, 0, -1) if pcn % (eff_osw * m) == 0]
                eff_obwm = cands[0] if cands else 1
            key = (cx, cy, pcn, eff_ibw, eff_osw, eff_obwm, osh, fid, adt, fp32, pl1, wmem, omem, amem)
            if key in seen_eff:
                continue
            seen_eff.add(key)
            trials.append(
                dict(
                    cx=cx,
                    cy=cy,
                    nc=nc,
                    pcn=pcn,
                    ibw=eff_ibw,
                    osw=eff_osw,
                    obwm=eff_obwm,
                    osh=osh,
                    fid=fid,
                    adt=adt,
                    fp32=fp32,
                    pl1=pl1,
                    wmem=wmem,
                    omem=omem,
                    amem=amem,
                )
            )
    if args.max_configs and len(trials) > args.max_configs:
        print(f"# generated {len(trials)} trials, capping to --max-configs={args.max_configs}", flush=True)
        trials = trials[: args.max_configs]
    total = len(trials)
    print(
        f"# {len(grids)} rectangular grids x knobs = {total} trials (per-trial timeout={args.trial_timeout}s)",
        flush=True,
    )

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
        "--sparse-input",
        args.sparse_input,
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

    for gi, t in enumerate(trials):
        cx, cy, nc, pcn = t["cx"], t["cy"], t["nc"], t["pcn"]
        label = (
            f"grid{cx}x{cy}_nc{nc}_pcN{pcn}_ib{t['ibw']}_sb{t['osh']}x{t['osw']}_obwm{t['obwm']}"
            f"_{t['fid']}_{t['adt']}_fp32d{t['fp32']}_pl1{t['pl1']}"
            f"_w{t['wmem']}_o{t['omem']}_a{t['amem']}"
        )
        write_progress(gi, label)  # mark 'current' before launching
        cmd = base_cmd + [
            "--cx",
            str(cx),
            "--cy",
            str(cy),
            "--pcn",
            str(pcn),
            "--in0-block-w",
            str(t["ibw"]),
            "--out-subblock-w",
            str(t["osw"]),
            "--obw-mult",
            str(t["obwm"]),
            "--out-subblock-h",
            str(t["osh"]),
            "--fidelity",
            t["fid"],
            "--act-dtype",
            t["adt"],
            "--fp32-dest",
            str(t["fp32"]),
            "--packer-l1-acc",
            str(t["pl1"]),
            "--weight-mem",
            t["wmem"],
            "--out-mem",
            t["omem"],
            "--act-mem",
            t["amem"],
        ]
        try:
            if result_file.exists():
                result_file.unlink()
        except Exception:
            pass

        status, med, mn, p, note, gbs, bwpct = "CRASH", "", "", "", "", "", ""
        try:
            proc = subprocess.run(
                cmd, timeout=args.trial_timeout, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            # Read the worker's structured result (survives even if stdout is noisy).
            if result_file.exists():
                r = json.loads(result_file.read_text())
                status = r.get("status", "CRASH")
                med, mn, p, note = r.get("median_ms", ""), r.get("min_ms", ""), r.get("pcc", ""), r.get("note", "")
                gbs, bwpct = r.get("gbs", ""), r.get("bw_pct", "")
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
                best = (med, label, nc, bwpct)
        elif status == "LOWPCC":
            n_lowpcc += 1
        elif status == "SKIP":
            n_skip += 1
        elif status == "HANG":
            n_hang += 1
        else:
            n_crash += 1

        if status not in ("HANG",):
            print(
                f"RESULT {label} median_ms={med} min_ms={mn} pcc={p} GB/s={gbs} BW%={bwpct} [{status}] {note}",
                flush=True,
            )
        rows.append(
            dict(
                proj=args.proj,
                cx=cx,
                cy=cy,
                num_cores=nc,
                per_core_N=pcn,
                in0_block_w=t["ibw"],
                out_subblock_w=t["osw"],
                obw_mult=t["obwm"],
                out_subblock_h=t["osh"],
                fidelity=t["fid"],
                act_dtype=t["adt"],
                fp32_dest=t["fp32"],
                packer_l1_acc=t["pl1"],
                weight_mem=t["wmem"],
                out_mem=t["omem"],
                act_mem=t["amem"],
                median_ms=med,
                min_ms=mn,
                pcc=p,
                gbs=gbs,
                bw_pct=bwpct,
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
        bw = best[3] if len(best) > 3 else ""
        print(
            f"BEST {args.proj} (sparse-input {args.sparse_input}): median_ms={best[0]:.4f} BW%={bw}  {best[1]}  (num_cores={best[2]})"
        )
    else:
        print(f"BEST {args.proj}: (no PCC-passing config)")

    if args.csv:
        p = Path(args.csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "proj",
            "cx",
            "cy",
            "num_cores",
            "per_core_N",
            "in0_block_w",
            "out_subblock_w",
            "obw_mult",
            "out_subblock_h",
            "fidelity",
            "act_dtype",
            "fp32_dest",
            "packer_l1_acc",
            "weight_mem",
            "out_mem",
            "act_mem",
            "median_ms",
            "min_ms",
            "pcc",
            "gbs",
            "bw_pct",
            "status",
            "note",
        ]
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"# wrote {len(rows)} rows to {p}")


def main():
    ap = argparse.ArgumentParser(description="MoE sparse_matmul core-grid sweep (subprocess-per-trial)")
    # --sparse-input is the generic knob: 'a' => activation is the sparse/per-expert
    # operand (is_input_a_sparse=True, gpt-oss 'down'); 'b' => weight is sparse and
    # activation is shared (gpt-oss 'gate_up'). --proj is a gpt-oss-friendly alias.
    ap.add_argument(
        "--sparse-input",
        choices=["a", "b"],
        default=None,
        help="which operand is sparse: a=activation/per-expert (down), b=weight/shared-act (gate_up)",
    )
    ap.add_argument(
        "--proj",
        choices=["gate_up", "down"],
        default=None,
        help="gpt-oss alias: gate_up=>--sparse-input b, down=>--sparse-input a",
    )
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
    # These accept MULTIPLE values in the orchestrator (swept); the worker reads a
    # single scalar (the orchestrator passes one value per subprocess).
    ap.add_argument("--act-dtypes", nargs="+", default=["bfloat8_b"], choices=list(DTYPES.keys()))
    ap.add_argument("--fidelities", nargs="+", default=["LoFi"], choices=list(FIDELITIES.keys()))
    ap.add_argument(
        "--in0-block-ws",
        type=int,
        nargs="+",
        default=[1, 2, 3, 5],
        help="in0_block_w values to sweep (snapped to a divisor of Kt)",
    )
    ap.add_argument(
        "--obw-mults",
        type=int,
        nargs="+",
        default=[1],
        help="out_block_w = obw_mult * out_subblock_w (i.e. in1_num_subblocks). "
        "1 = Lucas Chin's obw=osw. Higher explores out_block_w > out_subblock_w, "
        "which was never tested. Must divide per_core_N.",
    )
    ap.add_argument(
        "--out-subblock-ws",
        type=int,
        nargs="+",
        default=[1],
        help="out_subblock_w values to sweep. DEFAULT [1]: osw>1 frequently DEADLOCKS the "
        "sparse mcast kernel (see 'AVOIDING DEADLOCKS' in the module docstring) and never "
        "won in any decode sweep. Only pass >1 deliberately; expect HANGs (contained by timeout).",
    )
    ap.add_argument(
        "--out-subblock-hs",
        type=int,
        nargs="+",
        default=[1],
        help="out_subblock_h values to sweep (per_core_M=1 so 1 is the only valid value)",
    )
    ap.add_argument(
        "--fp32-dests", type=int, nargs="+", default=[0], choices=[0, 1], help="fp32_dest_acc_en values to sweep"
    )
    ap.add_argument(
        "--packer-l1-accs", type=int, nargs="+", default=[1], choices=[0, 1], help="packer_l1_acc values to sweep"
    )
    # Memory placement axes (default = in-model decode placement: act/out L1, weight DRAM).
    ap.add_argument(
        "--weight-mems",
        nargs="+",
        default=["dram"],
        choices=list(MEMCFGS.keys()),
        help="where the weight tensor lives: dram|l1. Sweep both to test if pinning the reloaded weight in L1 raises BW (may OOM for large expert weights).",
    )
    ap.add_argument(
        "--out-mems",
        nargs="+",
        default=["l1"],
        choices=list(MEMCFGS.keys()),
        help="where the output tensor lives: dram|l1",
    )
    ap.add_argument(
        "--act-mems",
        nargs="+",
        default=["l1"],
        choices=list(MEMCFGS.keys()),
        help="where the activation tensor lives: dram|l1",
    )
    # worker-only scalars (set by orchestrator per subprocess)
    ap.add_argument("--act-dtype", default="bfloat8_b", choices=list(DTYPES.keys()))
    ap.add_argument("--fidelity", default="LoFi", choices=list(FIDELITIES.keys()))
    ap.add_argument("--in0-block-w", type=int, default=2)
    ap.add_argument("--out-subblock-h", type=int, default=1)
    ap.add_argument("--out-subblock-w", type=int, default=1)
    ap.add_argument("--obw-mult", type=int, default=1)
    ap.add_argument("--fp32-dest", type=int, default=0)
    ap.add_argument("--packer-l1-acc", type=int, default=1)
    ap.add_argument("--weight-mem", default="dram", choices=list(MEMCFGS.keys()))
    ap.add_argument("--out-mem", default="l1", choices=list(MEMCFGS.keys()))
    ap.add_argument("--act-mem", default="l1", choices=list(MEMCFGS.keys()))
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--grid", type=int, nargs=2, default=[13, 10], help="max core grid x y")
    ap.add_argument("--pcc", type=float, default=0.99)
    ap.add_argument("--max-configs", type=int, default=0, help="cap total trials (0 = no cap)")
    ap.add_argument("--extra-cores", type=int, nargs="*", default=[12, 30, 45, 90])
    ap.add_argument(
        "--min-cores",
        type=int,
        default=12,
        help="skip grids with fewer cores. sparse_matmul mcast_in0 DEADLOCKS on tiny grids "
        "(<~12 cores); see 'AVOIDING DEADLOCKS' in the module docstring. Do not lower below 12.",
    )
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

    # Reconcile --proj (gpt-oss alias) and --sparse-input (generic). Keep args.proj
    # populated (used for labels/CSV) and derive the canonical sparse-input side.
    if args.proj is None and args.sparse_input is None:
        args.proj, args.sparse_input = "gate_up", "b"
    elif args.proj is not None and args.sparse_input is None:
        args.sparse_input = "a" if args.proj == "down" else "b"
    elif args.proj is None and args.sparse_input is not None:
        args.proj = "down" if args.sparse_input == "a" else "gate_up"
    else:
        expected = "a" if args.proj == "down" else "b"
        if args.sparse_input != expected:
            ap.error(f"--proj {args.proj} implies --sparse-input {expected}, got {args.sparse_input}")

    if args.worker:
        sys.exit(run_worker(args))
    run_orchestrator(args)


if __name__ == "__main__":
    main()
