# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate + on-device measurement for the sumsq/reduce merge bake-off.

Measurement recipe (the box's `run_safe_pytest.sh --profile` produces no
ops_perf_results CSV, so this uses the IN-PROCESS profiler):

    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 \
    TT_METAL_PROFILER_CPP_POST_PROCESS=1 timeout 1800 scripts/tt-probe.sh rms_norm <<'EOF'
    from ttnn.operations.rms_norm.perf_experiments.sumsq_reduce_merge.harness import main
    main()
    EOF

ONE fresh run per (variant, regime). No trial loops over the dispatch: `iters`
repeats the STAGE inside the kernel so the tiny per-block payload dominates the
program's fixed cost, and the reported per-block number is (total / iters). The
overhead cancels in the baseline-vs-candidate difference either way.
"""

from __future__ import annotations

import os
import shutil
import statistics

import torch
import ttnn

from .bench import BASELINE, VARIANTS, run_variant, sharded_config

TILE = 32
POISON = 1000.0  # pad-column value in a ragged hidden tile: finite, square is finite, x^2 ~ 1e6

# name -> (rows_t, core_w, valid_last, chunk_tiles, iters)
#   rows_t     tile-rows in the block  (the op's block_row_tiles)
#   core_w     hidden tiles owned by the core (incl. a ragged last tile)
#   valid_last valid columns in the LAST hidden tile (32 = aligned)
#   chunk_tiles the op's CB_CHUNK_TILES (None = one chunk, every interleaved geometry)
REGIMES = {
    # THE FOCUS SHAPE: (1,1,32,7168) interleaved -> G=110 cores, C=2-3, R=1, ONE block.
    "focus_r1_c3": (1, 3, TILE, None, 64),
    "decode_r1_c16": (1, 16, TILE, None, 32),
    "bshard_r16_c4": (16, 4, TILE, None, 16),
    "prefill_r5_c112": (5, 112, TILE, None, 4),
    # ragged hidden tile (W % 32 != 0): the masked tail stat column. nc = 2.
    "tail_r1_c3_v17": (1, 3, 17, None, 64),
    # hidden-axis chunking (a pinned HEIGHT shard): 4 chunks -> nc = 4 stat columns.
    "chunked_r1_c16_wc4": (1, 16, TILE, 4, 32),
}


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    if torch.allclose(a, b):
        return 1.0
    da, db = a - a.mean(), b - b.mean()
    denom = da.norm() * db.norm()
    if denom == 0:
        return 1.0 if da.norm() == db.norm() else 0.0
    return float((da * db).sum() / denom)


def make_input(device, rows_t, core_w, valid_last):
    torch.manual_seed(0)
    rows, cols = rows_t * TILE, core_w * TILE
    x = torch.randn((rows, cols), dtype=torch.float32)
    w_true = cols
    if valid_last != TILE:
        w_true = (core_w - 1) * TILE + valid_last
        x[:, w_true:] = POISON  # pad columns: must be annihilated by the mask, not squared
    xb = x.to(torch.bfloat16)
    tt = ttnn.from_torch(
        xb,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_config(rows_t, core_w),
    )
    ref = (xb[:, :w_true].float() ** 2).sum(-1) / w_true  # per row: mean of squares
    return tt, ref


def check(device, name, variant, **kw):
    """Run once and return (pcc, ratio_median) against the torch reference."""
    rows_t, core_w, valid_last, chunk, _ = REGIMES[name]
    tt_in, ref = make_input(device, rows_t, core_w, valid_last)
    out = run_variant(
        tt_in, variant=variant, rows_t=rows_t, core_w=core_w, valid_last=valid_last, chunk_tiles=chunk, iters=1, **kw
    )
    got = ttnn.to_torch(out)[:, 0].float()
    ttnn.deallocate(out)
    ttnn.deallocate(tt_in)
    ratios = (got / ref).tolist()
    return _pcc(got, ref), statistics.median(ratios), got, ref


def measure(device, name, variant, iters=None, **kw):
    """One fresh run; returns (device_kernel_ns, iters)."""
    rows_t, core_w, valid_last, chunk, dflt_iters = REGIMES[name]
    n = dflt_iters if iters is None else iters
    tt_in, _ = make_input(device, rows_t, core_w, valid_last)
    ttnn.ReadDeviceProfiler(device)  # flush the from_torch traffic
    out = run_variant(
        tt_in, variant=variant, rows_t=rows_t, core_w=core_w, valid_last=valid_last, chunk_tiles=chunk, iters=n, **kw
    )
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    ns = None
    for programs in per_chip.values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get("DEVICE KERNEL DURATION [ns]")
            if entry is None:
                continue
            d = float(entry.duration)
            ns = d if ns is None else max(ns, d)
    ttnn.deallocate(out)
    ttnn.deallocate(tt_in)
    return ns, n


def main(regimes=None, variants=None, zone_pass=("focus_r1_c3",)):
    regimes = list(REGIMES) if regimes is None else list(regimes)
    variants = list(VARIANTS) if variants is None else list(variants)
    device = ttnn.open_device(device_id=0)
    try:
        # ---- correctness gate first: a faster wrong answer is disqualified ----
        for name in regimes:
            for v in variants:
                try:
                    pcc, ratio, _, _ = check(device, name, v)
                    print(f"CHECK {name:22s} {v:18s} pcc={pcc:.6f} ratio_median={ratio:.6f}")
                except Exception as exc:  # noqa: BLE001 - report, keep going
                    print(f"CHECK {name:22s} {v:18s} FAILED {type(exc).__name__}: {exc}")
        # ---- one-shot latency: iters=1. The FOCUS geometry runs this stage exactly
        # ONCE per core (R=1, one block), so the steady-state loop below (which lets
        # the baseline's two phases pipeline across iterations) is NOT the faithful
        # metric there -- this is.
        for name in regimes:
            base1 = None
            for v in variants:
                try:
                    ns, _ = measure(device, name, v, iters=1)
                    if v == BASELINE:
                        base1 = ns
                    sp = "" if (base1 is None or not ns) else f" speedup={base1 / ns:.3f}x"
                    print(f"ONE   {name:22s} {v:18s} total_ns={ns}{sp}")
                except Exception as exc:  # noqa: BLE001
                    print(f"ONE   {name:22s} {v:18s} FAILED {type(exc).__name__}: {exc}")
        # ---- perf: one run per (variant, regime), steady state ----
        for name in regimes:
            base = None
            for v in variants:
                try:
                    ns, n = measure(device, name, v)
                    per = None if ns is None else ns / n
                    if v == BASELINE:
                        base = per
                    sp = "" if (base is None or per is None) else f" speedup={base / per:.3f}x"
                    print(f"PERF  {name:22s} {v:18s} iters={n:3d} total_ns={ns} per_block_ns={per:.1f}{sp}")
                except Exception as exc:  # noqa: BLE001
                    print(f"PERF  {name:22s} {v:18s} FAILED {type(exc).__name__}: {exc}")
        # ---- per-phase zone pass (iters=1, so the per-phase zones fit the budget) ----
        # The device profiler log is CUMULATIVE across ReadDeviceProfiler dumps, so
        # record each run's line range and slice the file per variant afterwards.
        csv = os.path.join(
            os.environ.get("TT_METAL_HOME", "."), "generated", "profiler", ".logs", "profile_log_device.csv"
        )

        def _lines():
            try:
                with open(csv) as fh:
                    return sum(1 for _ in fh)
            except OSError:
                return -1

        for name in zone_pass:
            for v in variants:
                try:
                    a = _lines()
                    ns, n = measure(device, name, v, iters=1)
                    b = _lines()
                    print(f"ZONE1 {name:22s} {v:18s} total_ns={ns} csv_lines={a}:{b}")
                except Exception as exc:  # noqa: BLE001
                    print(f"ZONE1 {name:22s} {v:18s} FAILED {type(exc).__name__}: {exc}")
        dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "zones_all.csv")
        try:
            shutil.copyfile(csv, dst)
            print(f"ZONECSV -> {dst}")
        except OSError as exc:
            print(f"ZONECSV failed: {exc}")
    finally:
        ttnn.close_device(device)
