# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""tilize perf bench — measurement only, NO correctness assertions.

Underscore-prefixed so pytest's default `test_*` collection ignores it and it
never enters the golden matrix. Run it explicitly:

    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py

Reports, per regime: the **device core count** the program actually launched on
(so lever A0 is machine-checkable rather than eyeballed), the median device
kernel duration over a trial loop, DRAM traffic, and achieved GB/s.

Ablation (`/perf-measure`) is driven by env flags read by the program
descriptor; output is garbage by design, which is why this file asserts no PCC:

    TILIZE_BENCH_ABLATE=1   # run all four variants per regime
      full        -> baseline
      no_compute  -> TILIZE_SKIP_COMPUTE=1 (CB dance kept, tilize LLK dropped)
      no_dm       -> TILIZE_SKIP_DM=1      (CB dance + barriers kept, NoC dropped)
      sync_only   -> both

Other knobs: TILIZE_BENCH_TRIALS (default 10), TILIZE_BENCH_REGIMES (comma-sep
regime names, default all).
"""

from __future__ import annotations

import os

# Enable the on-device profiler IN-PROCESS. All three are required by
# ttnn.get_latest_programs_perf_data() and must be set BEFORE the device opens.
# Module-scoped (not a dir conftest) so the op's correctness tests in this same
# directory are not perturbed. setdefault -> respects an outer tracy run.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics

import pytest
import torch

import ttnn
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize.tilize_program_descriptor import build_plan

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

N_WARMUP = 3
N_TRIALS = int(os.environ.get("TILIZE_BENCH_TRIALS", "10"))  # launches per round
N_ROUNDS = int(os.environ.get("TILIZE_BENCH_ROUNDS", "5"))  # rounds -> median + CV
ABLATE = os.environ.get("TILIZE_BENCH_ABLATE", "0") == "1"

_L1 = ttnn.BufferType.L1
_ROW = ttnn.ShardOrientation.ROW_MAJOR


def _crs(end_x, end_y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))})


def _shard(scheme, grid, shape):
    return ttnn.MemoryConfig(scheme, _L1, ttnn.ShardSpec(grid, shape, _ROW))


# name -> dict(shape, dtype, out_dtype, in_cfg, out_cfg, multicore, why)
# (a)-(f) are the mandatory regimes from op_design.md "Perf bench".
REGIMES = {
    # (a) grid-filling square — per-core DRAM efficiency once the grid is full.
    "a_square": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16),
    # (b) wide-short (nt_h=1, Wt=512) — THE gate: does the split fill the grid?
    #     A height-only split strands this on one core.
    "b_wide_short": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16),
    # (c) single-core reference baseline.
    "c_single_core": dict(shape=(1, 1, 512, 512), dtype=ttnn.bfloat16, multicore=False),
    # (d) tall-narrow guard — no-regression witness for the height regime.
    "d_tall_narrow": dict(shape=(1, 1, 2048, 32), dtype=ttnn.bfloat16),
    # (e) dtype sweep — page size changes the bound.
    "e_square_fp32": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.float32),
    "e_square_bf8b_out": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, out_dtype=ttnn.bfloat8_b),
    # (f) sharded, same spec (Path B, zero-copy) — small (~1 us) and large.
    "f_sharded_small": dict(
        shape=(1, 1, 512, 64),
        dtype=ttnn.bfloat16,
        in_cfg=_shard(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _crs(3, 0), (128, 64)),
        same_cfg=True,
    ),
    "f_sharded_large": dict(
        shape=(1, 1, 2048, 512),
        dtype=ttnn.bfloat16,
        in_cfg=_shard(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _crs(7, 7), (256, 64)),
        same_cfg=True,
    ),
    # --- Mode-C counterfactuals: the same regime with ONE lever flipped off ----
    # C16 depth-2 CBs off -> reader/writer serialize instead of pipelining.
    "x_square_depth1": dict(shape=(1, 1, 2048, 2048), dtype=ttnn.bfloat16, double_buffer=False),
    # A0 2D split off -> the wide-short shape (nt_h=1) collapses onto one core,
    # which is exactly what a height-only split_work_to_cores(nt_h) would do.
    "x_wide_short_1core": dict(shape=(1, 1, 32, 16384), dtype=ttnn.bfloat16, multicore=False),
    # C16 on the smallest sharded regime (lever B0: per-core-overhead levers must
    # be counterfactualed on the SMALLEST shape they run in).
    "x_sharded_small_depth1": dict(
        shape=(1, 1, 512, 64),
        dtype=ttnn.bfloat16,
        in_cfg=_shard(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _crs(3, 0), (128, 64)),
        same_cfg=True,
        double_buffer=False,
    ),
}

_SELECTED = os.environ.get("TILIZE_BENCH_REGIMES", "")
REGIME_NAMES = [n.strip() for n in _SELECTED.split(",") if n.strip()] or list(REGIMES)


def _read_kernel_ns(device):
    """Summed on-device kernel duration for programs dispatched since the last read.

    ReadDeviceProfiler flushes the queue and *consumes* the window, so a
    flush-read then a work-read brackets exactly the launches in between.
    """
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure_median_ns(device, run_fn):
    """Median ns/launch over N_ROUNDS rounds of N_TRIALS launches each.

    Reads are BATCHED per round on purpose: ReadDeviceProfiler after a single
    launch reliably returns an empty window on this build, so each round runs
    N_TRIALS launches and divides. Warm-up window is flushed and discarded.
    Rounds give the std-dev / CV that the /perf-measure noise threshold needs.
    """
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # drop the warm-up window

    samples = []
    for _ in range(N_ROUNDS):
        for _ in range(N_TRIALS):
            run_fn()
        value = _read_kernel_ns(device)
        if value is not None:
            samples.append(value / N_TRIALS)
    if not samples:
        return None, None
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return statistics.median(samples), std


def _build(device, spec):
    shape = spec["shape"]
    dtype = spec["dtype"]
    in_cfg = spec.get("in_cfg", ttnn.DRAM_MEMORY_CONFIG)
    out_cfg = in_cfg if spec.get("same_cfg") else spec.get("out_cfg", ttnn.DRAM_MEMORY_CONFIG)

    torch.manual_seed(0)
    if dtype == ttnn.float32:
        torch_input = torch.randn(shape, dtype=torch.float32)
    else:
        torch_input = torch.randn(shape).bfloat16()

    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_cfg
    )
    return tt_input, out_cfg


def _plan_for(device, tt_input, spec, out_cfg):
    """Rebuild the plan host-side to report ncores / chunk_wt / CB bytes."""
    out_dtype = spec.get("out_dtype") or tt_input.dtype
    probe_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(tt_input.shape)), out_dtype, ttnn.TILE_LAYOUT, device, out_cfg
    )
    return build_plan(
        tt_input,
        probe_out,
        device,
        use_multicore=spec.get("multicore", True),
        use_double_buffer=spec.get("double_buffer", True),
    )


_VARIANTS = (
    [("full", "0", "0"), ("no_compute", "0", "1"), ("no_dm", "1", "0"), ("sync_only", "1", "1")]
    if ABLATE
    else [("full", "0", "0")]
)


def test_bench_tilize(device):
    """Measure; never assert correctness. The only assertion is that the
    profiler produced a number (i.e. this is a profiler-enabled build)."""
    grid = device.compute_with_storage_grid_size()
    rows = []

    for name in REGIME_NAMES:
        spec = REGIMES[name]
        tt_input, out_cfg = _build(device, spec)
        plan = _plan_for(device, tt_input, spec, out_cfg)

        elem_in = tt_input.element_size()
        bytes_read = plan["folded_h"] * plan["width"] * elem_in
        bytes_written = plan["total_tiles"] * plan["tile_out"]
        # Path B is zero-copy on BOTH sides: no DRAM traffic at all.
        traffic = 0 if plan["path"] == "alias" else bytes_read + bytes_written

        for variant, skip_dm, skip_compute in _VARIANTS:
            os.environ["TILIZE_SKIP_DM"] = skip_dm
            os.environ["TILIZE_SKIP_COMPUTE"] = skip_compute
            run_fn = lambda t=tt_input, s=spec, c=out_cfg: tilize(
                t,
                c,
                dtype=s.get("out_dtype"),
                use_multicore=s.get("multicore", True),
                use_double_buffer=s.get("double_buffer", True),
            )
            ns, std = _measure_median_ns(device, run_fn)
            assert ns is not None, f"profiler produced no data for {name}/{variant}"
            rows.append(
                dict(
                    regime=name,
                    variant=variant,
                    path=plan["path"],
                    ncores=plan["ncores"],
                    chunk_wt=plan["chunk_wt"],
                    cb_bytes=plan["cb_bytes_per_core"],
                    ns=ns,
                    cv=(std / ns * 100.0) if ns else 0.0,
                    traffic=traffic,
                )
            )

        os.environ["TILIZE_SKIP_DM"] = "0"
        os.environ["TILIZE_SKIP_COMPUTE"] = "0"

    arch = os.environ.get("ARCH_NAME", "unknown")
    lines = [
        "",
        "=== tilize device perf bench ===",
        f"    grid={grid.x}x{grid.y}  arch={arch}  rounds={N_ROUNDS}x{N_TRIALS} launches  ablate={ABLATE}",
        "    A0 gate: interleaved -> ncores == min(grid_cores, total_tiles); sharded -> shard's own cores",
        f"    {'regime':<20} {'variant':<11} {'path':<8} {'cores':>5} {'chk':>4} "
        f"{'cbB/core':>9} {'ns':>10} {'cv%':>5} {'MB':>7} {'GB/s':>7}",
    ]
    for r in rows:
        gbps = (r["traffic"] / r["ns"]) if (r["traffic"] and r["ns"]) else 0.0
        lines.append(
            f"    {r['regime']:<20} {r['variant']:<11} {r['path']:<8} {r['ncores']:>5} "
            f"{r['chunk_wt']:>4} {r['cb_bytes']:>9} {r['ns']:>10.1f} {r['cv']:>5.1f} "
            f"{r['traffic'] / 1e6:>7.2f} {gbps:>7.1f}"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
