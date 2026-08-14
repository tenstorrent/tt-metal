# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""compact_stat_gather — isolated bake-off for rms_norm's CROSS-CORE STAT COMBINE.

    scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/compact_stat_gather/test_compact_stat_gather.py

Env knobs
    CSG_CASES=focus,s_sweep,b_sweep     which case groups to run
    CSG_MODES=0,1,2,3                   which MODEs to measure
    CSG_TRIALS=1                        fresh-cache runs per point (device kernel
                                        time has no warm-up transient; >1 only to
                                        re-check a number that looks implausible)

PRECISION CONTRACT (fixed, never a lever): bf16 activations, float32 stat tiles,
math_fidelity=HiFi2, fp32_dest_acc_en=False, math_approx_mode=False.  Every MODE
runs under the identical config; the only thing that changes is WHERE the
within-tile collapse happens and how many bytes cross the NoC.

MEASURED (Blackhole p150b @1350 MHz, DEVICE KERNEL DURATION [ns], one fresh run
per point, DRAM drain payload stubbed; repeat spread ±0.05%, e.g. baseline
63484/63518/63466 and candidate 51074/51008/51050):

  focus BLOCK (1,1,8192,1024) shard [1024,128] on 8x8 -> s=8, S=4, shard_rows=32
    B    baseline  collapse_4k  collapse_2k   row_128b   speedup   cb(base/cand)
    1      85576       90537        84331       78569     1.089x    46 /  22 KB
    2      73187       75730        71196       62439     1.172x    90 /  42 KB
    4      66625         —            —         54377     1.225x   178 /  82 KB
    8      63534       65449        65879       51025     1.245x   354 / 162 KB   <- the op's B
   16      61981         —            —         49084     1.263x   706 / 322 KB
   32   (>L1: 1410KB)    —            —         48904       —      —   / 642 KB   <- one-shot

  s sweep WIDTH, H=1024, S=4, B=8, ONE row-group of s cores (only the fan-in moves)
    s    baseline  collapse_4k  collapse_2k   row_128b   speedup
    2      44712       46531        47209       50170     0.891x  <- REGRESSION
    3      47415       49336        49639       50398     0.941x  <- REGRESSION
    4      55147       56967        57319       50359     1.095x
    6      59252       61105        61474       50563     1.172x
    8      63485       65341        65865       51051     1.243x
   16      80043       81610        84048       52503     1.525x
   32   (>L1: 1122KB)    —            —         55166       —
   64   (>L1: 2146KB)    —            —         63194       —

  READ THE MECHANISM OFF THE TABLE: the baseline is LINEAR in s (~+2.4 us per
  slice); row_128b is FLAT (50.2 -> 52.5 us from s=2 to s=16).  `collapse_2k`
  halves the payload BYTES and is a NULL/slight regression, which says the root's
  cost is not bytes-in-flight but the NUMBER OF TILES its unpacker+FPU chew:
  row_128b is the only variant that cuts that (B instead of B*s per block).
  `collapse_4k` cuts neither and costs the contributor's extra pass (+3%).

HELPER GAP verified here (see test_reduce_within_tile_skip_is_unreachable):
`ReduceWithinTile::Skip` — the template value that WOULD express "the contributor
already collapsed this axis" — cannot be instantiated.  The
"Skip is AccumulateViaAdd-only" static_assert in
ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl:884-889 sits at FUNCTION
scope, AFTER the `if constexpr (resolved_algorithm == AccumulateViaAdd) { ...
return; }` block, so it is not part of a discarded statement and is checked for
EVERY instantiation of reduce() — including the AccumulateViaAdd one it is
supposed to permit.
"""

from __future__ import annotations

import os
import statistics

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.rms_norm.perf_experiments.compact_stat_gather.csg_descriptor import (
    MODE_COLLAPSE_2K,
    MODE_COLLAPSE_4K,
    MODE_NAMES,
    MODE_RAW_TILE,
    MODE_ROW_128B,
    cb_bytes,
    create_program_descriptor,
    plan,
)

TILE = 32
EPS = 1e-6
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

TRIALS = int(os.environ.get("CSG_TRIALS", "1"))
MODES = [int(m) for m in os.environ.get("CSG_MODES", "0,1,2,3").split(",") if m != ""]
CASES = [c for c in os.environ.get("CSG_CASES", "focus").split(",") if c != ""]
# CSG_DRAIN=1 leaves the DRAM stat drain IN the measured program (so the ns
# include this bench's correctness plumbing); 0 (default) stubs its payload.
DRAIN = os.environ.get("CSG_DRAIN", "0") == "1"

# Conservative per-core CB budget (the op's own ledger uses 1 MB - 96 KB).
L1_CB_BUDGET = 1024 * 1024 - 96 * 1024


def _compute_config():
    """The focus case's user-provided precision contract. FIXED for every variant."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=False,
        math_approx_mode=False,
    )


# ---------------------------------------------------------------------------
# Geometries
# ---------------------------------------------------------------------------


def _block_sharded(device, h, w, shard_h, shard_w):
    gx = w // shard_w
    gy = h // shard_h
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _width_sharded(device, h, w, shard_w, gx, gy):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [h, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _rect_for(s):
    """A grid rect holding exactly `s` cores (row-major shard order)."""
    return {2: (2, 1), 3: (3, 1), 4: (4, 1), 6: (6, 1), 8: (8, 1), 16: (8, 2), 32: (8, 4), 64: (8, 8)}[s]


def _cases():
    """(label, kind, h, w, shard_h, shard_w, grid, block_rows) tuples."""
    out = []
    if "focus" in CASES:
        # THE perf-flagged target: (1,1,8192,1024) BLOCK_SHARDED [1024,128] on 8x8
        # => s=8, S=4, shard_rows=32, B=8, 4 blocks, 8 row-groups of 8 cores.
        out.append(("focus_8192x1024_block", "block", 8192, 1024, 1024, 128, None, 8))
    if "b_sweep" in CASES:
        for b in (1, 2, 4, 8, 16, 32):
            out.append((f"focus_B{b}", "block", 8192, 1024, 1024, 128, None, b))
    if "s_sweep" in CASES:
        # One row-group of `s` cores, per-core geometry held CONSTANT (S=4,
        # shard_rows=32) so only the fan-in `s` moves.
        for s in (2, 3, 4, 6, 8, 16, 32, 64):
            gx, gy = _rect_for(s)
            out.append((f"s{s}_width", "width", 1024, s * 128, 1024, 128, (gx, gy), 8))
    if "s_sweep_b1" in CASES:
        for s in (32, 64):
            gx, gy = _rect_for(s)
            out.append((f"s{s}_width_B1", "width", 1024, s * 128, 1024, 128, (gx, gy), 1))
    return out


def _make_memcfg(device, kind, h, w, shard_h, shard_w, grid):
    if kind == "block":
        return _block_sharded(device, h, w, shard_h, shard_w)
    gx, gy = grid
    return _width_sharded(device, h, w, shard_w, gx, gy)


# ---------------------------------------------------------------------------
# Reference + measurement
# ---------------------------------------------------------------------------


def _reference(x):
    """1/sqrt(mean(x^2) + eps) per row, in float64."""
    xf = x.to(torch.float64)
    return torch.rsqrt((xf * xf).mean(dim=-1) + EPS)


def _pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    return 1.0 if denom == 0 else float((a @ b) / denom)


def _output_pcc(x_rows, got, expected):
    """PCC of the OP-LEVEL result x*(1/rms) implied by the measured stat vector."""
    a = x_rows * got.unsqueeze(-1)
    b = x_rows * expected.unsqueeze(-1)
    return _pcc(a, b)


def _flush_profiler(device):
    """Drain everything already on device so the next read holds ONE program.

    Without this the read after a launch also picks up every UNREAD program still
    buffered (e.g. the correctness launch), and summing them silently doubles the
    number — measured: 130144 for a program whose real duration is 63502.
    """
    ttnn.ReadDeviceProfiler(device)
    ttnn.get_latest_programs_perf_data()


def _read_kernel_ns(device):
    """DEVICE KERNEL DURATION [ns] of the single program launched since the flush."""
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    durations = []
    for programs in per_chip.values():
        for program in programs:
            analyses = getattr(program, "program_analyses_results", None) or {}
            entry = analyses.get(_DURATION_KEY)
            if entry is not None:
                durations.append(float(entry.duration))
    if not durations:
        return None
    assert len(durations) == 1, f"expected ONE program in the profiler read, got {len(durations)}: {durations}"
    return durations[0]


def _run(device, x_tt, stat_out, *, mode, block_rows, poison=False, drain=True):
    desc, _p = create_program_descriptor(
        x_tt,
        stat_out,
        mode=mode,
        epsilon=EPS,
        block_rows=block_rows,
        poison_landing=poison,
        drain=drain,
        compute_kernel_config=_compute_config(),
    )
    return ttnn.generic_op([x_tt, stat_out], desc)


def _extract(stat_out, num_row_tiles):
    """Column 0 of every fp32 stat tile carries the per-row 1/rms."""
    t = ttnn.to_torch(stat_out).to(torch.float64)
    return t[0, 0, : num_row_tiles * TILE, 0]


# ---------------------------------------------------------------------------
# The bake-off
# ---------------------------------------------------------------------------


def test_compact_stat_gather(device):
    torch.manual_seed(7)
    table = []
    wrong = []

    for label, kind, h, w, shard_h, shard_w, grid, block_rows in _cases():
        memcfg = _make_memcfg(device, kind, h, w, shard_h, shard_w, grid)
        x = torch.randn((1, 1, h, w), dtype=torch.float32) * 0.7
        x_bf16 = x.to(torch.bfloat16)
        try:
            x_tt = ttnn.from_torch(
                x_bf16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg
            )
        except Exception as exc:  # geometry does not fit this grid / device
            logger.warning(f"SKIP {label}: cannot place input ({exc})")
            continue

        expected = _reference(x_bf16.to(torch.float32))[0, 0]
        x_ref_rows = x_bf16.to(torch.float64)[0, 0]
        stat_out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, h, TILE]),
            ttnn.float32,
            ttnn.TILE_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        p = plan(x_tt, block_rows=block_rows)
        s, S, B = p["s"], p["S"], p["B"]
        row = {"case": label, "s": s, "S": S, "B": B, "blocks": p["shard_rows"] // B}

        for mode in MODES:
            need = cb_bytes(mode, s, S, B, in_tile_bytes=ttnn.tile_size(ttnn.bfloat16))
            if need > L1_CB_BUDGET:
                logger.warning(f"SKIP {label}/{MODE_NAMES[mode]}: CB footprint {need / 1024:.0f} KB over budget")
                row[MODE_NAMES[mode]] = None
                row[MODE_NAMES[mode] + "_pcc"] = None
                continue
            try:
                out = _run(device, x_tt, stat_out, mode=mode, block_rows=B)
                got = _extract(out, p["row_tiles"])
            except Exception as exc:
                logger.error(f"FAIL {label}/{MODE_NAMES[mode]}: {exc}")
                raise
            pcc = _pcc(got, expected)
            rel = ((got - expected).abs() / expected.abs().clamp_min(1e-12)).max().item()
            # The op's soft gate (0.9995) is measured on x*(1/rms)*gamma, which is
            # dominated by x; the STAT vector on its own is ~32x more sensitive.
            # `out_pcc` reconstructs the op-level number from the measured stat so
            # the two are comparable.
            out_pcc = _output_pcc(x_ref_rows, got, expected)
            logger.info(f"{label} {MODE_NAMES[mode]}: stat_pcc={pcc:.7f} out_pcc={out_pcc:.7f} max_rel={rel:.3e}")
            # GATE. `stat_pcc` alone is not a correctness gate here: it degrades with
            # W purely from the FIXED precision contract (a bf16 DEST accumulating
            # W terms), e.g. 0.9966 at W=1024 and 0.9676 at W=8192 with the SAME
            # kernel.  What must hold is (a) the op-level number, and (b) a
            # per-row relative error in the bf16-accumulation family (~1e-2), which
            # is what separates "rounds differently" from "loses a contributor"
            # (a lost contributor shows up as max_rel ~ 0.4-2.5, measured).
            if out_pcc <= 0.9999 or rel >= 2e-2:
                # Record and keep going: a broken MODE is data about that MODE, and
                # the rest of the menu still has to be measured.
                wrong.append(f"{label}/{MODE_NAMES[mode]}: out_pcc {out_pcc:.7f}, max_rel {rel:.3e}")
                logger.error(f"WRONG {label}/{MODE_NAMES[mode]} — measuring it anyway, then failing at the end")

            # PERF: the DRAM drain of the finalized stat tiles is this bench's
            # correctness plumbing, not part of the part under test, and on the ROOT
            # it sits squarely in the combine's serial chain.  Stub the payload,
            # keep every barrier / CB handshake / trip count (/perf-measure).
            ns = []
            for _ in range(TRIALS):
                _flush_profiler(device)
                _run(device, x_tt, stat_out, mode=mode, block_rows=B, drain=DRAIN)
                d = _read_kernel_ns(device)
                assert d is not None, f"no profiler duration for {label}/{MODE_NAMES[mode]}"
                ns.append(d)
            med = statistics.median(ns)
            row[MODE_NAMES[mode]] = med
            row[MODE_NAMES[mode] + "_pcc"] = pcc
            row[MODE_NAMES[mode] + "_outpcc"] = out_pcc
            row[MODE_NAMES[mode] + "_rel"] = rel
            row[MODE_NAMES[mode] + "_cb_kb"] = need // 1024
            logger.info(f"{label} {MODE_NAMES[mode]}: {med:.0f} ns  (samples {[f'{v:.0f}' for v in ns]})")

        table.append(row)

    logger.info("=" * 108)
    base_name = MODE_NAMES[MODE_RAW_TILE]
    hdr = f"{'case':<24}{'s':>4}{'S':>3}{'B':>3}{'blk':>4}"
    for m in MODES:
        hdr += f"{MODE_NAMES[m][:18]:>20}"
    logger.info(hdr)
    for row in table:
        line = f"{row['case']:<24}{row['s']:>4}{row['S']:>3}{row['B']:>3}{row['blocks']:>4}"
        base = row.get(base_name)
        for m in MODES:
            v = row.get(MODE_NAMES[m])
            if v is None:
                line += f"{'L1-skip':>20}"
            elif base and m != MODE_RAW_TILE:
                line += f"{f'{v:.0f} ({base / v:.2f}x)':>20}"
            else:
                line += f"{f'{v:.0f}':>20}"
        logger.info(line)
    logger.info("=" * 108)
    for row in table:
        for m in MODES:
            if row.get(MODE_NAMES[m]) is None:
                continue
            logger.info(
                f"PRECISION {row['case']:<22} {MODE_NAMES[m]:<20} "
                f"stat_pcc={row[MODE_NAMES[m] + '_pcc']:.7f} "
                f"out_pcc={row[MODE_NAMES[m] + '_outpcc']:.7f} "
                f"max_rel={row[MODE_NAMES[m] + '_rel']:.3e} "
                f"cb={row[MODE_NAMES[m] + '_cb_kb']}KB"
            )

    assert not wrong, "modes with an incorrect result:\n  " + "\n  ".join(wrong)


# ---------------------------------------------------------------------------
# Correctness pins
# ---------------------------------------------------------------------------


def test_landing_pad_rows_contribute_zero(device):
    """Rows s..31 of MODE 3's single landing tile MUST contribute ZERO.

    The landing tile is a 32-row buffer holding only `s` real contributor rows.
    If the unwritten rows leaked stale L1 into the root's REDUCE_COL, the result
    would be silently wrong — exactly the class of bug rms_norm's `pad_poison`
    golden group exists to catch.  This test pre-fills the WHOLE landing buffer
    with 1e30 at boot and then lets the kernel's own NoC zeroing run over it; a
    surviving poison lane turns the answer into inf/NaN, so a passing PCC is a
    proof that the zeroing covers every lane the reduce reads.

    Same pin applies to MODE 2, whose contributors write only 2 of 4 faces.
    """
    torch.manual_seed(11)
    h, w = 1024, 1024  # s=8, S=4, shard_rows=32, one row-group
    memcfg = _width_sharded(device, h, w, 128, 8, 1)
    x = (torch.randn((1, 1, h, w), dtype=torch.float32) * 0.7).to(torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memcfg)
    expected = _reference(x.to(torch.float32))[0, 0]
    stat_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, h, TILE]), ttnn.float32, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    p = plan(x_tt, block_rows=8)
    for mode in (MODE_COLLAPSE_2K, MODE_ROW_128B):
        clean = _extract(_run(device, x_tt, stat_out, mode=mode, block_rows=8, poison=False), p["row_tiles"]).clone()
        dirty = _extract(_run(device, x_tt, stat_out, mode=mode, block_rows=8, poison=True), p["row_tiles"]).clone()
        pcc = _pcc(dirty, expected)
        rel = ((dirty - expected).abs() / expected.abs().clamp_min(1e-12)).max().item()
        logger.info(f"pad_poison {MODE_NAMES[mode]}: pcc_vs_torch={pcc:.7f} max_rel={rel:.3e}")
        assert torch.isfinite(dirty).all(), f"{MODE_NAMES[mode]}: poison leaked (non-finite output)"
        # The sharpest pin available: poisoned and clean runs must be BIT-IDENTICAL.
        # Any surviving poison lane in a row the reduce reads would move a value.
        assert torch.equal(clean, dirty), (
            f"{MODE_NAMES[mode]}: poison changed the result — the landing buffer's "
            f"un-owned lanes are NOT fully neutralized "
            f"(max abs diff {(clean - dirty).abs().max().item():.3e})"
        )


@pytest.mark.skipif(
    os.environ.get("CSG_RUN_SKIP_PROBE", "0") != "1",
    reason="compile-only probe of the ReduceWithinTile::Skip helper gap; set CSG_RUN_SKIP_PROBE=1",
)
def test_reduce_within_tile_skip_is_unreachable(device):
    """Documented as a capability gap, not exercised by default (it cannot compile).

    See the module docstring: reduce_helpers_compute.inl:884-889.
    """
    pytest.skip("see module docstring — ReduceWithinTile::Skip does not instantiate")
