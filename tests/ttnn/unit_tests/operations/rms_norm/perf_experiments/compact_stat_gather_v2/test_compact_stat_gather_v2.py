# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""compact_stat_gather_v2 — isolated bake-off for rms_norm's CROSS-CORE STAT GATHER.

    scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/compact_stat_gather_v2/test_compact_stat_gather_v2.py

The bench itself (program descriptor + kernels) lives in the op's experiment dir:
    tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/compact_stat_gather_v2/
Only the pytest driver lives here — a test file INSIDE the ttnn package tree makes
pytest's rootdir insertion import ttnn twice ("Operation ... already registered").

Env knobs
    CSG2_CASES=focus,b_sweep,s_sweep,s_b32,s_wide,decode
    CSG2_MODES=0,1,2,3
    CSG2_TRIALS=1   (device kernel time has no warm-up transient; >1 only to
                     re-check a number that looks implausible)
    CSG2_DRAIN=1    keep the DRAM stat drain IN the measured program

PRECISION CONTRACT (fixed, never a lever): bf16 activations, float32 stat tiles,
math_fidelity=HiFi2, fp32_dest_acc_en=False, math_approx_mode=False.  Every MODE
runs under the identical config; the only thing that changes is WHERE the
within-tile collapse happens and how many bytes/transactions cross the NoC.
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import torch
import ttnn
from loguru import logger

_BENCH_DIR = (
    Path(__file__).resolve().parents[7] / "tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/compact_stat_gather_v2"
)
assert _BENCH_DIR.is_dir(), f"bench dir not found: {_BENCH_DIR}"
sys.path.insert(0, str(_BENCH_DIR))

from csg2_descriptor import (  # noqa: E402
    MODE_NAMES,
    MODE_RAW_4K,
    MODE_ROW_128B,
    MODE_ROW_64B_PROBE,
    cb_bytes,
    create_program_descriptor,
    plan,
)

TILE = 32
EPS = 1e-6
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

TRIALS = int(os.environ.get("CSG2_TRIALS", "1"))
MODES = [int(m) for m in os.environ.get("CSG2_MODES", "0,1").split(",") if m != ""]
CASES = [c for c in os.environ.get("CSG2_CASES", "focus").split(",") if c != ""]
DRAIN = os.environ.get("CSG2_DRAIN", "0") == "1"

# Conservative per-core CB budget (the op's own ledger uses 1 MB - 96 KB).
L1_CB_BUDGET = 1024 * 1024 - 96 * 1024

# MODE 3 is an ablation probe, not an option: it ships one of MODE 1's two
# face-row writes, so half of every tile-row's stat is missing by construction.
# It is measured, never gated.
PROBE_MODES = (MODE_ROW_64B_PROBE,)


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


def _block_sharded(h, w, shard_h, shard_w):
    gx = w // shard_w
    gy = h // shard_h
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_h, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _width_sharded(h, shard_w, gx, gy):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [h, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


_RECT = {2: (2, 1), 3: (3, 1), 4: (4, 1), 6: (6, 1), 8: (8, 1), 16: (8, 2), 28: (7, 4), 32: (8, 4)}


def _cases():
    """(label, kind, h, w, shard_h, shard_w, grid, block_rows) tuples."""
    out = []
    if "focus" in CASES:
        # THE perf-flagged target: (1,1,8192,1024) BLOCK_SHARDED [1024,128] on 8x8
        # => s=8, S=4, shard_rows=32, B=16, 2 blocks, num_owners=8, own_rows=2.
        out.append(("focus_8192x1024_B16", "block", 8192, 1024, 1024, 128, None, 16))
    if "b_sweep" in CASES:
        for b in (1, 2, 4, 8, 16, 32):
            out.append((f"focus_B{b}", "block", 8192, 1024, 1024, 128, None, b))
    if "s_sweep" in CASES:
        # One row-group of `s` cores, per-core geometry held CONSTANT (S=4,
        # shard_rows=32, B=8) so only the fan-in `s` moves.
        for s in (2, 4, 8, 16, 28, 32):
            gx, gy = _RECT[s]
            out.append((f"s{s}_B8", "width", 1024, s * 128, 1024, 128, (gx, gy), 8))
    if "s_b32" in CASES:
        # own_rows 8 and 16 (B=32 with a narrow fan-in).
        for s in (2, 4, 8):
            gx, gy = _RECT[s]
            out.append((f"s{s}_B32", "width", 1024, s * 128, 1024, 128, (gx, gy), 32))
    if "s_wide" in CASES:
        # S = 8 hidden tiles per core (twice the local work per stat tile).
        for s in (4, 8):
            gx, gy = _RECT[s]
            out.append((f"s{s}_S8_B8", "width", 1024, s * 256, 1024, 256, (gx, gy), 8))
    if "decode" in CASES:
        # block_rows == 1 / num_owners == 1: a DIFFERENT topology (flat root, no
        # funnel).  One tile-row per core, so there is nothing to scatter.
        for s in (8, 32):
            gx, gy = _RECT[s]
            out.append((f"decode_s{s}", "width", 32, s * 128, 32, 128, (gx, gy), 1))
    return out


def _make_memcfg(kind, h, w, shard_h, shard_w, grid):
    if kind == "block":
        return _block_sharded(h, w, shard_h, shard_w)
    gx, gy = grid
    return _width_sharded(h, shard_w, gx, gy)


# ---------------------------------------------------------------------------
# Reference + measurement
# ---------------------------------------------------------------------------


def _reference(x):
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

    `get_latest_programs_perf_data()` returns every UNREAD program, so without
    this the read after a launch also picks up the correctness launch and summing
    them silently doubles the number.
    """
    ttnn.ReadDeviceProfiler(device)
    ttnn.get_latest_programs_perf_data()


def _read_kernel_ns(device):
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


def _run(device, x_tt, stat_out, *, mode, block_rows, drain=True):
    desc, _p = create_program_descriptor(
        x_tt,
        stat_out,
        mode=mode,
        epsilon=EPS,
        block_rows=block_rows,
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


def test_compact_stat_gather_v2(device):
    torch.manual_seed(7)
    table = []
    wrong = []

    for label, kind, h, w, shard_h, shard_w, grid, block_rows in _cases():
        memcfg = _make_memcfg(kind, h, w, shard_h, shard_w, grid)
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
        row = {
            "case": label,
            "s": s,
            "S": S,
            "B": B,
            "blk": p["shard_rows"] // B,
            "own": p["own_rows"],
            "nown": p["num_owners"],
        }

        for mode in MODES:
            need = cb_bytes(mode, s, S, B, p["num_owners"], p["own_rows"])
            if need > L1_CB_BUDGET:
                logger.warning(f"SKIP {label}/{MODE_NAMES[mode]}: CB footprint {need / 1024:.0f} KB over budget")
                row[MODE_NAMES[mode]] = None
                continue
            out = _run(device, x_tt, stat_out, mode=mode, block_rows=B)
            got = _extract(out, p["row_tiles"])
            pcc = _pcc(got, expected)
            rel = ((got - expected).abs() / expected.abs().clamp_min(1e-12)).max().item()
            # The op's soft gate (0.9995) is measured on x*(1/rms)*gamma, which is
            # dominated by x; the STAT vector on its own is ~32x more sensitive.
            # `out_pcc` reconstructs the op-level number from the measured stat so
            # the two are comparable.
            out_pcc = _output_pcc(x_ref_rows, got, expected)
            tag = " [PROBE, wrong by construction]" if mode in PROBE_MODES else ""
            logger.info(
                f"{label} {MODE_NAMES[mode]}: stat_pcc={pcc:.7f} out_pcc={out_pcc:.7f} max_rel={rel:.3e}{tag}"
            )
            if mode not in PROBE_MODES and (out_pcc <= 0.9999 or rel >= 2e-2):
                # Record and keep going: a broken MODE is data about that MODE,
                # and the rest of the menu still has to be measured.
                wrong.append(f"{label}/{MODE_NAMES[mode]}: out_pcc {out_pcc:.7f}, max_rel {rel:.3e}")
                logger.error(f"WRONG {label}/{MODE_NAMES[mode]} — measuring it anyway, then failing at the end")

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

    logger.info("=" * 120)
    base_name = MODE_NAMES[MODE_RAW_4K]
    hdr = f"{'case':<22}{'s':>4}{'S':>3}{'B':>4}{'blk':>4}{'own':>4}"
    for m in MODES:
        hdr += f"{MODE_NAMES[m][:18]:>22}"
    logger.info(hdr)
    for r in table:
        line = f"{r['case']:<22}{r['s']:>4}{r['S']:>3}{r['B']:>4}{r['blk']:>4}{r['own']:>4}"
        base = r.get(base_name)
        for m in MODES:
            v = r.get(MODE_NAMES[m])
            if v is None:
                line += f"{'L1-skip':>22}"
            elif base and m != MODE_RAW_4K:
                line += f"{f'{v:.0f} ({base / v:.3f}x)':>22}"
            else:
                line += f"{f'{v:.0f}':>22}"
        logger.info(line)
    logger.info("=" * 120)
    for r in table:
        for m in MODES:
            if r.get(MODE_NAMES[m]) is None:
                continue
            logger.info(
                f"PRECISION {r['case']:<20} {MODE_NAMES[m]:<18} "
                f"stat_pcc={r[MODE_NAMES[m] + '_pcc']:.7f} "
                f"out_pcc={r[MODE_NAMES[m] + '_outpcc']:.7f} "
                f"max_rel={r[MODE_NAMES[m] + '_rel']:.3e} "
                f"cb={r[MODE_NAMES[m] + '_cb_kb']}KB"
            )

    assert not wrong, "modes with an incorrect result:\n  " + "\n  ".join(wrong)


# ---------------------------------------------------------------------------
# Landing-pad contract
# --------------------------------------------------------------------------# ---------------------------------------------------------------------------
# Landing-pad contract
# ---------------------------------------------------------------------------
#
# The un-owned lanes of a compact landing tile (rows s..31, and MODE 2's faces 1
# and 3) must read as ZERO or they enter the owner's reduce as a phantom
# contributor.  Round 1 pinned this with an adversarial poison test because its
# zeroing was a CROSS-CORE race (root zeroes, contributors write, nothing orders
# them).  Here the contract is structural instead: every byte of the landing
# buffer has exactly ONE writer, and that writer's own per-block
# `noc_async_write_barrier()` + `gather_progress` increment is the edge the
# owner's reduce already waits on.  There is no race left to poison.
#
# What still has to be pinned is the ARITHMETIC of the pad map (an off-by-one in
# the `valid` row count would leave a live lane un-zeroed).  The sweep does that:
# `s` runs over {2,4,8,16,28,32} at fixed geometry, which walks the pad width
# from 30 rows down to 0 and crosses both face boundaries (s<16, s==16, s>16,
# s==32), and every point is gated on out_pcc/max_rel against torch.  The lanes
# start as whatever the previous program left in L1 -- real, non-zero tile data --
# so a missed lane shows up as a large max_rel, not as a benign zero.
