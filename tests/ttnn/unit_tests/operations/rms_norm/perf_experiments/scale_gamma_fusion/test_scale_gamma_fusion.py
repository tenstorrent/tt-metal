# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness + device-kernel timing for the scale/gamma fusion bake-off.

Correctness is the ONLY pass/fail (every variant must reproduce x * (1/rms) * gamma).  Perf is
measured (DEVICE KERNEL DURATION [ns]) and reported, never asserted.  One run per variant — device
kernel time has no warm-up transient, so a trial loop would only re-measure the same number.

    scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/scale_gamma_fusion/test_scale_gamma_fusion.py

Env knobs: SGF_VARIANTS (comma list), SGF_CASES (comma list of B:S[:R]), SGF_DEST_CAP.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.rms_norm.perf_experiments.scale_gamma_fusion.scale_gamma_bench import (
    BASELINE,
    DEST_AUTO_LIMIT,
    GRID,
    VARIANTS,
    dest_block_for,
    plan_for,
    run_variant,
    sharded_memory_config,
)

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_NUM_CORES = GRID[0] * GRID[1]

# (B, S, row_tiles or None) sweep points.
#   (8, 4)  — the focus geometry's plan (S=4, B=8, 32 shard tile-rows/core, 4 blocks)
#   (1, 4)  — the decode regime's inner shape (a core owns ONE tile-row per block)
#   (1, 4, 1) — decode EXACTLY: one block, so the fused variants' one-off gamma expansion
#               is amortized over nothing
#   (1, 5)  — the 5120 geometry: S=5 makes DEST_BLOCK 5 or 1
#   (1, 8)  — the 7168/8 geometry
#   (32, 4) — one big block
#   (8, 1)  — narrow S (DEST_BLOCK is forced to 1)
_DEFAULT_CASES = ((8, 4, None), (1, 4, None), (1, 4, 1), (1, 5, None), (1, 8, None), (32, 4, None), (8, 1, None))


def _cases():
    raw = os.environ.get("SGF_CASES")
    if not raw:
        return _DEFAULT_CASES
    out = []
    for item in raw.split(","):
        parts = [int(p) for p in item.split(":")]
        out.append((parts[0], parts[1], parts[2] if len(parts) > 2 else None))
    return tuple(out)


def _variants():
    raw = os.environ.get("SGF_VARIANTS")
    return tuple(raw.split(",")) if raw else VARIANTS


def _quant(t):
    return t.to(torch.bfloat16).to(torch.float32)


def _make_tensors(device, plan):
    """Build x, 1/rms, gamma, out for one plan.  Fresh every variant: two_pass rewrites x in
    place and the fused variants expand gamma in place, so no tensor survives a run."""
    r_tiles, s = plan["row_tiles"], plan["S"]
    rows = _NUM_CORES * r_tiles * TILE
    width = s * TILE

    torch.manual_seed(11)
    x = (torch.rand(rows, width) * 2 - 1).to(torch.bfloat16)
    # 1/rms: column 0 of each stat tile carries the per-row reciprocal; the other 31 columns are
    # deliberately garbage, so a variant that does not honour the Col broadcast fails PCC.
    stat = torch.rand(rows, TILE) * 2 - 1
    stat[:, 0] = torch.rand(rows) + 0.5  # [0.5, 1.5)
    # gamma: ROW 0 of each core's tile-row carries the [W] vector; rows 1..31 are garbage, so a
    # variant that does not honour the Row broadcast (or expands gamma wrongly) fails PCC.
    gamma_full = (torch.rand(_NUM_CORES * TILE, width) * 2 - 1).to(torch.bfloat16)
    g = (torch.rand(width) + 0.5).to(torch.bfloat16)
    gamma_full[0::TILE, :] = g

    expected = _quant(x) * stat[:, 0:1].to(torch.float32) * _quant(g).unsqueeze(0)

    def _dev(t, dtype):
        return ttnn.from_torch(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            # per-core shard = the tensor's rows split evenly across the grid
            memory_config=sharded_memory_config((t.shape[0] // _NUM_CORES, t.shape[1])),
        )

    x_dev = _dev(x, ttnn.bfloat16)
    stat_dev = _dev(stat.to(torch.float32), ttnn.float32)
    gamma_dev = _dev(gamma_full, ttnn.bfloat16)
    out_dev = ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, width]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        sharded_memory_config((r_tiles * TILE, width)),
    )
    return x_dev, stat_dev, gamma_dev, out_dev, expected


def _pcc(actual, expected):
    a = actual.flatten().to(torch.float64)
    e = expected.flatten().to(torch.float64)
    return torch.corrcoef(torch.stack([a, e]))[0, 1].item()


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


@pytest.mark.parametrize("case", _cases(), ids=lambda c: f"B{c[0]}_S{c[1]}" + (f"_R{c[2]}" if c[2] else ""))
def test_scale_gamma_fusion(device, case):
    b, s, row_tiles = case
    dest_cap = int(os.environ.get("SGF_DEST_CAP", DEST_AUTO_LIMIT))
    plan = plan_for(b, s, row_tiles=row_tiles)
    results = {}

    for variant in _variants():
        x, stat, gamma, out, expected = _make_tensors(device, plan)
        run_variant(x, stat, gamma, out, variant=variant, plan=plan, dest_cap=dest_cap)
        ttnn.synchronize_device(device)
        ns = _read_kernel_ns(device)
        actual = ttnn.to_torch(out).to(torch.float32)
        pcc = _pcc(actual, expected)
        results[variant] = (ns, pcc)
        if not (pcc >= 0.99):  # nan-safe
            logger.error(
                f"{variant}: pcc={pcc} actual[min,max,mean]="
                f"[{actual.min():.4g},{actual.max():.4g},{actual.mean():.4g}] "
                f"expected[min,max,mean]=[{expected.min():.4g},{expected.max():.4g},{expected.mean():.4g}]"
            )
        for t in (x, stat, gamma, out):
            ttnn.deallocate(t)

    base_ns = results.get(BASELINE, (None, None))[0]
    logger.info(
        f"\n=== scale/gamma fusion | B={b} S={s} R={plan['row_tiles']} "
        f"blocks={plan['num_blocks']} tiles/core={plan['capacity']} "
        f"cores={_NUM_CORES} dest_cap={dest_cap} ==="
    )
    logger.info(f"{'variant':<16}{'dest_blk':>9}{'ns':>12}{'vs base':>10}   pcc")
    for variant, (ns, pcc) in results.items():
        rel = f"{ns / base_ns:.3f}x" if (base_ns and ns) else "-"
        logger.info(f"{variant:<16}{dest_block_for(variant, s, dest_cap):>9}{ns:>12.0f}{rel:>10}   {pcc:.5f}")

    bad = {v: pcc for v, (_, pcc) in results.items() if not (pcc >= 0.99)}  # nan-safe
    assert not bad, f"variants failed correctness (pcc < 0.99): {bad}"
