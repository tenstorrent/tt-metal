# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of paged_update_cache over the supported cells (bfloat16, TILE, L1 height-sharded, one user).

The op moves values and computes nothing (rule R1); its numeric requirement is `exact` (NUMERIC_RESOLVED). This
measures PCC, max / mean abs error, relative RMS (RMS error / std of the reference, as eval/metrics.py's
check_output), the largest ULP distance, and the got/true ratio spread (the scale-bug detector), on the whole
cache and on the region written. Results go to record_property and to
generated/paged_update_cache_precision_baseline.json (test output is not shown under the simulator's workers).
"""

import json
from pathlib import Path

import pytest
import torch
import ttnn

from eval.specs.paged_update_cache.spec import reference
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.paged_update_cache import paged_update_cache

TILE = 32
RESULTS = Path("generated") / "paged_update_cache_precision_baseline.json"

# (cache shape, input rows, position, page table): the two traced calls, the 3x3-tile entry, the widest entry.
SHAPES = [
    pytest.param((128, 8, 32, 64), 8, 77, None, id="llama_128x8x32x64"),
    pytest.param((8, 8, 32, 128), 8, 8, None, id="qwen_8x8x32x128"),
    pytest.param((6, 12, 96, 96), 12, 401, None, id="6x12x96x96"),
    pytest.param((8, 4, 32, 512), 4, 255, None, id="8x4x32x512"),
]


def _ulp_distance_bf16(a, b):
    """Largest distance in bfloat16 units in the last place (ordered-integer view of the bits)."""

    def ordered(t):
        bits = t.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return int((ordered(a) - ordered(b)).abs().max())


def _metrics(got, expected):
    _, pcc = comp_pcc(expected, got, pcc=0.0)
    _, allclose_msg = comp_allclose(expected, got)
    diff = (got - expected).abs()
    rel_rms = float(torch.sqrt((diff**2).mean()) / expected.std()) if expected.numel() > 1 else 0.0
    mask = torch.isfinite(expected) & (expected != 0)
    ratio = (got[mask] / expected[mask]).float()
    return {
        "pcc": float(pcc),
        "max_abs_err": float(diff.max()),
        "mean_abs_err": float(diff.mean()),
        "rel_rms": rel_rms,
        "max_ulp": _ulp_distance_bf16(got, expected),
        "ratio_median": float(ratio.median()),
        "ratio_p5": float(torch.quantile(ratio, 0.05)),
        "ratio_p95": float(torch.quantile(ratio, 0.95)),
        "allclose": allclose_msg,
    }


@pytest.mark.parametrize("cache_shape,rows,position,table", SHAPES)
def test_precision_baseline(device, record_property, cache_shape, rows, position, table):
    blocks, heads, block_size, D = cache_shape
    torch.manual_seed(1234)
    cache = torch.randn(cache_shape).to(torch.bfloat16)
    x = torch.randn(1, 1, rows, D).to(torch.bfloat16)
    pt = torch.randperm(blocks).reshape(1, blocks).to(torch.int32) if table is None else torch.tensor(table)
    idx = torch.tensor([position], dtype=torch.int32)

    shard = ttnn.create_sharded_memory_config(
        (TILE, D),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    dram = ttnn.DRAM_MEMORY_CONFIG
    cache_dev = ttnn.from_torch(cache, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram)
    out = paged_update_cache(
        cache_dev,
        ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=shard),
        update_idxs_tensor=ttnn.from_torch(idx, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                           memory_config=dram),
        page_table=ttnn.from_torch(pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                   memory_config=dram),
    )
    got = ttnn.to_torch(out).to(torch.float64)
    expected = reference(cache, x, idx, pt)

    block = int(pt[0, position // block_size])
    region = (block, slice(None), position % block_size, slice(None))
    result = {"whole_cache": _metrics(got, expected), "written_region": _metrics(got[region], expected[region])}
    for scope, m in result.items():
        for key, value in m.items():
            record_property(f"{scope}.{key}", value)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    existing["x".join(map(str, cache_shape))] = result
    RESULTS.write_text(json.dumps(existing, indent=1, sort_keys=True))

    # The requirement is `exact`: PCC 1, relative RMS 0, 0 ULP, every ratio exactly 1.
    for scope, m in result.items():
        assert m["max_ulp"] == 0 and m["max_abs_err"] == 0.0 and m["rel_rms"] == 0.0, (scope, m)
        assert m["pcc"] >= 0.9999, (scope, m)
        assert m["ratio_p5"] == m["ratio_p95"] == 1.0, (scope, m)
