# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of paged_fill_cache (verifier).

The op moves whole tiles and computes nothing (rule R1), and every cell is graded `exact`
(NUMERIC_RESOLVED: class `movement`). So the baseline is a bit-exactness statement, measured the
usual way: PCC (comp_pcc), max / mean abs error (comp_allclose), relative RMS (RMS error / std of the
reference, as eval.metrics.check_output) and the got/true ratio spread over the written region (the
scale-bug detector: a tight cluster at a constant other than 1.0 would be a structural bug).

The numbers are recorded with `record_property` and, when PFC_PRECISION_OUT is set, appended to that
file as JSON lines (test output is not shown under the simulator's workers).
"""

import json
import os

import pytest
import torch

import ttnn
from eval.specs.paged_fill_cache.spec import reference
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.paged_fill_cache import paged_fill_cache

TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16}


def _written_mask(cache_shape, seq, table_row):
    """The cache elements the call writes (entries other than -1, positions below seq)."""
    mask = torch.zeros(cache_shape, dtype=torch.bool)
    block_size = cache_shape[2]
    for k, start in enumerate(range(0, seq, block_size)):
        if table_row[k] != -1:
            mask[table_row[k], :, : min(block_size, seq - start), :] = True
    return mask


@pytest.mark.parametrize(
    "cache_shape,input_shape,table,batch_idx",
    [
        # small: the traced Qwen3-VL call (16 tiles, one per cluster)
        ((8, 2, 32, 128), (1, 2, 64, 128), [[5, 2, 7, 0]], 0),
        # medium: a block and a half, last of three users (a partly filled block, R2)
        ((6, 3, 64, 96), (1, 3, 96, 96), [[0, 1], [2, 3], [4, 1]], 2),
        # larger: the captured Llama 3.2 1B call (256 tiles, 8 per cluster)
        ((128, 8, 32, 64), (1, 8, 512, 64), [list(range(127, 111, -1))], 0),
        # wide: one tile row 256 tiles wide, split over the whole grid
        ((2, 1, 32, 8192), (1, 1, 32, 8192), [[1], [0]], 0),
    ],
    ids=["small_qwen", "medium_partial_block", "large_llama", "wide_8192"],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
def test_precision_baseline(device, record_property, cache_shape, input_shape, table, batch_idx, dtype):
    torch.manual_seed(1234)
    tdt = TORCH_DTYPE[dtype]
    cache = torch.randn(cache_shape).to(tdt)
    x = torch.randn(input_shape).to(tdt)
    pt = torch.tensor(table, dtype=torch.int32)

    tile = dict(dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tc, tx = ttnn.from_torch(cache, **tile), ttnn.from_torch(x, **tile)
    tp = ttnn.from_torch(
        pt, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = paged_fill_cache(tc, tx, tp, batch_idx=batch_idx)

    got = ttnn.to_torch(out).double()
    expected = reference(cache, x, pt, batch_idx)

    _, pcc = comp_pcc(expected, got, 0.9999)
    _, allclose_msg = comp_allclose(expected, got)
    err = got - expected
    max_abs = float(err.abs().max())
    mean_abs = float(err.abs().mean())
    rel_rms = float(err.pow(2).mean().sqrt() / expected.std())

    mask = _written_mask(cache_shape, input_shape[2], table[batch_idx]) & (expected != 0) & torch.isfinite(expected)
    ratio = got[mask] / expected[mask]
    ratio_median = float(ratio.median())
    ratio_p5, ratio_p95 = (float(torch.quantile(ratio, q)) for q in (0.05, 0.95))

    row = {
        "shape": f"{list(cache_shape)}<-{list(input_shape)}",
        "dtype": str(dtype),
        "pcc": float(pcc),
        "max_abs_err": max_abs,
        "mean_abs_err": mean_abs,
        "rel_rms": rel_rms,
        "ratio_median": ratio_median,
        "ratio_p5": ratio_p5,
        "ratio_p95": ratio_p95,
        "written_elements": int(mask.sum()),
        "comp_allclose": allclose_msg,
    }
    for k, v in row.items():
        record_property(k, v)
    if os.environ.get("PFC_PRECISION_OUT"):
        with open(os.environ["PFC_PRECISION_OUT"], "a") as f:
            f.write(json.dumps(row) + "\n")

    # Graded `exact`: the whole cache, bit for bit (R1, R2), and the ratio is exactly 1 everywhere written.
    assert torch.equal(got, expected), row
    assert ratio_p5 == ratio_p95 == 1.0, row
    assert pcc >= 0.9999 and rel_rms == 0.0, row
