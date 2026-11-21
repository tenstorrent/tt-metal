# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of slice_write (verifier). The op only moves values, so the numeric requirement of every
cell is `exact` (spec NUMERIC_RESOLVED): PCC 1, every error 0 and the got/true ratio exactly 1.0 everywhere.
The measurements are written to SLICE_WRITE_BASELINE_OUT (default /tmp/slice_write_precision_baseline.jsonl),
since a test's own output is hidden under the simulator's workers."""

import json
import math
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.slice_write import slice_write

OUT = os.environ.get("SLICE_WRITE_BASELINE_OUT", "/tmp/slice_write_precision_baseline.jsonl")

# (destination, start, end, cores of the source's height-shard grid)
SHAPES = [
    ((1, 64, 32, 32), (0, 0, 0, 0), (1, 32, 32, 32), 32),       # ResNet-50 small traced call
    ((1, 115, 115, 32), (0, 0, 0, 0), (1, 59, 115, 32), 32),    # ResNet-50 traced call (H non-aligned)
    ((1, 4, 512, 32), (0, 1, 0, 0), (1, 3, 512, 32), 8),        # tall planes
    ((1, 2, 32, 4096), (0, 1, 0, 0), (1, 2, 32, 4096), 4),      # wide rows (8 KiB pages)
]


def _sharded_source(device, block, num_cores):
    src = torch.randn(*block, dtype=torch.bfloat16)
    rows = math.prod(block[:-1])
    cores = ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size(), row_wise=True)
    config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                               ttnn.ShardSpec(cores, [math.ceil(rows / num_cores), block[-1]],
                                              ttnn.ShardOrientation.ROW_MAJOR))
    return src, ttnn.from_torch(src, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                                memory_config=config)


@pytest.mark.parametrize("shape,start,end,cores", SHAPES, ids=lambda v: "x".join(map(str, v)) if isinstance(v, tuple) else str(v))
def test_precision_baseline(device, shape, start, end, cores):
    torch.manual_seed(1234)
    block = tuple(b - a for a, b in zip(start, end))
    src, src_dev = _sharded_source(device, block, cores)
    dst = torch.randn(*shape, dtype=torch.bfloat16)
    dst_dev = ttnn.from_torch(dst, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
                              memory_config=ttnn.DRAM_MEMORY_CONFIG)

    out = slice_write(src_dev, dst_dev, list(start), list(end), [1, 1, 1, 1])
    assert out is dst_dev

    expected = dst.double().clone()
    expected[tuple(slice(a, b) for a, b in zip(start, end))] = src.double()
    actual = ttnn.to_torch(out).double()

    pcc_ok, pcc = comp_pcc(expected, actual, 0.9999)
    _, allclose_msg = comp_allclose(expected, actual)
    diff = (actual - expected).abs()
    max_abs, mean_abs = float(diff.max()), float(diff.mean())
    rel_rms = float(torch.sqrt(((actual - expected) ** 2).mean()) / expected.std())
    nonzero = expected != 0
    ratio = actual[nonzero] / expected[nonzero]
    record = {"shape": list(shape), "start": list(start), "end": list(end), "cores": cores,
              "pcc": float(pcc), "max_abs": max_abs, "mean_abs": mean_abs, "rel_rms": rel_rms,
              "ratio_median": float(ratio.median()), "ratio_p5": float(ratio.quantile(0.05)),
              "ratio_p95": float(ratio.quantile(0.95)), "allclose": allclose_msg}
    with open(OUT, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Exact: the block holds the source, everything else holds the destination (R1, R2).
    assert pcc_ok and rel_rms == 0.0 and max_abs == 0.0
    assert record["ratio_p5"] == record["ratio_p95"] == 1.0
    assert torch.equal(ttnn.to_torch(src_dev), src)  # R4
