# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of max_pool2d over first-phase shapes (verifier, phase 0).

max_pool2d selects an input element, so every cell is graded `exact` (the specification's NUMERIC_RESOLVED). The
baseline still records PCC, max / mean absolute error, relative RMS (RMS error / std of the reference, as
eval/metrics.py's check_output), the ULP error and the got/true ratio spread (the scale-bug detector): a tight
cluster of got/true around a constant other than 1.0 would mean a uniform scale bug rather than rounding.

Output of a test is hidden under the simulator's workers, so each case appends one JSON line to
$MAX_POOL2D_BASELINE_OUT (default /tmp/max_pool2d_precision_baseline.jsonl) and records it with record_property.
"""

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc, comp_ulp
from ttnn.operations.max_pool2d import max_pool2d

OUT_FILE = os.environ.get("MAX_POOL2D_BASELINE_OUT", "/tmp/max_pool2d_precision_baseline.jsonl")


def reference(t, n, h, w, c, k, s, p, d=(1, 1)):
    images = t.double().reshape(n, h, w, c).permute(0, 3, 1, 2)
    pooled = torch.nn.functional.max_pool2d(images, k, stride=s, padding=p, dilation=d)
    N, C, Ho, Wo = pooled.shape
    return pooled.permute(0, 2, 3, 1).reshape(1, 1, N * Ho * Wo, C)


def placement(kind, rows, channels, device):
    if kind == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if kind == "l1":
        return ttnn.L1_MEMORY_CONFIG
    grid = ttnn.num_cores_to_corerangeset(2, device.compute_with_storage_grid_size(), row_wise=True)
    return ttnn.create_sharded_memory_config(
        shape=(rows // 2, channels), core_grid=grid, strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)


# (id, (N, H, W, C, kernel, stride, padding, dilation), layout, placement): small, the YOLOv8 SPPF call, the
# ResNet-50 call, and a larger one with different H and W windows.
CASES = [
    ("small_8x8x64_k5", (1, 8, 8, 64, (5, 5), (1, 1), (2, 2), (1, 1)), ttnn.ROW_MAJOR_LAYOUT, "l1"),
    ("sppf_20x20x256_k5", (1, 20, 20, 256, (5, 5), (1, 1), (2, 2), (1, 1)), ttnn.ROW_MAJOR_LAYOUT, "dram"),
    ("resnet_32x32x64_k3s2", (1, 32, 32, 64, (3, 3), (2, 2), (1, 1), (1, 1)), ttnn.TILE_LAYOUT, "height"),
    ("large_32x64x32_k3x5", (1, 32, 64, 32, (3, 5), (2, 1), (1, 2), (1, 1)), ttnn.TILE_LAYOUT, "dram"),
]


@pytest.mark.parametrize("case_id,args,layout,where", CASES, ids=[c[0] for c in CASES])
def test_precision_baseline(device, record_property, case_id, args, layout, where):
    n, h, w, c = args[:4]
    torch.manual_seed(42)
    t = torch.randn(1, 1, n * h * w, c).to(torch.bfloat16)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device,
                        memory_config=placement(where, n * h * w, c, device))
    got = ttnn.to_torch(max_pool2d(x, *args)).double()
    expected = reference(t, *args)
    assert got.shape == expected.shape

    _, pcc = comp_pcc(expected, got, pcc=0.99999)
    _, allclose = comp_allclose(got, expected)
    _, ulp = comp_ulp(expected.to(torch.bfloat16), got.to(torch.bfloat16), ulp_threshold=0)
    err = got - expected
    rel_rms = (err.pow(2).mean().sqrt() / expected.std()).item()
    nz = (expected != 0) & torch.isfinite(expected)
    ratio = (got[nz] / expected[nz]).float()
    result = {
        "case": case_id,
        "pcc": float(pcc),
        "max_abs_err": err.abs().max().item(),
        "mean_abs_err": err.abs().mean().item(),
        "rel_rms": rel_rms,
        "allclose": allclose,
        "ulp": str(ulp),
        "ratio_median": ratio.median().item(),
        "ratio_p5": ratio.quantile(0.05).item(),
        "ratio_p95": ratio.quantile(0.95).item(),
        "mismatches": int((err != 0).sum().item()),
        "elements": err.numel(),
    }
    record_property("precision_baseline", json.dumps(result))
    with open(OUT_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")

    # Graded `exact`: every output element is the input element torch selects.
    assert result["mismatches"] == 0, result
    assert result["pcc"] >= 0.99999 and result["rel_rms"] == 0.0, result
