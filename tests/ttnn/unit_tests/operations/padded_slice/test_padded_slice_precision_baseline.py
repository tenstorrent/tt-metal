# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline of padded_slice (Phase 0: bfloat16, row-major, DRAM -> L1 height shards).

The op moves values and computes nothing (spec R1; NUMERIC: every cell `exact`), so the baseline is a
proof of exactness on shapes from small to large: PCC, abs errors, relative RMS, max ULP distance and the
got/true ratio spread (the scale-bug detector). Each case writes its numbers to
`precision_baseline_results/<id>.json` beside this file (a test's own output is not shown under the
simulator's workers) and records them with `record_property`.
"""

import json
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from ttnn.operations.padded_slice import padded_slice

RESULTS_DIR = Path(__file__).parent / "precision_baseline_results"


def _config(device, rows, width):
    """The runner's rule: the largest number of the device's cores that divides the rows."""
    grid = device.compute_with_storage_grid_size()
    cores = max(k for k in range(1, grid.x * grid.y + 1) if rows % k == 0)
    return ttnn.create_sharded_memory_config(
        shape=(rows // cores, width),
        core_grid=ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _ulp_max(got, expected):
    """Largest distance in bfloat16 units in the last place (ordered integer encoding)."""

    def ordered(t):
        bits = t.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)

    return int((ordered(got) - ordered(expected)).abs().max())


@pytest.mark.parametrize(
    "shape,start,end",
    [
        ((1, 4, 4, 16), (0, 0, 0, 0), (1, 2, 4, 16)),  # small: 8 rows of 16
        ((1, 64, 32, 32), (0, 0, 0, 0), (1, 32, 32, 32)),  # medium: traced call, 1024 rows
        ((1, 17, 50, 64), (0, 3, 7, 0), (1, 12, 40, 64)),  # odd starts in H and W: 297 rows
        ((1, 115, 115, 32), (0, 0, 0, 0), (1, 59, 115, 32)),  # large: the ResNet-50 stem, 6785 rows
    ],
    ids=["small", "medium", "odd", "stem"],
)
def test_precision_baseline(device, shape, start, end, request, record_property):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    c = end[3] - start[3]
    expected = torch_input[start[0] : end[0], start[1] : end[1], start[2] : end[2], start[3] : end[3]]
    expected = expected.reshape(1, 1, -1, c)
    config = _config(device, expected.shape[2], c)

    got = ttnn.to_torch(padded_slice(tt_input, start, end, [1, 1, 1, 1], memory_config=config))

    _, pcc = comp_pcc(expected.float(), got.float())
    _, allclose = comp_allclose(expected.float(), got.float())
    err = (got.double() - expected.double()).abs()
    rel_rms = float(err.pow(2).mean().sqrt() / expected.double().std())
    nz = expected.double() != 0
    ratio = got.double()[nz] / expected.double()[nz]
    result = {
        "shape": list(shape), "start": list(start), "end": list(end), "rows": int(expected.shape[2]),
        "pcc": float(pcc), "max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
        "relative_rms": rel_rms, "ulp_max": _ulp_max(got, expected),
        "ratio_median": float(ratio.median()),
        "ratio_p5": float(ratio.quantile(0.05)), "ratio_p95": float(ratio.quantile(0.95)),
        "allclose": allclose,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{request.node.callspec.id}.json").write_text(json.dumps(result, indent=1))
    for key, value in result.items():
        record_property(key, value)

    # A movement op: exact (NUMERIC_RESOLVED: `exact` for every cell).
    assert torch.equal(got, expected), result
    assert result["ulp_max"] == 0 and result["relative_rms"] == 0.0 and pcc >= 0.9999, result
