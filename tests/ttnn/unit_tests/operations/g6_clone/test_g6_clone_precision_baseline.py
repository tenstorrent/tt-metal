# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for g6_clone over the Phase 0 SUPPORTED cells (float32, TILE).

g6_clone moves bytes and computes nothing, so the baseline is bit equality. PCC, abs errors, relative
RMS, max ULP and the got/true ratio spread are recorded as well so the report can show the op sits at
the exact end of every metric; the gate is torch.equal on the bit patterns.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc, comp_ulp
from ttnn.operations.g6_clone import g6_clone

SHAPES = [
    pytest.param((1, 1), id="1x1_partial"),
    pytest.param((1, 12, 128, 128), id="1x12x128x128"),
    pytest.param((8, 12, 128, 128), id="8x12x128x128"),
    pytest.param((16, 12, 128, 128), id="16x12x128x128"),
]


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    _, pcc = comp_pcc(expected, actual, pcc=0.0)
    _, allclose_msg = comp_allclose(expected, actual)
    _, ulp_msg = comp_ulp(expected, actual, ulp_threshold=0)
    diff = (actual - expected).abs()
    std = expected.std().item() if expected.numel() > 1 else 0.0
    rms = diff.pow(2).mean().sqrt().item()
    rel_rms = rms / std if std > 1e-12 else rms
    nonzero = expected != 0
    ratio = actual[nonzero] / expected[nonzero]
    return {
        "pcc": float(pcc),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "rel_rms": rel_rms,
        "allclose": allclose_msg,
        "ulp": ulp_msg,
        "ratio_median": ratio.median().item() if ratio.numel() else float("nan"),
        "ratio_p5": torch.quantile(ratio.float(), 0.05).item() if ratio.numel() else float("nan"),
        "ratio_p95": torch.quantile(ratio.float(), 0.95).item() if ratio.numel() else float("nan"),
    }


@pytest.mark.parametrize("shape", SHAPES)
def test_g6_clone_precision_baseline(device, shape):
    torch.manual_seed(1234)
    torch_input = torch.randn(shape, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=True,
    )
    actual = ttnn.to_torch(g6_clone(ttnn_input))
    expected = torch_input.clone()

    m = _metrics(actual, expected)
    print(
        f"PRECISION {shape}: pcc={m['pcc']} max_abs={m['max_abs']} mean_abs={m['mean_abs']} "
        f"rel_rms={m['rel_rms']} ratio median={m['ratio_median']} p5={m['ratio_p5']} p95={m['ratio_p95']} "
        f"| {m['allclose']} | {m['ulp']}"
    )

    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), "g6_clone is not bit-exact"
    assert m["max_abs"] == 0.0 and m["rel_rms"] == 0.0
    assert m["ratio_median"] == m["ratio_p5"] == m["ratio_p95"] == 1.0
