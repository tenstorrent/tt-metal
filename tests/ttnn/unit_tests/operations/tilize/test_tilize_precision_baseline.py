# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for the tilize op (ROW_MAJOR -> TILE layout conversion).

tilize does NO arithmetic — it only reorders bytes. So for value-exact dtypes
(bfloat16, float32) the read-back must be bit-identical (PCC = 1.0, max abs
err = 0). The only lossy path is a value-preserving cast into bfloat8_b at pack
time. This test records PCC, max/mean abs error and relative RMS error across a
small shape sweep so refinements have a numerical reference point.

Run:
  scripts/run_safe_pytest.sh --run-all \
    tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose

from ttnn.operations.tilize import tilize


# small / medium / larger / non-square — kept tile-aligned (op does NOT pad).
SHAPES = [
    (1, 1, 32, 32),
    (1, 1, 64, 128),
    (2, 3, 128, 256),
    (1, 1, 512, 512),
]

# tilize preserves values, so exact dtypes must hit PCC ~1.0; bf8b is lossy.
PCC_THRESHOLD = {
    ttnn.bfloat16: 0.9999,
    ttnn.float32: 0.9999,
    ttnn.bfloat8_b: 0.99,
}


def _make_input(dtype, shape):
    torch.manual_seed(1234)
    if dtype == ttnn.float32:
        return torch.randn(shape, dtype=torch.float32)
    return torch.randn(shape).bfloat16()


def _metrics(golden, calculated):
    g = golden.float()
    c = calculated.float()
    diff = (g - c).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rms = diff.pow(2).mean().sqrt().item()
    denom = g.pow(2).mean().sqrt().item()
    rel_rms = rms / denom if denom > 0 else 0.0
    return max_abs, mean_abs, rel_rms


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize(
    "in_dtype,out_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.float32, ttnn.float32),
        (ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_tilize_precision(device, shape, in_dtype, out_dtype):
    torch_input = _make_input(in_dtype, shape)

    tt_input = ttnn.from_torch(torch_input, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_output = tilize(tt_input, dtype=out_dtype)

    assert tt_output.layout == ttnn.TILE_LAYOUT
    assert tt_output.dtype == out_dtype

    output = ttnn.to_torch(tt_output).to(torch_input.dtype)

    max_abs, mean_abs, rel_rms = _metrics(torch_input, output)
    _, allclose_msg = comp_allclose(torch_input, output)
    print(
        f"\n[tilize precision] shape={tuple(shape)} {in_dtype}->{out_dtype} "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel_rms={rel_rms:.3e} "
        f"| {allclose_msg}"
    )

    assert_with_pcc(torch_input, output, PCC_THRESHOLD[out_dtype])

    # Value-exact dtypes: tilize must be bit-identical (no arithmetic).
    if out_dtype in (ttnn.bfloat16, ttnn.float32) and in_dtype == out_dtype:
        assert max_abs == 0.0, f"tilize must be value-exact for {out_dtype}, got max_abs={max_abs}"
