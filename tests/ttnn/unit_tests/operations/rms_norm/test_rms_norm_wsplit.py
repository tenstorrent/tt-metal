# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""W-split regime (Refinement 2): few tile-rows / wide rows are split across cores with a partial combine."""
import pytest
import ttnn

from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd
from tests.ttnn.unit_tests.operations.rms_norm.test_rms_norm import run

# (rows, W): decode, 1-row wide, wide-resilience, a ragged-tail W, and a 2-row-block case.
CASES = [
    (1, 1, 32, 5120),
    (1, 1, 32, 1024),
    (1, 1, 32, 7168),
    (1, 1, 1, 16384),  # H non-aligned (1 valid row)
    (1, 1, 32, 32768),
    (1, 1, 64, 12288),
    (1, 1, 96, 4000),  # W non-aligned: ragged last slice, Rb row groups of 1-3 rows
    (2, 1, 32, 3072),
]


def test_pick_w_split_core_counts():
    # decode / wide shapes must be spread over far more cores than their tile-row count (Rb-limited)
    for Wt, rows in [(160, 1), (512, 1), (1024, 1), (384, 2), (224, 1)]:
        n_w, n_r = pd.pick_w_split(Wt, rows, 64, 4096)
        assert n_w > 1 and n_w * n_r > rows * 4, (Wt, rows, n_w, n_r)
    # tall shapes keep the row-split path
    assert pd.pick_w_split(160, 256, 64, 4096)[0] == 1


@pytest.mark.parametrize("shape", CASES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("use_gamma", [False, True], ids=["no_gamma", "gamma"])
def test_rms_norm_wsplit(device, shape, dtype, layout, use_gamma):
    run(device, shape, dtype, layout, use_gamma)
