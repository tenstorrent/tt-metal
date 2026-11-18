# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regime-pinned rms_norm tests: R2 (streamed two-pass), streamed gamma, Rb>1, ragged cores."""
import pytest
import ttnn

from tests.ttnn.unit_tests.operations.rms_norm.test_rms_norm import run

CASES = [
    ((1, 1, 32, 8192), ttnn.float32),  # R2 (fp32 row > X_BUDGET), streamed gamma
    ((2, 1, 64, 4096), ttnn.float32),  # R1 depth-1, gamma streamed
    ((1, 1, 64, 12288), ttnn.bfloat16),  # R2 bf16
    ((1, 1, 32, 8200), ttnn.float32),  # R2 + W tail
    ((3, 5, 40, 33), ttnn.bfloat16),  # Rb>1, many cores, ragged
    ((1, 1, 2048, 256), ttnn.bfloat16),  # many row blocks
    ((1, 1, 32, 17 * 32), ttnn.float32),  # prime Wt (Sb=1)
]


@pytest.mark.parametrize("shape,dtype", CASES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("use_gamma", [False, True], ids=["no_gamma", "gamma"])
def test_rms_norm_regimes(device, shape, dtype, layout, use_gamma):
    run(device, shape, dtype, layout, use_gamma)
