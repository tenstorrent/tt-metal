# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU: `assert_quality`, which gives every device test its verdict. It passes a close output and fails
the errors PCC alone does not see: a scale, an offset, and a few NaN or Inf values (`comp_pcc` sets
them to zero)."""

from __future__ import annotations

import pytest
import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import assert_quality, quality, random_tensor

RELATIVE_RMSE = 0.003


@pytest.fixture(scope="module")
def golden() -> torch.Tensor:
    return random_tensor(256, 128, dtype=torch.float32)


def test_passes_close_output(golden):
    noise = random_tensor(*golden.shape, dtype=torch.float32, seed=1)
    q = assert_quality(golden, golden + 1e-4 * noise, pcc=0.9999, relative_rmse=RELATIVE_RMSE)
    assert q["nonfinite"] == 0


SCALE_AND_OFFSET = {
    "scale_1.01": lambda g: 1.01 * g,
    "scale_0.5": lambda g: 0.5 * g,
    "offset": lambda g: g + 0.1 * g.std(),
}


@pytest.mark.parametrize("error", SCALE_AND_OFFSET.values(), ids=SCALE_AND_OFFSET.keys())
def test_fails_scale_and_offset(golden, expect_error, error):
    """PCC is 1 for these, so only the relative RMSE limit fails them."""
    actual = error(golden)
    assert quality(golden, actual)["pcc"] > 0.9999
    with expect_error(AssertionError, "relative RMSE"):
        assert_quality(golden, actual, pcc=0.9999, relative_rmse=RELATIVE_RMSE)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
@pytest.mark.parametrize("relative_rmse", [RELATIVE_RMSE, None], ids=["rmse_limit", "pcc_only"])
def test_fails_one_nonfinite_value(golden, expect_error, value, relative_rmse):
    """One NaN or Inf fails, also without a relative RMSE limit, where PCC alone would pass it."""
    actual = golden.clone()
    actual[3, 5] = value
    assert quality(golden, actual)["pcc"] > 0.9999
    with expect_error(AssertionError, "1 NaN or Inf values"):
        assert_quality(golden, actual, pcc=0.9999, relative_rmse=relative_rmse)
