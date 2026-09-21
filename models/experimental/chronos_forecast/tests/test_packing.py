# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub Block 1: raw packing V (history) / W (future).

Oracle: tt/model_preprocessing.prepare_chronos2_inputs (done, dependency-free).
Reference takes packed (B,T) context downstream, so this stub locks the V/W
contract: rows = targets + covariates, NaN where unknown, group per series.

Example: 7-day horizon, signals = demand (unknown future) + temp/holiday (known).
"""

from __future__ import annotations

import torch

from models.experimental.chronos_forecast.tests.golden_helpers import log_golden
from models.experimental.chronos_forecast.tt.model_preprocessing import prepare_chronos2_inputs


def test_block1_univariate_with_known_future():
    torch.manual_seed(0)
    t, h = 14, 7
    demand = torch.randn(t)

   # pass in known future split(0)
    past_only_hist = torch.zeros(t)
    temp_hist = torch.randn(t)
    temp_fut = torch.randn(h)

    packed = prepare_chronos2_inputs(
        demand,
        prediction_length=h,
        past_covariates=torch.stack([past_only_hist, temp_hist]),
        future_covariates=temp_fut,  # suffix row: only temp is known ahead
    )

    # V: demand + past-only + temp rows; W: NaN head for demand/past-only, known temp tail
    assert packed.context.shape == (3, t), packed.context.shape
    assert packed.future_covariates.shape == (3, h), packed.future_covariates.shape
    assert torch.isnan(packed.future_covariates[0]).all()  # demand unknown
    assert torch.isnan(packed.future_covariates[1]).all()  # past-only has no future
    torch.testing.assert_close(packed.future_covariates[2], temp_fut)
    assert torch.equal(packed.group_ids, torch.tensor([0, 0, 0]))
    assert packed.target_idx_ranges == [(0, 1)]
    log_golden("block1/V_context", packed.context)
    log_golden("block1/W_future", packed.future_covariates)


def test_block1_multiseries_shapes():
    torch.manual_seed(0)
    packed = prepare_chronos2_inputs([torch.randn(10), torch.randn(12)], prediction_length=4)
    assert packed.context.shape == (2, 12), packed.context.shape
    assert torch.isnan(packed.context[0, :2]).all()  # left-pad short series
    assert torch.equal(packed.group_ids, torch.tensor([0, 1]))
    assert packed.target_idx_ranges == [(0, 1), (1, 2)]
    log_golden("block1/multi_V", packed.context)
