# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for independent Chronos-2 Step 0 packing in tt/model_preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from models.experimental.chronos_forecast.tt.model_preprocessing import prepare_chronos2_inputs

PREPROCESS_SRC = (
    Path(__file__).resolve().parents[1] / "tt" / "model_preprocessing.py"
)


def test_implementation_does_not_import_chronos():
    text = PREPROCESS_SRC.read_text()
    assert "import chronos" not in text
    assert "from chronos" not in text


def test_univariate_no_covariates():
    target = torch.arange(8, dtype=torch.float32)
    packed = prepare_chronos2_inputs(target, prediction_length=4)

    assert packed.context.shape == (1, 8)
    assert packed.future_covariates.shape == (1, 4)
    assert torch.equal(packed.context[0], target)
    assert torch.isnan(packed.future_covariates).all()
    assert torch.equal(packed.group_ids, torch.tensor([0]))
    assert packed.target_idx_ranges == [(0, 1)]


def test_multivariate_no_covariates_shared_group():
    target = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    packed = prepare_chronos2_inputs(target, prediction_length=2)

    assert packed.context.shape == (3, 4)
    assert torch.equal(packed.context, target)
    assert packed.future_covariates.shape == (3, 2)
    assert torch.isnan(packed.future_covariates).all()
    assert torch.equal(packed.group_ids, torch.zeros(3, dtype=torch.long))
    assert packed.target_idx_ranges == [(0, 3)]


def test_known_future_covariates_suffix_rows():
    target = torch.tensor([1.0, 2.0, 3.0, 4.0])
    past = torch.stack(
        [
            torch.tensor([10.0, 11.0, 12.0, 13.0]),  # past-only
            torch.tensor([20.0, 21.0, 22.0, 23.0]),  # known-future
        ]
    )
    future = torch.tensor([[30.0, 31.0, 32.0]])
    packed = prepare_chronos2_inputs(
        target,
        prediction_length=3,
        past_covariates=past,
        future_covariates=future,
    )

    assert packed.context.shape == (3, 4)
    assert torch.equal(packed.context[0], target)
    assert torch.equal(packed.context[1], past[0])
    assert torch.equal(packed.context[2], past[1])
    assert torch.isnan(packed.future_covariates[0]).all()
    assert torch.isnan(packed.future_covariates[1]).all()
    assert torch.equal(packed.future_covariates[2], future[0])
    assert packed.target_idx_ranges == [(0, 1)]


def test_two_series_left_pad_and_group_ids():
    short = torch.tensor([1.0, 2.0])
    long = torch.tensor([3.0, 4.0, 5.0, 6.0])
    packed = prepare_chronos2_inputs([short, long], prediction_length=2)

    assert packed.context.shape == (2, 4)
    assert torch.isnan(packed.context[0, :2]).all()
    assert torch.equal(packed.context[0, 2:], short)
    assert torch.equal(packed.context[1], long)
    assert packed.future_covariates.shape == (2, 2)
    assert torch.equal(packed.group_ids, torch.tensor([0, 1]))
    assert packed.target_idx_ranges == [(0, 1), (1, 2)]


def test_numpy_1d_target_accepted():
    packed = prepare_chronos2_inputs(np.arange(5, dtype=np.float32), prediction_length=1)
    assert packed.context.shape == (1, 5)
    assert packed.context.dtype == torch.float32


def test_future_without_past_raises():
    with pytest.raises(ValueError, match="future_covariates requires past_covariates"):
        prepare_chronos2_inputs(
            torch.ones(4),
            prediction_length=2,
            future_covariates=torch.ones(2),
        )


def test_future_rows_exceeding_past_raises():
    with pytest.raises(ValueError, match="suffix"):
        prepare_chronos2_inputs(
            torch.ones(4),
            prediction_length=2,
            past_covariates=torch.ones(4),
            future_covariates=torch.ones(2, 2),
        )


def test_prediction_length_required_without_future():
    with pytest.raises(ValueError, match="prediction_length is required"):
        prepare_chronos2_inputs(torch.ones(3))


def test_oracle_from_tensor_multiseries():
    from models.experimental.chronos_forecast.common.chronos_src import ensure_chronos_on_path

    ensure_chronos_on_path()
    from chronos.chronos2.preprocess import from_tensor

    series0 = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    series1 = torch.arange(8, 16, dtype=torch.float32).reshape(2, 4)
    data = torch.stack([series0, series1], dim=0)
    amazon = from_tensor(data, prediction_length=3)

    packed = prepare_chronos2_inputs([series0, series1], prediction_length=3)

    amazon_context = torch.cat([item["context"] for item in amazon], dim=0)
    amazon_future = torch.cat([item["future_covariates"] for item in amazon], dim=0)
    torch.testing.assert_close(packed.context, amazon_context, equal_nan=True)
    torch.testing.assert_close(packed.future_covariates, amazon_future, equal_nan=True)
    assert torch.equal(packed.group_ids, torch.tensor([0, 0, 1, 1]))
    assert packed.target_idx_ranges == [(0, 2), (2, 4)]


def test_oracle_from_list_of_dicts_with_covariates():
    from models.experimental.chronos_forecast.common.chronos_src import ensure_chronos_on_path

    ensure_chronos_on_path()
    from chronos.chronos2.preprocess import from_list_of_dicts

    target = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    past_only = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float32)
    known_past = np.array([20.0, 21.0, 22.0, 23.0], dtype=np.float32)
    known_future = np.array([30.0, 31.0], dtype=np.float32)

    amazon = from_list_of_dicts(
        [
            {
                "target": target,
                "past_covariates": {
                    "past_only": past_only,
                    "known_future": known_past,
                },
                "future_covariates": {"known_future": known_future},
            }
        ],
        prediction_length=2,
    )
    amazon_ctx = amazon[0]["context"]
    amazon_fut = amazon[0]["future_covariates"]

    # Tensor API uses caller order: past-only rows, then known-future suffix.
    packed = prepare_chronos2_inputs(
        torch.from_numpy(target),
        prediction_length=2,
        past_covariates=torch.stack([torch.from_numpy(past_only), torch.from_numpy(known_past)]),
        future_covariates=torch.from_numpy(known_future),
    )

    torch.testing.assert_close(packed.context, amazon_ctx, equal_nan=True)
    torch.testing.assert_close(packed.future_covariates, amazon_fut, equal_nan=True)
