# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for independent Chronos-2 Step 0 packing in tt/model_preprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from models.experimental.chronos_forecast.tt.model_preprocessing import (
    encode_categorical_covariate,
    instance_norm,
    instance_norm_inverse,
    normalize_chronos2_inputs,
    prepare_chronos2_inputs,
    target_encode,
)

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


def test_target_encode_unseen_future_falls_back_to_item_mean():
    target = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    _, encoded_future = target_encode(
        id_codes=np.array([0, 0, 0, 1, 1, 1]),
        cat_codes=np.array([0, 1, 0, 0, 1, 1]),
        target=target,
        n_items=2,
        n_categories=2,
        future_id_codes=np.array([0, 1]),
        future_cat_codes=np.array([2, 2]),
        smooth=1.0,
    )
    np.testing.assert_array_almost_equal(encoded_future, [2.0, 5.0], decimal=5)


def test_target_encode_seen_category_uses_smoothed_mean():
    encoded_past, _ = target_encode(
        id_codes=np.array([0, 0, 0, 0]),
        cat_codes=np.array([0, 0, 1, 1]),
        target=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        n_items=1,
        n_categories=2,
        smooth=1.0,
    )
    np.testing.assert_array_almost_equal(encoded_past, [18.3333, 18.3333, 31.6667, 31.6667], decimal=3)


def test_target_encode_handles_nans_in_target():
    encoded_past, _ = target_encode(
        id_codes=np.array([0, 0, 0, 0]),
        cat_codes=np.array([0, 0, 1, 1]),
        target=np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32),
        n_items=1,
        n_categories=2,
        smooth=1.0,
    )
    np.testing.assert_array_almost_equal(encoded_past, [1.8333, 1.8333, 3.2222, 3.2222], decimal=3)
    assert np.isfinite(encoded_past).all()


def test_encode_categorical_unseen_future_is_item_mean():
    target = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    past = np.array(["a", "b", "a", "b"])
    future = np.array(["zzz_unseen", "zzz_unseen"])
    _, enc_future = encode_categorical_covariate(past, target=target, future=future)
    np.testing.assert_array_almost_equal(enc_future, np.full(2, target.mean()), decimal=4)


def test_prepare_inputs_target_encodes_string_covariate():
    rng = np.random.default_rng(123)
    target = rng.standard_normal(20).astype(np.float32)
    past_cat = np.array(["a", "b", "a", "b", "a"] * 4)
    future_cat = np.array(["zzz_unseen"] * 5)
    packed = prepare_chronos2_inputs(
        target,
        prediction_length=5,
        past_covariates=past_cat,
        future_covariates=future_cat,
    )
    np.testing.assert_array_almost_equal(
        packed.future_covariates[-1].numpy(),
        np.full(5, float(target.mean()), dtype=np.float32),
        decimal=4,
    )


def test_prepare_inputs_multivariate_falls_back_to_ordinal():
    rng = np.random.default_rng(7)
    packed = prepare_chronos2_inputs(
        rng.standard_normal((2, 10)).astype(np.float32),
        prediction_length=5,
        past_covariates=np.array(["a", "b"] * 5),
        future_covariates=np.array(["a", "b", "a", "b", "a"]),
        use_target_encoding=True,
    )
    assert set(np.unique(packed.context[-1].numpy()).tolist()).issubset({0.0, 1.0})
    assert set(np.unique(packed.future_covariates[-1].numpy()).tolist()).issubset({0.0, 1.0})


def test_prepare_inputs_nan_is_its_own_target_encoded_category():
    target = np.array([10.0, 0.0] * 6, dtype=np.float32)
    packed = prepare_chronos2_inputs(
        target,
        prediction_length=4,
        past_covariates=np.array(["x", None] * 6, dtype=object),
        future_covariates=np.array([None] * 4, dtype=object),
    )
    x_enc, nan_enc = packed.context[-1].numpy()[:2]
    future_row = packed.future_covariates[-1].numpy()
    assert nan_enc < target.mean() < x_enc
    np.testing.assert_array_almost_equal(future_row, np.full(4, nan_enc), decimal=5)


def test_prepare_inputs_ordinal_unseen_future_is_nan():
    rng = np.random.default_rng(8)
    packed = prepare_chronos2_inputs(
        rng.standard_normal((2, 10)).astype(np.float32),
        prediction_length=5,
        past_covariates=np.array(["a", "b"] * 5),
        future_covariates=np.array(["a", "zzz_unseen", "a", "b", "a"]),
    )
    future_cat_row = packed.future_covariates[-1].numpy()
    assert np.isnan(future_cat_row[1])
    assert np.isfinite(future_cat_row[[0, 2, 3, 4]]).all()


def test_oracle_from_list_of_dicts_categorical():
    from models.experimental.chronos_forecast.common.chronos_src import ensure_chronos_on_path

    ensure_chronos_on_path()
    from chronos.chronos2.preprocess import from_list_of_dicts

    rng = np.random.default_rng(123)
    target = rng.standard_normal(20).astype(np.float32)
    past_cat = np.array(["a", "b", "a", "b", "a"] * 4)
    future_cat = np.array(["zzz_unseen"] * 5)
    data = [
        {
            "target": target,
            "past_covariates": {"cat": past_cat},
            "future_covariates": {"cat": future_cat},
        }
    ]
    amazon = from_list_of_dicts(data, prediction_length=5, use_target_encoding=True)[0]
    packed = prepare_chronos2_inputs(
        target,
        prediction_length=5,
        past_covariates=past_cat,
        future_covariates=future_cat,
        use_target_encoding=True,
    )
    torch.testing.assert_close(packed.context, amazon["context"], equal_nan=True, atol=0, rtol=0)
    torch.testing.assert_close(
        packed.future_covariates, amazon["future_covariates"], equal_nan=True, atol=0, rtol=0
    )


def test_instance_norm_matches_standardization_formula():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0]])
    normalized, (loc, scale) = instance_norm(x)
    torch.testing.assert_close(normalized[0], normalized[1])
    torch.testing.assert_close(loc.squeeze(), torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(scale.squeeze(), torch.tensor([1.41421, 1.41421]), atol=1e-4, rtol=1e-4)


def test_instance_norm_preserves_nans_and_inverse():
    x = torch.tensor([[1.0, float("nan"), 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, float("nan")]])
    normalized, loc_scale = instance_norm(x)
    assert torch.equal(normalized.isnan(), x.isnan())
    restored = instance_norm_inverse(normalized, loc_scale)
    torch.testing.assert_close(restored, x, equal_nan=True)


def test_instance_norm_zero_variance_uses_eps():
    x = torch.ones(1, 4)
    _, (loc, scale) = instance_norm(x, eps=1e-5)
    torch.testing.assert_close(loc, torch.ones(1, 1))
    torch.testing.assert_close(scale, torch.full((1, 1), 1e-5))


def test_instance_norm_all_nan_row():
    x = torch.full((1, 3), float("nan"))
    y, (loc, scale) = instance_norm(x)
    torch.testing.assert_close(loc, torch.zeros(1, 1))
    torch.testing.assert_close(scale, torch.ones(1, 1))
    assert torch.isnan(y).all()


def test_instance_norm_arcsinh_and_inverse():
    torch.manual_seed(0)
    x = torch.randn(2, 8)
    y, loc_scale = instance_norm(x, use_arcsinh=True)
    y_no, _ = instance_norm(x, use_arcsinh=False)
    assert not torch.allclose(y, y_no)
    restored = instance_norm_inverse(y, loc_scale, use_arcsinh=True)
    torch.testing.assert_close(restored, x, atol=1e-5, rtol=1e-5)


def test_instance_norm_oracle_vs_amazon():
    from models.experimental.chronos_forecast.common.chronos_src import ensure_chronos_on_path

    ensure_chronos_on_path()
    from chronos.chronos_bolt import InstanceNorm as UpInstanceNorm

    torch.manual_seed(0)
    x = torch.randn(3, 16)
    x[0, 3] = float("nan")
    for use_arcsinh in (False, True):
        up = UpInstanceNorm(use_arcsinh=use_arcsinh)
        ref_y, ref_ls = up(x)
        y, ls = instance_norm(x, use_arcsinh=use_arcsinh)
        torch.testing.assert_close(y, ref_y, equal_nan=True, atol=0, rtol=0)
        torch.testing.assert_close(ls[0], ref_ls[0], atol=0, rtol=0)
        torch.testing.assert_close(ls[1], ref_ls[1], atol=0, rtol=0)
        torch.testing.assert_close(
            instance_norm_inverse(y, ls, use_arcsinh=use_arcsinh),
            up.inverse(ref_y, ref_ls),
            equal_nan=True,
            atol=0,
            rtol=0,
        )


def test_normalize_packed_inputs_reuses_context_loc_scale_on_future():
    packed = prepare_chronos2_inputs(torch.arange(8, dtype=torch.float32), prediction_length=4)
    assert packed.loc_scale is None
    normed = normalize_chronos2_inputs(packed)
    ctx, loc_scale = instance_norm(packed.context)
    fut, _ = instance_norm(packed.future_covariates, loc_scale)
    torch.testing.assert_close(normed.context, ctx, equal_nan=True)
    torch.testing.assert_close(normed.future_covariates, fut, equal_nan=True)
    torch.testing.assert_close(normed.loc_scale[0], loc_scale[0])
    torch.testing.assert_close(normed.loc_scale[1], loc_scale[1])


def test_prepare_inputs_apply_instance_norm_flag():
    target = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    raw = prepare_chronos2_inputs(target, prediction_length=2)
    normed = prepare_chronos2_inputs(target, prediction_length=2, apply_instance_norm=True)
    expected, loc_scale = instance_norm(raw.context)
    torch.testing.assert_close(normed.context, expected)
    torch.testing.assert_close(normed.loc_scale[0], loc_scale[0])
    torch.testing.assert_close(normed.loc_scale[1], loc_scale[1])
    fut, _ = instance_norm(raw.future_covariates, loc_scale)
    torch.testing.assert_close(normed.future_covariates, fut, equal_nan=True)
