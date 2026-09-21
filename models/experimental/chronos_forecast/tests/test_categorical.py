# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Golden stub Block 2: categorical encoding on V/W.

Oracle: tt/model_preprocessing target/ordinal encoding (done).
Single-target series use smoothed per-item target means; multivariate falls
back to ordinal codes; NaN is its own category; unseen future -> item mean.
"""

from __future__ import annotations

import numpy as np
import torch

from models.experimental.chronos_forecast.tests.golden_helpers import log_golden
from models.experimental.chronos_forecast.tt.model_preprocessing import (
    encode_categorical_covariate,
    prepare_chronos2_inputs,
    target_encode,
)


def test_block2_target_encoding_golden():
    encoded_past, _ = target_encode(
        id_codes=np.array([0, 0, 0, 0]),
        cat_codes=np.array([0, 0, 1, 1]),
        target=np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        n_items=1,
        n_categories=2,
        smooth=1.0,
    )
    # smoothed: (1*item_mean + cat_sum) / (1 + cat_count)
    np.testing.assert_allclose(encoded_past, [18.3333, 18.3333, 31.6667, 31.6667], rtol=1e-3)
    log_golden("block2/target_encoded", torch.from_numpy(encoded_past))


def test_block2_unseen_future_falls_back_to_item_mean():
    target = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    _, enc_future = encode_categorical_covariate(
        np.array(["a", "b", "a", "b"]),
        target=target,
        future=np.array(["zzz_unseen", "zzz_unseen"]),
    )
    np.testing.assert_allclose(enc_future, np.full(2, target.mean()), rtol=1e-4)
    log_golden("block2/unseen_future", torch.from_numpy(enc_future))


def test_block2_end_to_end_string_covariate():
    rng = np.random.default_rng(123)
    target = rng.standard_normal(20).astype(np.float32)
    packed = prepare_chronos2_inputs(
        target,
        prediction_length=5,
        past_covariates=np.array(["a", "b", "a", "b", "a"] * 4),
        future_covariates=np.array(["zzz_unseen"] * 5),
    )
    # encoded covariate is the last context row; unseen future rows -> item mean
    assert packed.context.shape[0] == 2
    log_golden("block2/V_with_cat", packed.context)
    log_golden("block2/W_with_cat", packed.future_covariates)
