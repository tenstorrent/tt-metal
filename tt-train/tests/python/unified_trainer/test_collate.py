# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""collate_causal_lm: padding, next-token targets and the loss mask. Device-free."""

import numpy as np
import pytest

from ttml.datasets.collate import IGNORED_INDEX, collate_causal_lm

PAD = 0


def example(tokens, prompt_len):
    return {"input_ids": tokens, "labels": [IGNORED_INDEX] * prompt_len + tokens[prompt_len:]}


def test_layout_and_dtypes():
    batch = collate_causal_lm([example([5, 6, 7], 1), example([8, 9], 1)], max_length=4, pad_token_id=PAD)
    assert batch["input_ids"].shape == (2, 1, 1, 4) and batch["input_ids"].dtype == np.uint32
    assert batch["labels"].shape == (2, 4) and batch["labels"].dtype == np.uint32
    assert batch["loss_mask"].shape == (2, 1, 4, 1) and batch["loss_mask"].dtype == np.float32
    np.testing.assert_array_equal(batch["input_ids"].reshape(2, 4), [[5, 6, 7, PAD], [8, 9, PAD, PAD]])


def test_position_t_predicts_token_t_plus_one_of_the_response():
    batch = collate_causal_lm([example([5, 6, 7, 8], 2)], max_length=6, pad_token_id=PAD)
    mask = batch["loss_mask"].reshape(6)
    # Targets at positions 1 and 2 (tokens 7 and 8); the prompt, the last token and the padding have none.
    assert np.flatnonzero(mask).tolist() == [1, 2]
    np.testing.assert_array_equal(batch["labels"][0], [PAD, 7, 8, PAD, PAD, PAD])


def test_the_mask_averages_over_real_targets():
    batch = collate_causal_lm([example([5, 6, 7], 0), example([8, 9], 1)], max_length=4, pad_token_id=PAD)
    mask = batch["loss_mask"]
    assert mask.sum() == pytest.approx(mask.size)
    # 2 + 1 real targets: each weighs B * T / 3, so mean(loss * mask) is their mean.
    assert np.unique(mask[mask > 0]) == pytest.approx(8 / 3)


def test_a_batch_without_targets_has_a_zero_mask():
    batch = collate_causal_lm([example([5], 1)], max_length=2, pad_token_id=PAD)
    assert not batch["loss_mask"].any()


def test_an_example_longer_than_max_length_is_an_error(expect_error):
    with expect_error(ValueError, "longer than max_length=2"):
        collate_causal_lm([example([5, 6, 7], 1)], max_length=2, pad_token_id=PAD)
