# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.tt_transformers.tt.model_config import (
    compute_padded_vocab_size,
    should_pad_sampling_logits_to_power_of_2,
)


@pytest.mark.parametrize(
    ("vocab_size", "num_devices", "expected"),
    [
        (151936, 1, 151936),
        (151936, 4, 151936),
        (151936, 8, 152064),
        (151936, 32, 152576),
        (32001, 2, 32064),
    ],
)
def test_compute_padded_vocab_size(vocab_size, num_devices, expected):
    padded_vocab_size = compute_padded_vocab_size(vocab_size, num_devices)

    assert padded_vocab_size == expected
    assert padded_vocab_size >= vocab_size
    assert padded_vocab_size % (32 * num_devices) == 0
    assert (padded_vocab_size // num_devices) % 32 == 0


def test_compute_padded_vocab_size_rejects_invalid_num_devices():
    with pytest.raises(ValueError, match="num_devices must be >= 1"):
        compute_padded_vocab_size(32000, 0)


@pytest.mark.parametrize(
    ("base_model_name", "padded_vocab_size", "sampling_splits", "expected"),
    [
        ("Llama-3.1-70B", 128256, 4, True),
        ("Llama-3.1-70B", 131072, 4, False),
        ("Llama-3.1-8B", 128256, 4, False),
    ],
)
def test_should_pad_sampling_logits_to_power_of_2(base_model_name, padded_vocab_size, sampling_splits, expected):
    assert should_pad_sampling_logits_to_power_of_2(base_model_name, padded_vocab_size, sampling_splits) is expected


def test_should_pad_sampling_logits_to_power_of_2_rejects_invalid_sampling_splits():
    with pytest.raises(ValueError, match="sampling_splits must be >= 1"):
        should_pad_sampling_logits_to_power_of_2("Llama-3.1-70B", 128256, 0)
