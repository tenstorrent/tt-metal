# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from models.tt_transformers.tt.model_config import (
    ModelArgs,
    TensorGroup,
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


def test_compute_padded_vocab_size_rejects_invalid_num_devices(expect_error):
    with expect_error(ValueError, "num_devices must be >= 1"):
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


def test_should_pad_sampling_logits_to_power_of_2_rejects_invalid_sampling_splits(expect_error):
    with expect_error(ValueError, "sampling_splits must be >= 1"):
        should_pad_sampling_logits_to_power_of_2("Llama-3.1-70B", 128256, 0)


def _llama_model_args(device_name="P150"):
    args = object.__new__(ModelArgs)
    args.model_name = "Llama-3.1-8B-Instruct"
    args.device_name = device_name
    return args


def test_llama31_8b_p150_uses_projection_specific_dram_reader_counts():
    args = _llama_model_args()

    assert args.get_dram_sharded_matmul_num_workers(TensorGroup.WQKV) == 2
    assert args.get_dram_sharded_matmul_num_workers(TensorGroup.WO) == 2
    assert args.get_dram_sharded_matmul_num_workers(TensorGroup.FF2) == 2
    assert args.get_dram_sharded_matmul_num_workers(TensorGroup.FF1_FF3) == 2


def test_llama31_8b_dram_reader_counts_remain_default_on_unvalidated_devices():
    args = _llama_model_args(device_name="P300")

    for tensor_group in (TensorGroup.WQKV, TensorGroup.WO, TensorGroup.FF1_FF3, TensorGroup.FF2):
        assert args.get_dram_sharded_matmul_num_workers(tensor_group) == 1
