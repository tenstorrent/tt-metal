# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.tt_transformers.tt.model_config import (
    compute_galaxy_padded_vocab_size,
    compute_galaxy_width_shard_cores,
    compute_padded_vocab_size,
    create_galaxy_ff1_out_reduce_scatter_memcfg,
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
    ("vocab_size", "num_devices", "expected"),
    [
        (128256, 32, 128 * 1024),
        (151936, 32, 152576),
        (152064, 32, 152576),
    ],
)
def test_compute_galaxy_padded_vocab_size(vocab_size, num_devices, expected):
    padded_vocab_size = compute_galaxy_padded_vocab_size(vocab_size, num_devices)

    assert padded_vocab_size == expected
    assert padded_vocab_size >= vocab_size
    assert padded_vocab_size % (32 * num_devices) == 0


@pytest.mark.parametrize(
    ("width", "expected"),
    [
        (768, 24),
        (1024, 32),
        (1280, 20),
        (2048, 32),
        (3584, 28),
        (4096, 32),
    ],
)
def test_compute_galaxy_width_shard_cores(width, expected):
    num_cores = compute_galaxy_width_shard_cores(width)

    assert num_cores == expected
    assert (width // num_cores) % 32 == 0


@pytest.mark.parametrize(
    ("hidden_dim", "expected_cores", "expected_shard_width"),
    [
        # Galaxy is 8x4. The reduce-scatter OUTPUT width is
        # hidden_dim // mesh_rows // mesh_cols -- the op computes the shape itself as
        # input_shape[dim] / ring_size, so this config describes the scattered width.
        # Llama-70B: 28672 // 8 // 4 = 896 = 28 * 32, so the 7x4 grid is exact.
        pytest.param(28672, 28, 32, id="llama-70b"),
        # Qwen-72B padded: 32768 // 8 // 4 = 1024, which 28 cores cannot tile-align; 32 can.
        pytest.param(32768, 32, 32, id="qwen-72b-padded"),
        pytest.param(14336, 28, 16, id="llama-8b"),
    ],
)
def test_galaxy_ff1_out_reduce_scatter_memory_config(hidden_dim, expected_cores, expected_shard_width):
    memory_config = create_galaxy_ff1_out_reduce_scatter_memcfg(hidden_dim, mesh_rows=8, mesh_cols=4)

    assert memory_config.shard_spec.grid.num_cores() == expected_cores
    assert memory_config.shard_spec.shape == [32, expected_shard_width]


@pytest.mark.parametrize("hidden_dim", [28672, 32768])
def test_galaxy_ff1_shard_grid_exactly_covers_the_scattered_width(hidden_dim):
    """The shard grid must cover the reduce-scatter output width exactly, no more."""
    memory_config = create_galaxy_ff1_out_reduce_scatter_memcfg(hidden_dim, mesh_rows=8, mesh_cols=4)

    scattered_width = hidden_dim // 8 // 4
    covered = memory_config.shard_spec.grid.num_cores() * memory_config.shard_spec.shape[1]
    assert covered == scattered_width
    assert memory_config.shard_spec.shape[1] % 32 == 0


def test_galaxy_ff1_matches_the_validated_demo_ratio():
    """Cross-check against models/demos/llama3_70b_galaxy: 3840 in, 960 out, ratio 4."""
    memory_config = create_galaxy_ff1_out_reduce_scatter_memcfg(28672, mesh_rows=8, mesh_cols=4)

    covered = memory_config.shard_spec.grid.num_cores() * memory_config.shard_spec.shape[1]
    per_device_width = 28672 // 8
    assert covered == per_device_width // 4 == 896
    assert covered <= 960  # the demo's 30-core layout pads 896 up to 960


def test_compute_galaxy_width_shard_cores_rejects_non_positive_width(expect_error):
    with expect_error(ValueError, "positive multiple"):
        compute_galaxy_width_shard_cores(0)


@pytest.mark.parametrize(
    ("base_model_name", "padded_vocab_size", "sampling_splits", "expected"),
    [
        ("Llama-3.1-70B", 128256, 4, True),
        ("Llama-3.1-70B", 131072, 4, False),
        ("Llama-3.1-8B", 128256, 4, True),
    ],
)
def test_should_pad_sampling_logits_to_power_of_2(base_model_name, padded_vocab_size, sampling_splits, expected):
    assert should_pad_sampling_logits_to_power_of_2(base_model_name, padded_vocab_size, sampling_splits) is expected


def test_should_pad_sampling_logits_to_power_of_2_rejects_invalid_sampling_splits(expect_error):
    with expect_error(ValueError, "sampling_splits must be >= 1"):
        should_pad_sampling_logits_to_power_of_2("Llama-3.1-70B", 128256, 0)
