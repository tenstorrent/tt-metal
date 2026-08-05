# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Blackhole PCC tests for the composed TTNN KDA layer."""

from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.experimental.kimi_delta_attention.reference import kda_forward_reference
from models.experimental.kimi_delta_attention.tests.utils import (
    assert_accurate,
    assert_bit_identical,
    make_config,
    make_program_config,
    random_weights,
)
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention
from models.experimental.kimi_delta_attention.tt.weights import KDAWeights

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]


def _forward(
    layer: KimiDeltaAttention,
    hidden: torch.Tensor,
) -> torch.Tensor:
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=layer.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output = layer.forward(hidden_tt)
    return ttnn.to_torch(output)


def test_composed_layer_pcc(device: ttnn.Device) -> None:
    config = make_config()
    weights = random_weights(config)
    sequence = 32
    hidden = torch.randn(
        1,
        sequence,
        config.hidden_size,
        generator=torch.Generator().manual_seed(41 + sequence),
    ).to(torch.bfloat16)
    golden_output, golden_state = kda_forward_reference(hidden, weights, config)

    layer = KimiDeltaAttention(device, config, weights)
    layer.reset_state(batch_size=1)
    actual_output = _forward(layer, hidden)

    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    actual_recurrent = ttnn.to_torch(layer.recurrent_state)
    actual_convolution = ttnn.to_torch(layer.convolution_state)
    golden_convolution = torch.cat(
        (
            golden_state.q_convolution,
            golden_state.k_convolution,
            golden_state.v_convolution,
        ),
        dim=-1,
    )
    assert_accurate(golden_output, actual_output, name=f"T={sequence} output", pcc_threshold=0.98)
    assert_accurate(golden_state.recurrent, actual_recurrent, name=f"T={sequence} recurrent state", pcc_threshold=0.98)
    assert_accurate(golden_convolution, actual_convolution, name=f"T={sequence} convolution state", pcc_threshold=0.98)


def test_offline_cache_and_cache_only_layer_pcc(device: ttnn.Device, tmp_path: Path, expect_error) -> None:
    config = make_config()
    state_dict = random_weights(config)
    hidden = torch.randn(1, 32, config.hidden_size, generator=torch.Generator().manual_seed(151), dtype=torch.bfloat16)
    golden_output, _ = kda_forward_reference(hidden, state_dict, config)
    cache_prefix = "layer_0.kda"

    assert not KDAWeights.check_cache_complete(tmp_path, cache_prefix, config, device)
    with expect_error(FileNotFoundError, "incomplete KDA TTNN cache"):
        KDAWeights.from_cache(device, config, tmp_path, cache_prefix)

    KDAWeights.build_ttnn_cache(state_dict, tmp_path, cache_prefix, device, config)
    assert KDAWeights.check_cache_complete(tmp_path, cache_prefix, config, device)
    cached_weights = KDAWeights.from_cache(device, config, tmp_path, cache_prefix)
    layer = KimiDeltaAttention(device, config, weights=cached_weights)
    layer.reset_state(batch_size=1)

    actual_output = _forward(layer, hidden)
    assert_accurate(golden_output, actual_output, name="cache-only output", pcc_threshold=0.98)


def test_cache_only_load_rejects_corrupt_tensorbin(device: ttnn.Device, tmp_path: Path, expect_error) -> None:
    config = make_config()
    cache_prefix = "layer_0.kda"
    KDAWeights.build_ttnn_cache(random_weights(config), tmp_path, cache_prefix, device, config)
    next(tmp_path.glob("*.tensorbin")).write_bytes(b"corrupt")

    with expect_error(RuntimeError, "too small"):
        KDAWeights.from_cache(device, config, tmp_path, cache_prefix)


def test_non_tile_aligned_sequence_is_rejected(device: ttnn.Device, expect_error) -> None:
    config = make_config()
    hidden = torch.randn(
        1,
        4,
        config.hidden_size,
        generator=torch.Generator().manual_seed(45),
    ).to(torch.bfloat16)
    layer = KimiDeltaAttention(device, config, random_weights(config))
    layer.reset_state(batch_size=1)

    with expect_error(ValueError, r"requires local T .* divisible by 32, got T=4"):
        _forward(layer, hidden)


def test_batch_greater_than_one_is_rejected_at_state_setup(device: ttnn.Device, expect_error) -> None:
    config = make_config()
    layer = KimiDeltaAttention(device, config, random_weights(config))

    with expect_error(ValueError, "batch_size=1"):
        layer.reset_state(batch_size=2)


def test_segmented_prefill_cache_continuity(device: ttnn.Device) -> None:
    config = make_config()
    weights = random_weights(config)
    hidden = torch.randn(
        1,
        64,
        config.hidden_size,
        generator=torch.Generator().manual_seed(73),
    ).to(torch.bfloat16)
    golden_first, golden_state = kda_forward_reference(hidden[:, :32], weights, config)
    golden_second, golden_state = kda_forward_reference(hidden[:, 32:], weights, config, golden_state)

    layer = KimiDeltaAttention(device, config, weights)
    layer.reset_state(batch_size=1)
    actual_first = _forward(layer, hidden[:, :32])
    actual_second = _forward(layer, hidden[:, 32:])

    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    actual_recurrent = ttnn.to_torch(layer.recurrent_state)
    actual_convolution = ttnn.to_torch(layer.convolution_state)
    golden_convolution = torch.cat(
        (
            golden_state.q_convolution,
            golden_state.k_convolution,
            golden_state.v_convolution,
        ),
        dim=-1,
    )
    assert_accurate(golden_first, actual_first, name="first prefill segment output", pcc_threshold=0.98)
    assert_accurate(golden_second, actual_second, name="second prefill segment output", pcc_threshold=0.98)
    assert_accurate(golden_state.recurrent, actual_recurrent, name="cache recurrent state", pcc_threshold=0.98)
    assert_accurate(golden_convolution, actual_convolution, name="cache convolution state", pcc_threshold=0.98)


@pytest.mark.parametrize("recurrent_state_dtype", [ttnn.float32, ttnn.bfloat16])
def test_external_state_is_updated_in_place(device: ttnn.Device, recurrent_state_dtype: ttnn.DataType) -> None:
    config = make_config()
    weights = random_weights(config)
    hidden = torch.randn(
        1,
        64,
        config.hidden_size,
        generator=torch.Generator().manual_seed(109),
    ).to(torch.bfloat16)
    golden_output, golden_state = kda_forward_reference(hidden, weights, config)

    layer = KimiDeltaAttention(
        device,
        config,
        weights,
        program_config=make_program_config(recurrent_state_dtype=recurrent_state_dtype),
    )
    layer.reset_state(batch_size=1)
    assert layer.recurrent_state is not None
    assert layer.convolution_state is not None
    external_recurrent = layer.recurrent_state
    external_convolution = layer.convolution_state
    recurrent_address = external_recurrent.buffer_address()
    convolution_address = external_convolution.buffer_address()
    layer.set_external_state(external_recurrent, external_convolution)

    actual_output = torch.cat(
        (
            _forward(layer, hidden[:, :32]),
            _forward(layer, hidden[:, 32:]),
        ),
        dim=1,
    )

    assert layer.recurrent_state is external_recurrent
    assert layer.convolution_state is external_convolution
    assert layer.recurrent_state.buffer_address() == recurrent_address
    assert layer.convolution_state.buffer_address() == convolution_address
    assert layer.recurrent_state.dtype == recurrent_state_dtype
    actual_recurrent = ttnn.to_torch(layer.recurrent_state)
    actual_convolution = ttnn.to_torch(layer.convolution_state)
    golden_convolution = torch.cat(
        (
            golden_state.q_convolution,
            golden_state.k_convolution,
            golden_state.v_convolution,
        ),
        dim=-1,
    )
    dtype_name = str(recurrent_state_dtype)
    assert_accurate(golden_output, actual_output, name=f"external {dtype_name} output", pcc_threshold=0.98)
    assert_accurate(
        golden_state.recurrent,
        actual_recurrent,
        name=f"external {dtype_name} recurrent state",
        pcc_threshold=0.98,
    )
    assert_accurate(
        golden_convolution,
        actual_convolution,
        name=f"external {dtype_name} convolution state",
        pcc_threshold=0.98,
    )


def test_composed_layer_determinism(device: ttnn.Device) -> None:
    config = make_config()
    weights = random_weights(config)
    hidden = torch.randn(1, 32, config.hidden_size, generator=torch.Generator().manual_seed(1741)).to(torch.bfloat16)
    layer = KimiDeltaAttention(device, config, weights)
    results = []

    for _ in range(3):
        layer.reset_state(batch_size=1)
        output = _forward(layer, hidden)
        assert layer.recurrent_state is not None
        assert layer.convolution_state is not None
        results.append((output, ttnn.to_torch(layer.recurrent_state), ttnn.to_torch(layer.convolution_state)))

    names = ("output", "recurrent state", "convolution state")
    for iteration, result in enumerate(results[1:], start=1):
        for name, expected, actual in zip(names, results[0], result):
            assert_bit_identical(expected, actual, name=f"layer {name} iteration {iteration}")
