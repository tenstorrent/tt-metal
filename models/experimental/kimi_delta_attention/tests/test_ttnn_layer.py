# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Blackhole PCC tests for the composed TTNN KDA layer."""

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.reference import kda_forward_reference
from models.experimental.kimi_delta_attention.tests.test_factory import make_config, random_weights
from models.experimental.kimi_delta_attention.tt.layer import KimiDeltaAttention

pytestmark = [
    run_for_blackhole(),
    pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True),
]


def _assert_pcc(name: str, golden: torch.Tensor, actual: torch.Tensor, threshold: float = 0.98) -> float:
    passed, pcc = comp_pcc(golden, actual, pcc=threshold)
    max_abs = (golden.float() - actual.float()).abs().max().item()
    print(f"{name}: PCC={pcc:.6f}, max_abs={max_abs:.6e}")
    assert passed, f"{name} PCC {pcc:.6f} < {threshold}"
    return pcc


def _forward(
    layer: KimiDeltaAttention,
    hidden: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=layer.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output = layer.forward(hidden_tt, mode=mode)
    return ttnn.to_torch(output)


@pytest.mark.parametrize("sequence", [1, 4, 32])
def test_composed_layer_pcc(device: ttnn.Device, sequence: int) -> None:
    config = make_config()
    weights = random_weights(config)
    hidden = torch.randn(
        1,
        sequence,
        config.hidden_size,
        generator=torch.Generator().manual_seed(41 + sequence),
    ).to(torch.bfloat16)
    golden_output, golden_state = kda_forward_reference(hidden, weights, config)

    layer = KimiDeltaAttention(device, config, weights)
    layer.reset_state(batch_size=1)
    actual_output = _forward(layer, hidden, "recurrent" if sequence == 1 else "chunk")

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
    _assert_pcc(f"T={sequence} output", golden_output, actual_output)
    _assert_pcc(f"T={sequence} recurrent state", golden_state.recurrent, actual_recurrent)
    _assert_pcc(f"T={sequence} convolution state", golden_convolution, actual_convolution)


def test_target_shape_decode_pcc(device: ttnn.Device) -> None:
    config = KDAConfig(
        hidden_size=2304,
        num_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        norm_eps=1e-5,
    )
    weights = random_weights(config)
    hidden = torch.randn(
        1,
        1,
        config.hidden_size,
        generator=torch.Generator().manual_seed(101),
    ).to(torch.bfloat16)
    golden_output, golden_state = kda_forward_reference(hidden, weights, config)

    layer = KimiDeltaAttention(device, config, weights)
    layer.reset_state(batch_size=1)
    actual_output = _forward(layer, hidden, "recurrent")

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
    _assert_pcc("target decode output", golden_output, actual_output)
    _assert_pcc("target decode recurrent state", golden_state.recurrent, actual_recurrent)
    _assert_pcc("target decode convolution state", golden_convolution, actual_convolution)


def test_prefill_decode_cache_continuity(device: ttnn.Device) -> None:
    config = make_config()
    weights = random_weights(config)
    hidden = torch.randn(
        1,
        5,
        config.hidden_size,
        generator=torch.Generator().manual_seed(73),
    ).to(torch.bfloat16)
    golden_prefill, golden_state = kda_forward_reference(hidden[:, :4], weights, config)
    golden_decode, golden_state = kda_forward_reference(hidden[:, 4:], weights, config, golden_state)

    layer = KimiDeltaAttention(device, config, weights)
    layer.reset_state(batch_size=1)
    actual_prefill = _forward(layer, hidden[:, :4], "chunk")
    actual_decode = _forward(layer, hidden[:, 4:], "recurrent")

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
    _assert_pcc("cache prefill output", golden_prefill, actual_prefill)
    _assert_pcc("cache decode output", golden_decode, actual_decode)
    _assert_pcc("cache recurrent state", golden_state.recurrent, actual_recurrent)
    _assert_pcc("cache convolution state", golden_convolution, actual_convolution)


@pytest.mark.parametrize("recurrent_state_dtype", [ttnn.float32, ttnn.bfloat16])
def test_external_state_is_updated_in_place(device: ttnn.Device, recurrent_state_dtype: ttnn.DataType) -> None:
    config = make_config(recurrent_state_dtype=recurrent_state_dtype)
    weights = random_weights(config)
    hidden = torch.randn(
        1,
        2,
        config.hidden_size,
        generator=torch.Generator().manual_seed(109),
    ).to(torch.bfloat16)
    golden_output, golden_state = kda_forward_reference(hidden, weights, config)

    layer = KimiDeltaAttention(device, config, weights)
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
            _forward(layer, hidden[:, :1], "recurrent"),
            _forward(layer, hidden[:, 1:], "recurrent"),
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
    _assert_pcc(f"external {dtype_name} output", golden_output, actual_output)
    _assert_pcc(f"external {dtype_name} recurrent state", golden_state.recurrent, actual_recurrent)
    _assert_pcc(f"external {dtype_name} convolution state", golden_convolution, actual_convolution)
