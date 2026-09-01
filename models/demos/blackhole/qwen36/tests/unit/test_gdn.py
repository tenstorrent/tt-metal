# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Component PCC: single-device Gated DeltaNet (layer 0) vs torch reference.

``device`` and ``setup`` come from tests/unit/conftest.py.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc, get_pcc_threshold

from .conftest import DEVICE_PARAMS

pytestmark = [run_for_blackhole(), pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)]


def test_deltanet_pcc(device, setup, request):
    """Compare TTNN deltanet against the torch reference for layer 0."""
    args, sd, raw = setup
    from models.demos.blackhole.qwen36.tt.gdn import GDNConfig, Qwen36GatedDeltaNet
    from models.demos.blackhole.qwen36.utils.substate import substate
    from models.experimental.gated_attention_gated_deltanet.torch_functional.gated_deltanet import (
        gated_deltanet_forward,
    )

    layer_num = 0
    B, T = 1, 4

    prefix = f"layers.{layer_num}.linear_attn"
    x = torch.randn(B, T, 4096, dtype=torch.bfloat16)

    # Torch reference (note: torch uses [out, in] convention, no transpose).
    # Cast all weights to float32 to avoid dtype mismatch in the torch reference.
    def to_f32(t):
        return t.float() if t is not None else None

    # The split q/k/v_proj keys were removed; derive them by slicing the combined
    # qkv_proj.weight [8192, 4096] = [q(2048)+k(2048)+v(4096), in]. These slices are
    # byte-identical to the old split keys, so the PCC is unchanged.
    qkv_w = sd[f"{prefix}.qkv_proj.weight"]  # [8192, 4096] = [q(2048)+k(2048)+v(4096), in]
    ref_out, _ = gated_deltanet_forward(
        hidden_states=x.float(),
        q_proj_weight=to_f32(qkv_w[:2048, :]),
        k_proj_weight=to_f32(qkv_w[2048:4096, :]),
        v_proj_weight=to_f32(qkv_w[4096:, :]),
        a_proj_weight=to_f32(sd[f"{prefix}.in_proj_a.weight"]),
        b_proj_weight=to_f32(sd[f"{prefix}.in_proj_b.weight"]),
        o_proj_weight=to_f32(sd[f"{prefix}.out_proj.weight"]),
        q_conv_weight=to_f32(sd[f"{prefix}.q_conv.weight"]),
        k_conv_weight=to_f32(sd[f"{prefix}.k_conv.weight"]),
        v_conv_weight=to_f32(sd[f"{prefix}.v_conv.weight"]),
        q_conv_bias=to_f32(sd.get(f"{prefix}.q_conv.bias")),
        k_conv_bias=to_f32(sd.get(f"{prefix}.k_conv.bias")),
        v_conv_bias=to_f32(sd.get(f"{prefix}.v_conv.bias")),
        A_log=to_f32(sd[f"{prefix}.A_log"]),
        dt_bias=to_f32(sd[f"{prefix}.dt_bias"]),
        o_norm_weight=to_f32(sd[f"{prefix}.norm.weight"]),
        g_proj_weight=to_f32(sd[f"{prefix}.in_proj_z.weight"]),
        num_heads=16,
        num_v_heads=32,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        use_gate=True,
        norm_eps=1e-6,
        mode="fused_recurrent",
        recurrent_state=None,
        output_final_state=True,
    )

    # TTNN
    deltanet = Qwen36GatedDeltaNet(device, GDNConfig.from_args(args), substate(sd, f"layers.{layer_num}.linear_attn"))
    deltanet.reset_state(B)
    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(deltanet.forward(x_t, mode="recurrent"))

    pcc = compute_pcc(ref_out, out)
    logger.info(f"DeltaNet PCC: {pcc:.6f}")
    logger.info(
        f"Ref range: [{ref_out.min():.4f}, {ref_out.max():.4f}]  TTNN range: [{out.min():.4f}, {out.max():.4f}]"
    )
    assert pcc > get_pcc_threshold(request), f"DeltaNet PCC too low: {pcc}"


def test_qwen_gdn_reuses_native_conv_weights_after_state_reset(device):
    from models.demos.blackhole.qwen36.tt.gdn import GDNConfig, Qwen36GatedDeltaNet

    torch.manual_seed(0)
    config = GDNConfig(
        num_heads=1,
        num_v_heads=1,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        norm_eps=1e-6,
        q_dim=32,
        k_dim=32,
        v_dim=32,
    )

    def random_bfloat16(*shape):
        return torch.randn(shape, dtype=torch.bfloat16)

    state_dict = {
        "qkv_proj.weight": random_bfloat16(96, 32),
        "in_proj_a.weight": random_bfloat16(1, 32),
        "in_proj_b.weight": random_bfloat16(1, 32),
        "in_proj_z.weight": random_bfloat16(32, 32),
        "out_proj.weight": random_bfloat16(32, 32),
        "q_conv.weight": random_bfloat16(32, 1, 4),
        "k_conv.weight": random_bfloat16(32, 1, 4),
        "v_conv.weight": random_bfloat16(32, 1, 4),
        "A_log": torch.zeros(1, dtype=torch.bfloat16),
        "dt_bias": torch.zeros(1, dtype=torch.bfloat16),
        "norm.weight": torch.ones(32, dtype=torch.bfloat16),
    }
    deltanet = Qwen36GatedDeltaNet(device, config, state_dict)
    torch_input = random_bfloat16(1, 1, 32)

    def run_from_reset_state():
        deltanet.reset_state(1)
        input_tt = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        return ttnn.to_torch(deltanet.forward(input_tt, mode="recurrent"))

    first_output = run_from_reset_state()
    caches = (
        deltanet.weights.q_native_weight_cache,
        deltanet.weights.k_native_weight_cache,
        deltanet.weights.v_native_weight_cache,
    )
    assert [len(cache) for cache in caches] == [1, 1, 1]
    prepared_weight_addresses = [next(iter(cache.values())).buffer_address() for cache in caches]

    second_output = run_from_reset_state()
    assert [next(iter(cache.values())).buffer_address() for cache in caches] == prepared_weight_addresses
    assert list(first_output.shape) == [1, 1, 32]
    assert torch.isfinite(first_output).all()
    assert torch.isfinite(second_output).all()
    assert compute_pcc(first_output, second_output) > 0.999


def test_qwen35_9b_causal_conv1d_helper_uses_native_conv(device):
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import causal_conv1d_ttnn

    batch_size, sequence_length, channels, kernel_size = 1, 4, 8192, 4
    torch.manual_seed(0)
    torch_input = torch.randn(batch_size, channels, sequence_length, dtype=torch.bfloat16).float()
    torch_weight = torch.randn(channels, 1, kernel_size, dtype=torch.bfloat16).float()
    golden = torch.nn.functional.silu(
        torch.nn.functional.conv1d(
            torch.nn.functional.pad(torch_input, (kernel_size - 1, 0)),
            torch_weight,
            groups=channels,
        )
    )

    input_tt = ttnn.from_torch(
        torch_input.permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    weight_tt = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    native_weight_cache = {}
    output_tt, state_tt = causal_conv1d_ttnn(
        input_tt,
        weight_tt,
        None,
        kernel_size,
        device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        native_weight_cache=native_weight_cache,
    )
    output = ttnn.to_torch(output_tt).permute(0, 2, 1)
    state = ttnn.to_torch(state_tt).permute(0, 2, 1)

    output_pcc = compute_pcc(golden, output)
    state_pcc = compute_pcc(torch_input[:, :, -(kernel_size - 1) :], state)
    assert output_pcc > 0.995
    assert state_pcc > 0.999
    assert len(native_weight_cache) == 1
    first_prepared_weight = next(iter(native_weight_cache.values()))
    first_prepared_weight_address = first_prepared_weight.buffer_address()

    output_tt, _ = causal_conv1d_ttnn(
        input_tt,
        weight_tt,
        None,
        kernel_size,
        device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        native_weight_cache=native_weight_cache,
    )
    output = ttnn.to_torch(output_tt).permute(0, 2, 1)
    assert compute_pcc(golden, output) > 0.995
    assert next(iter(native_weight_cache.values())).buffer_address() == first_prepared_weight_address

    next_sequence_length = sequence_length + 1
    next_torch_input = torch.randn(batch_size, channels, next_sequence_length, dtype=torch.bfloat16).float()
    next_golden = torch.nn.functional.silu(
        torch.nn.functional.conv1d(
            torch.nn.functional.pad(next_torch_input, (kernel_size - 1, 0)),
            torch_weight,
            groups=channels,
        )
    )
    next_input_tt = ttnn.from_torch(
        next_torch_input.permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output_tt, _ = causal_conv1d_ttnn(
        next_input_tt,
        weight_tt,
        None,
        kernel_size,
        device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        native_weight_cache=native_weight_cache,
    )
    output = ttnn.to_torch(output_tt).permute(0, 2, 1)
    assert compute_pcc(next_golden, output) > 0.995
    assert len(native_weight_cache) == 1
    assert not first_prepared_weight.is_allocated()

    replacement_torch_weight = torch.randn(channels, 1, kernel_size, dtype=torch.bfloat16).float()
    replacement_weight_tt = ttnn.from_torch(
        replacement_torch_weight,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    replacement_golden = torch.nn.functional.silu(
        torch.nn.functional.conv1d(
            torch.nn.functional.pad(next_torch_input, (kernel_size - 1, 0)),
            replacement_torch_weight,
            groups=channels,
        )
    )
    previous_prepared_weight = next(iter(native_weight_cache.values()))
    output_tt, _ = causal_conv1d_ttnn(
        next_input_tt,
        replacement_weight_tt,
        None,
        kernel_size,
        device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        native_weight_cache=native_weight_cache,
    )
    output = ttnn.to_torch(output_tt).permute(0, 2, 1)
    assert compute_pcc(replacement_golden, output) > 0.995
    assert len(native_weight_cache) == 1
    assert not previous_prepared_weight.is_allocated()


@pytest.mark.parametrize("bias_location", ["host", "device"])
def test_causal_conv1d_helper_applies_bias(device, bias_location):
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import causal_conv1d_ttnn

    batch_size, sequence_length, channels, kernel_size = 1, 4, 128, 4
    torch.manual_seed(0)
    torch_input = torch.randn(batch_size, channels, sequence_length, dtype=torch.bfloat16).float()
    torch_weight = torch.randn(channels, 1, kernel_size, dtype=torch.bfloat16).float()
    torch_bias = torch.randn(channels, dtype=torch.bfloat16).float()
    golden = torch.nn.functional.silu(
        torch.nn.functional.conv1d(
            torch.nn.functional.pad(torch_input, (kernel_size - 1, 0)),
            torch_weight,
            bias=torch_bias,
            groups=channels,
        )
    )

    input_tt = ttnn.from_torch(
        torch_input.permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    weight_tt = ttnn.from_torch(torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias_tt = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias_dev = (
        ttnn.from_torch(
            torch_bias.reshape(1, 1, channels),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        if bias_location == "device"
        else None
    )
    native_weight_cache = {}
    output_tt, _ = causal_conv1d_ttnn(
        input_tt,
        weight_tt,
        bias_tt if bias_location == "host" else None,
        kernel_size,
        device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        bias_dev=bias_dev,
        native_weight_cache=native_weight_cache,
    )
    output = ttnn.to_torch(output_tt).permute(0, 2, 1)
    assert compute_pcc(golden, output) > 0.995
    assert native_weight_cache == {}
