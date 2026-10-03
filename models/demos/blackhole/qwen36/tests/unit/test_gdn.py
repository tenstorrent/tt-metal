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


def test_l2_norm_memory_config_contract(device):
    """L2 normalization inherits placement unless the caller requests an output placement."""
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import l2_norm_ttnn

    torch.manual_seed(0)
    x = torch.randn(1, 32, 1, 32, dtype=torch.float32)
    ref = torch.nn.functional.normalize(x, dim=-1)

    for input_mc in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG):
        x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_mc)
        inherited = l2_norm_ttnn(x_tt, dim=-1)
        assert inherited.memory_config() == input_mc
        assert compute_pcc(ref, ttnn.to_torch(inherited)) > 0.999

    x_l1 = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    in_dram = l2_norm_ttnn(x_l1, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert in_dram.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert compute_pcc(ref, ttnn.to_torch(in_dram)) > 0.999


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
