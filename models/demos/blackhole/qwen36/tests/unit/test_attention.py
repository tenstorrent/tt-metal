# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Component PCC: single-device gated attention (layer 3) vs torch reference.

``device`` and ``setup`` come from tests/unit/conftest.py. (The TP loader's
q_norm/k_norm +1 regression lives with the TP attention tests in
``tests/test_attention_tp.py::test_attention_tp_qknorm_offset``.)
"""
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc, get_pcc_threshold

from .conftest import DEVICE_PARAMS

pytestmark = [run_for_blackhole(), pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)]


def test_gated_attention_pcc(device, setup, request):
    """Compare TTNN gated attention against the torch reference for layer 3."""
    args, sd, raw = setup
    from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup, compute_rope_freqs
    from models.experimental.gated_attention_gated_deltanet.torch_functional.gated_attention import (
        gated_attention_forward,
    )

    layer_num = 3
    B, T = 1, 4

    # Torch reference (uses HF convention: weight is [out, in])
    prefix = f"layers.{layer_num}.self_attn"
    q_w = sd[f"{prefix}.q_proj.weight"]
    k_w = sd[f"{prefix}.k_proj.weight"]
    v_w = sd[f"{prefix}.v_proj.weight"]
    o_w = sd[f"{prefix}.o_proj.weight"]
    q_norm = sd[f"{prefix}.q_norm.weight"]
    k_norm = sd[f"{prefix}.k_norm.weight"]

    x = torch.randn(B, T, 4096, dtype=torch.bfloat16)

    # RoPE for torch (cast to bfloat16 to match input dtype)
    cos_cpu, sin_cpu = compute_rope_freqs(64, 2048, theta=10_000_000)
    pos_ids = torch.arange(T)
    cos_t = cos_cpu[pos_ids].unsqueeze(0).to(torch.bfloat16)  # [1, T, 64]
    sin_t = sin_cpu[pos_ids].unsqueeze(0).to(torch.bfloat16)

    ref_out, _, _ = gated_attention_forward(
        hidden_states=x,
        q_proj_weight=q_w,
        k_proj_weight=k_w,
        v_proj_weight=v_w,
        o_proj_weight=o_w,
        q_norm_weight=q_norm,
        k_norm_weight=k_norm,
        cos=cos_t,
        sin=sin_t,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        norm_eps=1e-6,
    )

    # TTNN
    from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention
    from models.demos.blackhole.qwen36.utils.substate import substate

    attn_state = substate(sd, f"layers.{layer_num}.self_attn")
    attn = Qwen36GatedAttention(device, AttentionConfig.from_args(args), attn_state)
    rope = Qwen36RoPESetup(device, args)
    pos = torch.arange(T).unsqueeze(0)
    cos_ttnn, sin_ttnn = rope.get_rot_mats(pos)

    x_t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(attn.forward(x_t, cos_ttnn, sin_ttnn))

    pcc = compute_pcc(ref_out, out)
    logger.info(f"Gated Attention PCC: {pcc:.6f}")
    logger.info(
        f"Ref range: [{ref_out.min():.4f}, {ref_out.max():.4f}]  TTNN range: [{out.min():.4f}, {out.max():.4f}]"
    )
    assert pcc > get_pcc_threshold(request), f"Gated Attention PCC too low: {pcc}"
