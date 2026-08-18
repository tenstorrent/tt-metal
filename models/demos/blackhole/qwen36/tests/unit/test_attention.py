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
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc, get_pcc_threshold

from .conftest import DEVICE_PARAMS

pytestmark = [
    run_for_wormhole_b0_or_blackhole(),
    pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True),
]


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


def test_gated_attention_decode_pcc(device, setup, request):
    """Compare TTNN paged decode (the production decode branch) against torch for layer 3.

    Fills a paged KV cache via paged prefill for a short prompt, then runs one paged
    decode step and checks it against a torch reference computed over the full
    prompt+decode sequence in one shot (equivalent to incremental KV-cache attention).
    """
    from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup, compute_rope_freqs
    from models.experimental.gated_attention_gated_deltanet.torch_functional.gated_attention import (
        gated_attention_forward,
    )

    args, sd, raw = setup
    layer_num = 3
    B, T_prefill = 1, 4
    T = T_prefill + 1  # prefix + one decode token

    prefix = f"layers.{layer_num}.self_attn"
    q_w = sd[f"{prefix}.q_proj.weight"]
    k_w = sd[f"{prefix}.k_proj.weight"]
    v_w = sd[f"{prefix}.v_proj.weight"]
    o_w = sd[f"{prefix}.o_proj.weight"]
    q_norm = sd[f"{prefix}.q_norm.weight"]
    k_norm = sd[f"{prefix}.k_norm.weight"]

    torch.manual_seed(0)
    x = torch.randn(B, T, 4096, dtype=torch.bfloat16)

    # Torch reference: full-sequence causal attention: last row == the decode step's output.
    cos_cpu, sin_cpu = compute_rope_freqs(64, 2048, theta=10_000_000)
    pos_ids = torch.arange(T)
    cos_t = cos_cpu[pos_ids].unsqueeze(0).to(torch.bfloat16)
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
    ref_decode = ref_out[:, -1, :]

    # TTNN: paged prefill fills the KV cache, then one paged decode step.
    from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention
    from models.demos.blackhole.qwen36.utils.substate import substate

    attn_state = substate(sd, f"layers.{layer_num}.self_attn")
    attn = Qwen36GatedAttention(device, AttentionConfig.from_args(args), attn_state)
    rope = Qwen36RoPESetup(device, args)

    BLOCK_SIZE = 64
    num_blocks = 4  # 256 tokens of cache, plenty for T_prefill=4

    def mk_cache():
        return ttnn.from_torch(
            torch.zeros(num_blocks, args.n_kv_heads, BLOCK_SIZE, args.head_dim, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

    attn.set_paged_kv_cache(mk_cache(), mk_cache())
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    page_table_tt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    x_prefill_t = ttnn.from_torch(x[:, :T_prefill, :], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cos_pf, sin_pf = rope.get_rot_mats(torch.arange(T_prefill).unsqueeze(0))
    attn.forward(
        x_prefill_t,
        cos_pf,
        sin_pf,
        page_table=page_table_tt,
        chunk_page_table=page_table_tt,
        chunk_start_idx=0,
    )

    x_decode_t = ttnn.from_torch(x[:, T_prefill:, :], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cos_d, sin_d = rope.get_rot_mats(torch.tensor([[T_prefill]]))
    position_tensor = ttnn.from_torch(torch.tensor([T_prefill], dtype=torch.int32), dtype=ttnn.int32, device=device)
    out = attn.forward(
        x_decode_t,
        cos_d,
        sin_d,
        position_tensor=position_tensor,
        page_table=page_table_tt,
    )
    out_torch = ttnn.to_torch(out).reshape(B, -1)

    pcc = compute_pcc(ref_decode, out_torch)
    logger.info(f"Gated Attention decode PCC: {pcc:.6f}")
    logger.info(
        f"Ref range: [{ref_decode.min():.4f}, {ref_decode.max():.4f}]  "
        f"TTNN range: [{out_torch.min():.4f}, {out_torch.max():.4f}]"
    )
    assert pcc > get_pcc_threshold(request), f"Gated Attention decode PCC too low: {pcc}"
