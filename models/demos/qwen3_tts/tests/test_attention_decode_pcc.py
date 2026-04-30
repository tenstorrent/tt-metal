# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Decode-mode and prefill-ISL=128 PCC test for Attention.

    python -m pytest -q models/demos/qwen3_tts/tests/test_attention_decode_pcc.py
"""
import pytest
import torch

import ttnn
from models.demos.qwen3_tts.reference.functional import attention as torch_attention
from models.demos.qwen3_tts.reference.functional import get_default_talker_config
from models.demos.qwen3_tts.tt.attention import Attention
from models.tt_transformers.tt.common import get_rot_transformation_mat


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    a = x.flatten().float()
    b = y.flatten().float()
    a_c = a - a.mean()
    b_c = b - b.mean()
    return ((a_c * b_c).sum() / (a_c.norm() * b_c.norm() + 1e-12)).item()


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_device(device_id=0)
    yield d
    ttnn.close_device(d)


@pytest.mark.parametrize("seq_len,mode", [(1, "decode"), (128, "prefill")])
def test_attention_pcc_modes(device, seq_len, mode):
    torch.manual_seed(42)
    cfg = get_default_talker_config()
    H = cfg.hidden_size
    NH = cfg.num_attention_heads
    NKV = cfg.num_key_value_heads
    HD = cfg.head_dim

    x_torch = torch.randn(1, seq_len, H, dtype=torch.bfloat16)
    q_w = torch.randn(NH * HD, H, dtype=torch.bfloat16)
    k_w = torch.randn(NKV * HD, H, dtype=torch.bfloat16)
    v_w = torch.randn(NKV * HD, H, dtype=torch.bfloat16)
    o_w = torch.randn(H, NH * HD, dtype=torch.bfloat16)
    qn_w = torch.ones(HD, dtype=torch.bfloat16)
    kn_w = torch.ones(HD, dtype=torch.bfloat16)

    cos_id = torch.ones(1, seq_len, HD, dtype=torch.bfloat16)
    sin_id = torch.zeros(1, seq_len, HD, dtype=torch.bfloat16)

    ref = torch_attention(
        x_torch,
        q_w,
        k_w,
        v_w,
        o_w,
        qn_w,
        kn_w,
        cos_id,
        sin_id,
        num_heads=NH,
        num_kv_heads=NKV,
        head_dim=HD,
        rms_norm_eps=cfg.rms_norm_eps,
        use_mrope=False,
    )

    sd = {
        "test_layer.self_attn.q_proj.weight": q_w,
        "test_layer.self_attn.k_proj.weight": k_w,
        "test_layer.self_attn.v_proj.weight": v_w,
        "test_layer.self_attn.o_proj.weight": o_w,
        "test_layer.self_attn.q_norm.weight": qn_w,
        "test_layer.self_attn.k_norm.weight": kn_w,
    }
    attn = Attention(
        device=device,
        hidden_size=H,
        num_heads=NH,
        num_kv_heads=NKV,
        head_dim=HD,
        state_dict=sd,
        layer_prefix="test_layer",
        rms_norm_eps=cfg.rms_norm_eps,
    )

    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(1),  # [B, 1, S, H]
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos_tt = ttnn.from_torch(
        cos_id.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sin_tt = ttnn.from_torch(
        sin_id.unsqueeze(1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    trans = get_rot_transformation_mat(dhead=HD)
    trans_tt = ttnn.from_torch(
        trans,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    y_tt, _ = attn(x_tt, cos_tt, sin_tt, trans_tt, mode=mode)
    y = ttnn.to_torch(y_tt).squeeze(1)
    pcc = pearson(ref, y)
    print(f"[seq_len={seq_len}, mode={mode}] Attention PCC = {pcc:.6f}")
    assert pcc > 0.99
