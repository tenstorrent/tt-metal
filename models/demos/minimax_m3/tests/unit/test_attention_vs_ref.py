# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-1 PCC test for the full MiniMax-M3 GQA attention block vs a hand-written torch reference.

Block: QKV proj -> head split -> per-head gemma QK-norm -> partial RoPE -> GQA causal SDPA ->
o_proj. M3 dims: 64 q heads / 4 kv heads / head_dim 128 / rotary_dim 64 / theta 5e6, per-head
QK-norm with gemma (1+w), use_gemma_norm=true. Anchor: transformers minimax_m3_vl attention.

Self-authored torch reference + identical random weights (no HF/checkpoint). The TT module is fed
the Meta-RoPE-swizzled weights via the production convert_hf_qkv_to_meta_format_partial helper
(which also swizzles the q/k-norm gains), so the TT swizzled-space path matches the HF-convention
torch reference. Single card (TP=1; the TP>1/CCL path needs a multi-card system).
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.attention import Attention, AttentionConfig
from models.demos.minimax_m3.tt.attention_configs import MiniMaxM3AttentionProgramConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.model import create_rope_setup
from models.demos.minimax_m3.utils.general_utils import get_default_num_links
from models.demos.minimax_m3.utils.weight_conversion import convert_hf_qkv_to_meta_format_partial

from ..test_factory import parametrize_mesh_with_fabric

# M3 attention dims (text_config).
HIDDEN, NQ, NKV, HEAD_DIM, ROTARY_DIM, THETA, EPS = 6144, 64, 4, 128, 64, 5_000_000.0, 1e-6


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _partial_rope(t, cos, sin):
    """Rotate the first ROTARY_DIM dims (HF rotate_half convention), pass the rest through."""
    t_rot, t_pass = t[..., :ROTARY_DIM], t[..., ROTARY_DIM:]
    t_rot = t_rot * cos + _rotate_half(t_rot) * sin
    return torch.cat([t_rot, t_pass], dim=-1)


def _gemma_per_head_norm(x, weight):
    """RMSNorm over head_dim (last dim) with gemma (1+w)."""
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + EPS)) * (1.0 + weight.float())


def _torch_attention(x, w, cos, sin):
    """HF-convention M3 GQA attention reference (fp32). w: dict of [out,in] proj weights + [128] norms."""
    B, S, _ = x.shape
    q = (x @ w["q"].t()).view(B, S, NQ, HEAD_DIM).transpose(1, 2)  # [B, NQ, S, HD]
    k = (x @ w["k"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
    v = (x @ w["v"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2)

    q = _gemma_per_head_norm(q, w["q_norm"])
    k = _gemma_per_head_norm(k, w["k_norm"])

    q = _partial_rope(q, cos, sin)
    k = _partial_rope(k, cos, sin)

    rep = NQ // NKV
    k = k.repeat_interleave(rep, dim=1)
    v = v.repeat_interleave(rep, dim=1)

    scale = HEAD_DIM**-0.5
    scores = (q @ k.transpose(-1, -2)) * scale
    causal = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    attn = torch.softmax(scores + causal, dim=-1)
    out = attn @ v  # [B, NQ, S, HD]
    out = out.transpose(1, 2).reshape(B, S, NQ * HEAD_DIM)
    return out @ w["o"].t()


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (8, 4)], linear_fabric=True)
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_attention_prefill_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Full M3 GQA attention block vs self-authored torch reference, random weights.

    (1,1) = TP=1. (8,4) = TP=4 (M3's KV-head-limited tensor-parallel), exercising the o_proj
    reduce-scatter/all-gather CCL; attention output is full-hidden replicated post-allreduce, so
    device[0] holds the full result. Needs TT_MESH_GRAPH_DESC_PATH=single_bh_galaxy ([8,4])."""
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, HIDDEN) * 0.1

    # Random projection + per-head norm weights ([out, in] HF layout; norms are [head_dim]).
    w = {
        "q": torch.randn(NQ * HEAD_DIM, HIDDEN) * 0.02,
        "k": torch.randn(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "v": torch.randn(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "o": torch.randn(HIDDEN, NQ * HEAD_DIM) * 0.02,
        "q_norm": torch.randn(HEAD_DIM) * 0.1,
        "k_norm": torch.randn(HEAD_DIM) * 0.1,
    }

    # HF-convention partial RoPE cos/sin (theta 5e6, rotary_dim 64), shape [1,1,S,rotary_dim].
    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    pos = torch.arange(seq_len).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_ref, sin_ref = emb.cos()[None, None], emb.sin()[None, None]

    ref_out = _torch_attention(x.float(), w, cos_ref, sin_ref)

    # --- TT module from the SAME weights, Meta-RoPE-swizzled (q/k proj + q/k norm gains) ---
    hf_state = {
        "q_proj.weight": w["q"],
        "k_proj.weight": w["k"],
        "v_proj.weight": w["v"],
        "o_proj.weight": w["o"],
        "q_norm.weight": w["q_norm"],
        "k_norm.weight": w["k_norm"],
    }
    state = convert_hf_qkv_to_meta_format_partial(hf_state, HEAD_DIM, ROTARY_DIM)

    hf_config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=NQ,
        num_key_value_heads=NKV,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        rope_theta=THETA,
        rope_scaling=None,
        rms_norm_eps=EPS,
        max_position_embeddings=max(seq_len, 128),
        use_qk_norm=True,
        use_gemma_norm=True,
    )

    mesh_config = MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    rope_setup = create_rope_setup(mesh_device=mesh_device, hf_config=hf_config, datatype=ttnn.bfloat16)
    trans_mats = rope_setup.get_both_trans_mats()

    attn_config = AttentionConfig(
        hidden_size=HIDDEN,
        num_heads=NQ,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        rms_norm_eps=EPS,
        use_qk_norm=True,
        use_gemma_norm=True,
        max_seq_len=max(seq_len, 128),
        max_local_batch_size=1,
    )
    attn = Attention(
        mesh_device=mesh_device,
        config=attn_config,
        state_dict=state,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=MiniMaxM3AttentionProgramConfig(),
        layer_idx=0,
        transformation_mats=trans_mats,
    )

    rope_mats = [
        rope_setup.cos_matrix_prefill[:, :, :seq_len, :],
        rope_setup.sin_matrix_prefill[:, :, :seq_len, :],
    ]
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq_len, HIDDEN),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = attn(x_tt, rope_mats=rope_mats, position_idx=None, kv_cache=None)
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).reshape(1, seq_len, HIDDEN)

    passing, pcc = comp_pcc(ref_out, out, 0.97)
    logger.info(f"attention prefill vs ref: {pcc}")
    assert passing, f"attention PCC fail: {pcc}"
