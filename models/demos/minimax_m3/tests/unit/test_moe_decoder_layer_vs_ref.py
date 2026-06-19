# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-1 PCC test for the full MiniMax-M3 MoE (sparse) decoder layer vs a torch reference.

Same pre-norm block as the dense layer, but the FFN is the MoE block (moe_layer_freq[idx]==1):
    x += attention(input_layernorm(x))
    x += moe(post_attention_layernorm(x))
where moe = routed experts (clamped swigluoai) with routed_scaling_factor + always-on shared
expert. Attention is the full M3 GQA block. Anchor: transformers minimax_m3_vl decoder layer.

Self-authored torch ref + identical random weights; attention weights Meta-RoPE-swizzled for the
TT module. Full attention dims (6144/64/4/128); modest MoE (8 experts/top-2) for speed. TP=1.
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig, ModeConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.layer import DecoderLayer
from models.demos.minimax_m3.tt.model import create_rope_setup
from models.demos.minimax_m3.utils.general_utils import get_default_num_links
from models.demos.minimax_m3.utils.weight_conversion import convert_hf_qkv_to_meta_format_partial

from ..test_factory import parametrize_mesh_with_fabric

HIDDEN, NQ, NKV, HEAD_DIM, ROTARY_DIM, THETA, EPS = 6144, 64, 4, 128, 64, 5_000_000.0, 1e-6
INTER, SHARED_INTER, E, TOPK, SCALE, ALPHA, LIMIT = 512, 512, 8, 2, 2.0, 1.702, 7.0


def _gemma_norm(x, w):
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + EPS)) * (1.0 + w.float())


def _gemma_per_head_norm(x, w):
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + EPS)) * (1.0 + w.float())


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _partial_rope(t, cos, sin):
    t_rot, t_pass = t[..., :ROTARY_DIM], t[..., ROTARY_DIM:]
    return torch.cat([t_rot * cos + _rotate_half(t_rot) * sin, t_pass], dim=-1)


def _attention(x, w, cos, sin):
    B, S, _ = x.shape
    q = (x @ w["q"].t()).view(B, S, NQ, HEAD_DIM).transpose(1, 2)
    k = (x @ w["k"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
    v = (x @ w["v"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
    q = _partial_rope(_gemma_per_head_norm(q, w["q_norm"]), cos, sin)
    k = _partial_rope(_gemma_per_head_norm(k, w["k_norm"]), cos, sin)
    rep = NQ // NKV
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    scores = (q @ k.transpose(-1, -2)) * (HEAD_DIM**-0.5)
    scores = scores + torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    out = (torch.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, S, NQ * HEAD_DIM)
    return out @ w["o"].t()


def _swiglu(g, u):
    g = g.clamp(max=LIMIT)
    u = u.clamp(min=-LIMIT, max=LIMIT)
    return (u + 1.0) * (g * torch.sigmoid(ALPHA * g))


def _ffn(x, w1, w3, w2):
    return _swiglu(x @ w1.t(), x @ w3.t()) @ w2.t()


def _moe(x, w):
    scores = torch.sigmoid(x @ w["gate"].t())
    _, idx = torch.topk(scores + w["bias"], TOPK, dim=-1)
    tw = torch.gather(scores, 1, idx)
    tw = (tw / tw.sum(-1, keepdim=True)) * SCALE
    routed = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(TOPK):
            e = idx[t, j].item()
            routed[t] += tw[t, j] * _ffn(x[t : t + 1], *w["experts"][e]).squeeze(0)
    return routed + _ffn(x, *w["shared"])


def _torch_moe_layer(x, w, cos, sin):
    x = x.float()
    x = x + _attention(_gemma_norm(x, w["input_ln"]), w, cos, sin)
    moe_out = _moe(_gemma_norm(x, w["post_ln"]).reshape(-1, HIDDEN), w).reshape(x.shape)
    return x + moe_out


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_moe_decoder_layer_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Full M3 MoE (sparse) decoder layer vs composed torch reference, random weights, TP=1."""
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, HIDDEN) * 0.1
    w = {
        "input_ln": torch.randn(HIDDEN) * 0.1,
        "post_ln": torch.randn(HIDDEN) * 0.1,
        "q": torch.randn(NQ * HEAD_DIM, HIDDEN) * 0.02,
        "k": torch.randn(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "v": torch.randn(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "o": torch.randn(HIDDEN, NQ * HEAD_DIM) * 0.02,
        "q_norm": torch.randn(HEAD_DIM) * 0.1,
        "k_norm": torch.randn(HEAD_DIM) * 0.1,
        "gate": torch.randn(E, HIDDEN) * 0.05,
        "bias": torch.randn(E) * 0.1,
        "experts": [
            (torch.randn(INTER, HIDDEN) * 0.05, torch.randn(INTER, HIDDEN) * 0.05, torch.randn(HIDDEN, INTER) * 0.05)
            for _ in range(E)
        ],
        "shared": (
            torch.randn(SHARED_INTER, HIDDEN) * 0.05,
            torch.randn(SHARED_INTER, HIDDEN) * 0.05,
            torch.randn(HIDDEN, SHARED_INTER) * 0.05,
        ),
    }

    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    emb = torch.cat([torch.outer(torch.arange(seq_len).float(), inv_freq)] * 2, dim=-1)
    cos_ref, sin_ref = emb.cos()[None, None], emb.sin()[None, None]

    ref_out = _torch_moe_layer(x, w, cos_ref, sin_ref)

    hf_state = {
        "input_layernorm.weight": w["input_ln"],
        "post_attention_layernorm.weight": w["post_ln"],
        "self_attn.q_proj.weight": w["q"],
        "self_attn.k_proj.weight": w["k"],
        "self_attn.v_proj.weight": w["v"],
        "self_attn.o_proj.weight": w["o"],
        "self_attn.q_norm.weight": w["q_norm"],
        "self_attn.k_norm.weight": w["k_norm"],
        "block_sparse_moe.gate.weight": w["gate"],
        "block_sparse_moe.e_score_correction_bias": w["bias"],
        "block_sparse_moe.shared_experts.gate_proj.weight": w["shared"][0],
        "block_sparse_moe.shared_experts.up_proj.weight": w["shared"][1],
        "block_sparse_moe.shared_experts.down_proj.weight": w["shared"][2],
    }
    for e, (w1, w3, w2) in enumerate(w["experts"]):
        hf_state[f"block_sparse_moe.experts.{e}.w1.weight"] = w1
        hf_state[f"block_sparse_moe.experts.{e}.w3.weight"] = w3
        hf_state[f"block_sparse_moe.experts.{e}.w2.weight"] = w2
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
        swiglu_alpha=ALPHA,
        swiglu_limit=LIMIT,
        intermediate_size=INTER,
        num_local_experts=E,
        num_experts_per_tok=TOPK,
        routed_scaling_factor=SCALE,
        moe_layer_freq=[1],  # layer 0 -> MoE path
    )

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1], ep=mesh_device.shape[0]))
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    rope_setup = create_rope_setup(mesh_device=mesh_device, hf_config=hf_config, datatype=ttnn.bfloat16)

    layer = DecoderLayer(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state,
        layer_idx=0,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        transformation_mats=rope_setup.get_both_trans_mats(),
        max_seq_len=max(seq_len, 128),
        create_kv_cache=True,
        expert_weight_dtype=ttnn.bfloat16,
    )
    assert not layer.is_dense, "layer with moe_layer_freq[0]==1 should select the MoE path"

    rope_mats = [rope_setup.cos_matrix_prefill[:, :, :seq_len, :], rope_setup.sin_matrix_prefill[:, :, :seq_len, :]]
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq_len, HIDDEN),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_out = layer(x_tt, position_embeddings=rope_mats)
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).reshape(1, seq_len, HIDDEN)

    passing, pcc = comp_pcc(ref_out, out, 0.97)
    logger.info(f"moe decoder layer vs ref: {pcc}")
    assert passing, f"MoE decoder layer PCC fail: {pcc}"
