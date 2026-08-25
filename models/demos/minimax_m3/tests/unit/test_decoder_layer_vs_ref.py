# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-1 PCC test for the full MiniMax-M3 DENSE decoder layer (layer 0) vs a torch reference.

Composes the validated dense-layer pieces with residuals (pre-norm transformer block):
    x += attention(input_layernorm(x))
    x += dense_mlp(post_attention_layernorm(x))
where the norms are gemma (1+w) RMSNorm over hidden_size, attention is the M3 GQA block
(per-head gemma QK-norm, partial RoPE, GQA causal SDPA), and the MLP is the clamped-swigluoai
dense FFN (moe_layer_freq[0]==0 selects the dense path in DecoderLayer). Anchor: transformers
minimax_m3_vl decoder layer. Self-authored torch ref + identical random weights; TT module fed
Meta-RoPE-swizzled attn weights. Single card (TP=1).
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.layer import DecoderLayer
from models.demos.minimax_m3.tt.model import create_rope_setup
from models.demos.minimax_m3.utils.general_utils import get_default_num_links
from models.demos.minimax_m3.utils.weight_conversion import convert_hf_qkv_to_meta_format_partial

from ..test_factory import parametrize_mesh_with_fabric

HIDDEN, NQ, NKV, HEAD_DIM, ROTARY_DIM, THETA, EPS = 6144, 64, 4, 128, 64, 5_000_000.0, 1e-6
DENSE_INTER, ALPHA, LIMIT = 12288, 1.702, 7.0


def _gemma_norm(x, w):
    """Full-width gemma RMSNorm over the last dim (hidden)."""
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
    t_rot = t_rot * cos + _rotate_half(t_rot) * sin
    return torch.cat([t_rot, t_pass], dim=-1)


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


def _dense_mlp(x, w):
    gate = (x @ w["gate"].t()).clamp(max=LIMIT)
    up = (x @ w["up"].t()).clamp(min=-LIMIT, max=LIMIT)
    act = (up + 1.0) * (gate * torch.sigmoid(ALPHA * gate))
    return act @ w["down"].t()


def _torch_dense_layer(x, w, cos, sin):
    x = x.float()
    x = x + _attention(_gemma_norm(x, w["input_ln"]), w, cos, sin)
    x = x + _dense_mlp(_gemma_norm(x, w["post_ln"]), w)
    return x


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_dense_decoder_layer_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Full M3 dense decoder layer (layer 0) vs composed torch reference, random weights, TP=1."""
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
        "gate": torch.randn(DENSE_INTER, HIDDEN) * 0.02,
        "up": torch.randn(DENSE_INTER, HIDDEN) * 0.02,
        "down": torch.randn(HIDDEN, DENSE_INTER) * 0.02,
    }

    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos_ref, sin_ref = emb.cos()[None, None], emb.sin()[None, None]

    ref_out = _torch_dense_layer(x, w, cos_ref, sin_ref)

    # Layer state dict; swizzle q/k proj + q/k norm gains for Meta RoPE (helper matches by substring).
    hf_state = {
        "input_layernorm.weight": w["input_ln"],
        "post_attention_layernorm.weight": w["post_ln"],
        "self_attn.q_proj.weight": w["q"],
        "self_attn.k_proj.weight": w["k"],
        "self_attn.v_proj.weight": w["v"],
        "self_attn.o_proj.weight": w["o"],
        "self_attn.q_norm.weight": w["q_norm"],
        "self_attn.k_norm.weight": w["k_norm"],
        "mlp.gate_proj.weight": w["gate"],
        "mlp.up_proj.weight": w["up"],
        "mlp.down_proj.weight": w["down"],
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
        swiglu_alpha=ALPHA,
        swiglu_limit=LIMIT,
        moe_layer_freq=[0],  # layer 0 -> dense path
    )

    mesh_config = MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
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
    )
    assert layer.is_dense, "layer 0 should select the dense MLP path"

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
    tt_out = layer(x_tt, position_embeddings=rope_mats)
    out = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).reshape(1, seq_len, HIDDEN)

    passing, pcc = comp_pcc(ref_out, out, 0.97)
    logger.info(f"dense decoder layer vs ref: {pcc}")
    assert passing, f"dense decoder layer PCC fail: {pcc}"
