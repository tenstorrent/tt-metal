# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-1 PCC test for the full MiniMax-M3 model ASSEMBLY vs a composed torch reference.

Validates the model.py assembly end-to-end on a reduced config: post-embedding hidden states ->
N decoder layers (hybrid dense/MoE schedule) -> final gemma norm -> lm_head -> logits, via
Model.ttnn_prefill_forward. Exercises the layer loop, the moe_layer_freq schedule across layers,
residual chaining, final norm, and lm_head. Anchor: transformers minimax_m3_vl.

Reduced config (2 layers: 1 dense + 1 MoE; small experts/vocab) so it fits one card with random
weights. Real attention dims (6144/64/4/128) + Meta-RoPE swizzle. ttnn_prefill_forward takes
post-embedding hidden states (embedding is applied upstream), so the reference starts from x.
TP=1; full 60-layer + real weights is the follow-up (needs ModelArgs config plumbing + the 869GB pull).
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig, ModeConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.model import Model
from models.demos.minimax_m3.utils.general_utils import get_default_num_links
from models.demos.minimax_m3.utils.weight_conversion import convert_hf_qkv_to_meta_format_partial

from ..test_factory import parametrize_mesh_with_fabric

HIDDEN, NQ, NKV, HEAD_DIM, ROTARY_DIM, THETA, EPS = 6144, 64, 4, 128, 64, 5_000_000.0, 1e-6
INTER, SHARED_INTER, E, TOPK, SCALE, ALPHA, LIMIT, V = 512, 512, 8, 2, 2.0, 1.702, 7.0, 2048
SCHEDULE = [0, 1]  # moe_layer_freq values: 0 -> dense layer, 1 -> MoE layer


def _is_dense(v):
    return v == 0


def _gemma_norm(x, w):
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + EPS)) * (1.0 + w.float())


def _gemma_head_norm(x, w):
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + EPS)) * (1.0 + w.float())


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _rope(t, cos, sin):
    r, p = t[..., :ROTARY_DIM], t[..., ROTARY_DIM:]
    return torch.cat([r * cos + _rotate_half(r) * sin, p], dim=-1)


def _attn(x, w, cos, sin):
    B, S, _ = x.shape
    q = (x @ w["q"].t()).view(B, S, NQ, HEAD_DIM).transpose(1, 2)
    k = (x @ w["k"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
    v = (x @ w["v"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
    q = _rope(_gemma_head_norm(q, w["q_norm"]), cos, sin)
    k = _rope(_gemma_head_norm(k, w["k_norm"]), cos, sin)
    rep = NQ // NKV
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = (q @ k.transpose(-1, -2)) * (HEAD_DIM**-0.5)
    s = s + torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    o = (torch.softmax(s, dim=-1) @ v).transpose(1, 2).reshape(B, S, NQ * HEAD_DIM)
    return o @ w["o"].t()


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
            routed[t] += tw[t, j] * _ffn(x[t : t + 1], *w["experts"][idx[t, j].item()]).squeeze(0)
    return routed + _ffn(x, *w["shared"])


def _layer(x, w, cos, sin, is_dense):
    x = x + _attn(_gemma_norm(x, w["input_ln"]), w, cos, sin)
    normed = _gemma_norm(x, w["post_ln"])
    ffn = _ffn(normed.reshape(-1, HIDDEN), *w["dense"]) if is_dense else _moe(normed.reshape(-1, HIDDEN), w)
    return x + ffn.reshape(x.shape)


def _rand(*s):
    return torch.randn(*s) * 0.02


def _attn_weights():
    return {
        "input_ln": torch.randn(HIDDEN) * 0.1,
        "post_ln": torch.randn(HIDDEN) * 0.1,
        "q": _rand(NQ * HEAD_DIM, HIDDEN),
        "k": _rand(NKV * HEAD_DIM, HIDDEN),
        "v": _rand(NKV * HEAD_DIM, HIDDEN),
        "o": _rand(HIDDEN, NQ * HEAD_DIM),
        "q_norm": torch.randn(HEAD_DIM) * 0.1,
        "k_norm": torch.randn(HEAD_DIM) * 0.1,
    }


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_model_assembly_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Full M3 model (2-layer hybrid, reduced) -> logits vs composed torch ref, TP=1."""
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, HIDDEN) * 0.1

    layer_w = []
    for val in SCHEDULE:
        w = _attn_weights()
        if _is_dense(val):
            w["dense"] = (_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER))
        else:
            w["gate"] = torch.randn(E, HIDDEN) * 0.05
            w["bias"] = torch.randn(E) * 0.1
            w["experts"] = [(_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER)) for _ in range(E)]
            w["shared"] = (_rand(SHARED_INTER, HIDDEN), _rand(SHARED_INTER, HIDDEN), _rand(HIDDEN, SHARED_INTER))
        layer_w.append(w)
    final_norm_w = torch.randn(HIDDEN) * 0.1
    lm_head_w = _rand(V, HIDDEN)

    # --- torch reference ---
    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    emb = torch.cat([torch.outer(torch.arange(seq_len).float(), inv_freq)] * 2, dim=-1)
    cos_ref, sin_ref = emb.cos()[None, None], emb.sin()[None, None]
    h = x.float()
    for w, val in zip(layer_w, SCHEDULE):
        h = _layer(h, w, cos_ref, sin_ref, _is_dense(val))
    ref_logits = _gemma_norm(h, final_norm_w) @ lm_head_w.t()  # [1, S, V]

    # --- build the TT Model from the same weights (config from the flat M3 config.json + overrides) ---
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "MiniMax-M3", "config.json")
    with open(cfg_path) as f:
        c = json.load(f)
    c.update(
        num_hidden_layers=len(SCHEDULE),
        num_local_experts=E,
        num_experts_per_tok=TOPK,
        intermediate_size=INTER,
        vocab_size=V,
        moe_layer_freq=list(SCHEDULE),
    )
    hf_config = SimpleNamespace(**c)

    state = {
        "model.embed_tokens.weight": _rand(V, HIDDEN),  # built but unused by prefill forward (hidden states in)
        "model.norm.weight": final_norm_w,
        "lm_head.weight": lm_head_w,
    }
    for i, (w, val) in enumerate(zip(layer_w, SCHEDULE)):
        p = f"model.layers.{i}."
        state[p + "input_layernorm.weight"] = w["input_ln"]
        state[p + "post_attention_layernorm.weight"] = w["post_ln"]
        state[p + "self_attn.q_proj.weight"] = w["q"]
        state[p + "self_attn.k_proj.weight"] = w["k"]
        state[p + "self_attn.v_proj.weight"] = w["v"]
        state[p + "self_attn.o_proj.weight"] = w["o"]
        state[p + "self_attn.q_norm.weight"] = w["q_norm"]
        state[p + "self_attn.k_norm.weight"] = w["k_norm"]
        if _is_dense(val):
            g, u, d = w["dense"]
            state[p + "mlp.gate_proj.weight"] = g
            state[p + "mlp.up_proj.weight"] = u
            state[p + "mlp.down_proj.weight"] = d
        else:
            state[p + "block_sparse_moe.gate.weight"] = w["gate"]
            state[p + "block_sparse_moe.e_score_correction_bias"] = w["bias"]
            state[p + "block_sparse_moe.shared_experts.gate_proj.weight"] = w["shared"][0]
            state[p + "block_sparse_moe.shared_experts.up_proj.weight"] = w["shared"][1]
            state[p + "block_sparse_moe.shared_experts.down_proj.weight"] = w["shared"][2]
            for e, (w1, w3, w2) in enumerate(w["experts"]):
                state[p + f"block_sparse_moe.experts.{e}.w1.weight"] = w1
                state[p + f"block_sparse_moe.experts.{e}.w3.weight"] = w3
                state[p + f"block_sparse_moe.experts.{e}.w2.weight"] = w2
    state = convert_hf_qkv_to_meta_format_partial(state, HEAD_DIM, ROTARY_DIM)

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1], ep=mesh_device.shape[0]))
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    model = Model(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        create_kv_cache=True,
        max_local_batch_size=1,
    )

    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq_len, HIDDEN),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    logits = model.ttnn_prefill_forward(x_tt, get_last_token=-1)
    out = ttnn.to_torch(
        logits, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(-2, -1))
    ).float()
    out = out.reshape(1, seq_len, -1)[..., :V]

    # bf8 lm_head over a multi-layer stack -> looser PCC; also check last-token argmax matches.
    passing, pcc = comp_pcc(ref_logits, out, 0.94)
    ref_tok = int(ref_logits[0, -1].argmax())
    out_tok = int(out[0, -1].argmax())
    logger.info(f"model assembly vs ref: pcc={pcc} ref_tok={ref_tok} out_tok={out_tok}")
    assert passing, f"model assembly PCC fail: {pcc}"
    assert ref_tok == out_tok, f"last-token argmax mismatch: ref={ref_tok} tt={out_tok}"
