# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Functional MULTI-CARD whole-model test: MiniMax-M3 full Model on (8,4) = TP=4 + EP=32 + DP=8,
vs a self-authored torch reference. Random weights, S<2048 (full-GQA placeholder).

Layout (the M2-validated pattern, M3 dims): 8 prompts, one per mesh ROW (DP=8, users_row_sharded);
attention/dense run TP=4 on the 4 cols; the MoE runs expert-parallel across all 32 chips
(num_experts % 32 == 0). Drives the real prefill I/O: prepare_inputs_prefill (token ids ->
embedding -> per-row shard) -> ttnn_prefill_forward (use_ep_moe) -> per-row logits gather.

TP=4 / EP=32 only DISTRIBUTE the computation, so the golden is the same single-device math as the
#9 test, just per prompt and with the embedding lookup. PCC vs that torch ref per prompt.

Needs TT_MESH_GRAPH_DESC_PATH=single_bh_galaxy ([8,4]). Anchor: transformers minimax_m3_vl.
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
INTER, SHARED_INTER, E, TOPK, SCALE, ALPHA, LIMIT, V = 512, 512, 32, 4, 2.0, 1.702, 7.0, 2048
SCHEDULE = [0, 1]  # layer 0 dense, layer 1 MoE


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


def _layer_weights(val):
    w = {
        "input_ln": torch.randn(HIDDEN) * 0.1,
        "post_ln": torch.randn(HIDDEN) * 0.1,
        "q": _rand(NQ * HEAD_DIM, HIDDEN),
        "k": _rand(NKV * HEAD_DIM, HIDDEN),
        "v": _rand(NKV * HEAD_DIM, HIDDEN),
        "o": _rand(HIDDEN, NQ * HEAD_DIM),
        "q_norm": torch.randn(HEAD_DIM) * 0.1,
        "k_norm": torch.randn(HEAD_DIM) * 0.1,
    }
    if _is_dense(val):
        w["dense"] = (_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER))
    else:
        w["gate"] = torch.randn(E, HIDDEN) * 0.05
        w["bias"] = torch.randn(E) * 0.1
        w["experts"] = [(_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER)) for _ in range(E)]
        w["shared"] = (_rand(SHARED_INTER, HIDDEN), _rand(SHARED_INTER, HIDDEN), _rand(HIDDEN, SHARED_INTER))
    return w


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_model_ep_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Full M3 model on (8,4) TP=4 + EP=32 + DP=8 (8 prompts) vs per-prompt torch ref."""
    rows, cols = mesh_device.shape
    assert (rows, cols) == (8, 4), "this test targets the (8,4) galaxy layout (TP=4, DP=8 rows)"
    torch.manual_seed(0)

    embed_w = _rand(V, HIDDEN)
    final_norm_w = torch.randn(HIDDEN) * 0.1
    lm_head_w = _rand(V, HIDDEN)
    layer_w = [_layer_weights(val) for val in SCHEDULE]

    # 8 prompts (one per row), random token ids.
    toks = torch.randint(0, V, (rows, seq_len), dtype=torch.int32)

    # --- torch reference per prompt: embed -> layers -> final norm -> lm_head ---
    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    emb = torch.cat([torch.outer(torch.arange(seq_len).float(), inv_freq)] * 2, dim=-1)
    cos_ref, sin_ref = emb.cos()[None, None], emb.sin()[None, None]
    ref_logits = []  # per prompt [S, V]
    for r in range(rows):
        h = embed_w[toks[r].long()].unsqueeze(0).float()  # [1, S, H]
        for w, val in zip(layer_w, SCHEDULE):
            h = _layer(h, w, cos_ref, sin_ref, _is_dense(val))
        ref_logits.append((_gemma_norm(h, final_norm_w) @ lm_head_w.t())[0])  # [S, V]

    # --- build the TT Model (config from flat M3 config.json + overrides) ---
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
        "model.embed_tokens.weight": embed_w,
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

    mesh_config = MeshConfig((rows, cols), decode=ModeConfig(tp=cols, ep=rows))
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    model = Model(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        max_local_batch_size=1,
        users_row_sharded=True,
        use_ep_moe=True,
        ep_seq_len_per_chip=seq_len,
    )

    host_out = model.prepare_inputs_prefill(toks, page_table=None, batched_prefill=True)
    logits = model.ttnn_prefill_forward(
        host_out[0],
        rot_mats_global=host_out[1],
        rot_mats_local=host_out[2],
        page_table=host_out[3],
        kv_cache=None,
        batch_size=1,
        get_last_token=-1,
    )
    ttnn.synchronize_device(mesh_device)

    # Per-row gather: device index = r*cols + c; concat col-shards over vocab.
    dts = ttnn.get_device_tensors(logits)
    for r in range(rows):
        row = torch.cat([ttnn.to_torch(dts[r * cols + c]) for c in range(cols)], dim=-1).float()
        row = row.reshape(-1, row.shape[-1])[:seq_len, :V]
        passing, pcc = comp_pcc(ref_logits[r], row, 0.93)
        logger.info(f"prompt{r}: pcc={pcc}")
        assert passing, f"prompt {r} PCC fail: {pcc}"
    logger.info(f"model EP multi-card: all {rows} prompts pass")
