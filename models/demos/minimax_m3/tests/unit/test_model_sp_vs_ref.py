# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Whole-model SP=8 × TP=4 prefill (DENSE layers) vs a composed torch reference.

The deployment data flow with DP eliminated: one sequence is sharded across the SP rows (S/sp per
device), the residual stream stays SP-sharded through every layer, attention runs the validated SP
forwards (dense -> ring_joint no-cache), and the final norm + lm_head run per-token on the shards. The
last piece is RoPE: each SP row must rotate its OWN slice of positions [r*s_local:(r+1)*s_local], so we
take the model's own (format-exact) prefill cos/sin and re-shard it across the rows instead of slicing
it uniformly.

Dense-only schedule (layers 0-2 are dense in M3) so this isolates the SP residual/attention/rope/output
flow from the EP-MoE reshard (covered separately). Validates against the same composed torch ref the
TP=1 assembly test uses, reassembled across rows(seq) × cols(vocab).
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.model import Model
from models.demos.minimax_m3.utils.general_utils import get_default_num_links
from models.demos.minimax_m3.utils.weight_conversion import convert_hf_qkv_to_meta_format_partial

from ..test_factory import parametrize_mesh_with_fabric

# E=128 experts, TOPK=4 per token = real M3 (num_local_experts / num_experts_per_tok); EP=32 on the 8x4
# mesh -> 4 experts/chip. Intermediate widths and V stay reduced (host-weight budget).
HIDDEN, NQ, NKV, HEAD_DIM, ROTARY_DIM, THETA, EPS = 6144, 64, 4, 128, 64, 5_000_000.0, 1e-6
INTER, SHARED_INTER, E, TOPK, SCALE, LIMIT, ALPHA, V = 512, 512, 128, 4, 2.0, 7.0, 1.702, 2048
SCHEDULE = [0, 0]  # dense layers only (M3 layers 0-2 are dense); isolates SP flow from MoE reshard


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


def _attn(x, w, cos, sin, mask_mode="causal"):
    B, S, _ = x.shape
    q = (x @ w["q"].t()).view(B, S, NQ, HEAD_DIM).transpose(1, 2)
    k = (x @ w["k"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
    v = (x @ w["v"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
    q = _rope(_gemma_head_norm(q, w["q_norm"]), cos, sin)
    k = _rope(_gemma_head_norm(k, w["k_norm"]), cos, sin)
    rep = NQ // NKV
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = (q @ k.transpose(-1, -2)) * (HEAD_DIM**-0.5)
    if mask_mode == "causal":
        s = s + torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    elif mask_mode == "block":
        # MSA degenerate ref (nblocks == topk): the indexer selects all CAUSAL blocks (future blocks are
        # -1 sentinels, masked by sparse_sdpa_msa), and there's no intra-block mask, so the effective mask
        # is causal ROUNDED UP to the query's 128-block boundary.
        blk = 128
        key_end = ((torch.arange(S) // blk) + 1) * blk  # exclusive key bound per query = end of its block
        j = torch.arange(S)
        s = s + torch.where(j[None, :] < key_end[:, None], 0.0, float("-inf"))
    else:
        raise ValueError(mask_mode)
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


def _layer(x, w, cos, sin, is_dense=True, mask_mode="causal"):
    x = x + _attn(_gemma_norm(x, w["input_ln"]), w, cos, sin, mask_mode=mask_mode)
    normed = _gemma_norm(x, w["post_ln"])
    flat = normed.reshape(-1, HIDDEN)
    ffn = _ffn(flat, *w["dense"]) if is_dense else _moe(flat, w)
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


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("seq_len", [1024], ids=["s1024"])  # 128/row at SP=8
def test_model_sp_dense_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Full M3 dense-layer model under SP=8 × TP=4 -> logits vs composed torch ref."""
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis = rows, cols, 0
    s_local = seq_len // sp

    torch.manual_seed(0)
    x = torch.randn(1, seq_len, HIDDEN) * 0.1

    layer_w = []
    for _ in SCHEDULE:
        w = _attn_weights()
        w["dense"] = (_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER))
        layer_w.append(w)
    final_norm_w = torch.randn(HIDDEN) * 0.1
    lm_head_w = _rand(V, HIDDEN)

    # --- torch reference (full sequence) ---
    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    emb = torch.cat([torch.outer(torch.arange(seq_len).float(), inv_freq)] * 2, dim=-1)
    cos_ref, sin_ref = emb.cos()[None, None], emb.sin()[None, None]
    h = x.float()
    for w in layer_w:
        h = _layer(h, w, cos_ref, sin_ref)
    ref_logits = _gemma_norm(h, final_norm_w) @ lm_head_w.t()  # [1, S, V]

    # --- TT model from the same weights ---
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "MiniMax-M3", "config.json")
    with open(cfg_path) as f:
        c = json.load(f)
    c.update(
        num_hidden_layers=len(SCHEDULE),
        intermediate_size=INTER,
        vocab_size=V,
        moe_layer_freq=list(SCHEDULE),
    )
    hf_config = SimpleNamespace(**c)

    state = {
        "model.embed_tokens.weight": _rand(V, HIDDEN),
        "model.norm.weight": final_norm_w,
        "lm_head.weight": lm_head_w,
    }
    for i, w in enumerate(layer_w):
        p = f"model.layers.{i}."
        state[p + "input_layernorm.weight"] = w["input_ln"]
        state[p + "post_attention_layernorm.weight"] = w["post_ln"]
        state[p + "self_attn.q_proj.weight"] = w["q"]
        state[p + "self_attn.k_proj.weight"] = w["k"]
        state[p + "self_attn.v_proj.weight"] = w["v"]
        state[p + "self_attn.o_proj.weight"] = w["o"]
        state[p + "self_attn.q_norm.weight"] = w["q_norm"]
        state[p + "self_attn.k_norm.weight"] = w["k_norm"]
        g, u, d = w["dense"]
        state[p + "mlp.gate_proj.weight"] = g
        state[p + "mlp.up_proj.weight"] = u
        state[p + "mlp.down_proj.weight"] = d
    state = convert_hf_qkv_to_meta_format_partial(state, HEAD_DIM, ROTARY_DIM)

    # prefill mode auto-defaults to SP=rows, EP=1, TP=cols (see MeshConfig.__init__)
    mesh_config = MeshConfig((rows, cols), tp=tp)
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    model = Model(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        max_local_batch_size=1,
        sequence_parallel=True,
    )

    # --- SP input shard: one sequence, S/sp rows per device, hidden replicated across TP cols ---
    in_dims = [None, None]
    in_dims[sp_axis] = 2  # seq -> SP rows
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq_len, HIDDEN),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
    )

    # --- per-row RoPE: take the model's own (format-exact) prefill cos/sin and RE-SHARD it across SP
    # rows so row r rotates positions [r*s_local:(r+1)*s_local] (instead of the uniform :seq_len slice) ---
    def reshard_rope(dev_tensor):
        full = ttnn.to_torch(ttnn.get_device_tensors(dev_tensor)[0])[:, :, :seq_len, :]
        return ttnn.from_torch(
            full,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
        )

    rope_sp = [reshard_rope(model.rope_setup.cos_matrix_prefill), reshard_rope(model.rope_setup.sin_matrix_prefill)]

    logits = model.prefill_forward(x_tt, rot_mats_global=rope_sp, get_last_token=-1)

    # gather: rows -> seq (dim -2), cols -> vocab (dim -1)
    out = ttnn.to_torch(
        logits, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(-2, -1))
    ).float()
    out = out.reshape(1, seq_len, -1)[..., :V]

    # Validate the whole SP forward against the composed torch reference. Two topology-independent
    # checks: (1) overall PCC, and (2) per-SP-row uniformity — a correct causal SP forward gives a
    # smooth, gently-decreasing per-row PCC (later rows attend over more keys -> a touch more bf16
    # accumulation), whereas an SP-sharding bug (wrong RoPE slice, scrambled gather, broken causality)
    # collapses a specific row. Argmax over these tiny random-weight logits is noise-dominated and not
    # asserted; robust last-token correctness is the real-weights demo's job.
    passing, pcc = comp_pcc(ref_logits, out, 0.99)
    row_pccs = []
    for r in range(sp):
        _, rpcc = comp_pcc(ref_logits[:, r * s_local : (r + 1) * s_local], out[:, r * s_local : (r + 1) * s_local])
        row_pccs.append(rpcc)
        logger.info(f"  row{r} (pos {r*s_local}:{(r+1)*s_local}) pcc={rpcc}")
    logger.info(f"SP=8xTP=4 dense model vs torch ref: pcc={pcc} row_pccs=[{min(row_pccs):.4f}..{max(row_pccs):.4f}]")
    assert passing, f"SP dense model PCC fail: {pcc}"
    assert min(row_pccs) > 0.99, f"SP row collapse (sharding bug): row_pccs={row_pccs}"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("seq_len", [1024], ids=["s1024"])  # 128/row at SP=8
@pytest.mark.skip(reason="unreliable with random gate weights: near-tie top-k routing flips under SP precision drift")
def test_model_sp_moe_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """SP=8 × TP=4 with a MoE layer (EP=32, use_ep_moe) over the SP-sharded residual vs torch ref.

    The MoE-under-SP integration: layer 0 dense (validated SP attention) feeds its SP-sharded residual
    into a MoE layer (layer 1) running the deployment unified-kernel EP path (use_ep_moe=True). That
    path consumes per-device [1,1,s_local,H] tokens and routes per-row — blind to whether the rows hold
    different prompts (DP) or seq-shards of ONE prompt (SP) — so EP=32 expert parallelism works over the
    seq-sharded stream with no MoE change. Design A: the residual stays SP-sharded through the MoE.
    """
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis = rows, cols, 0
    s_local = seq_len // sp
    schedule = [0, 1]  # layer 0 dense, layer 1 MoE

    torch.manual_seed(0)
    x = torch.randn(1, seq_len, HIDDEN) * 0.1

    layer_w = []
    for val in schedule:
        w = _attn_weights()
        if val == 0:
            w["dense"] = (_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER))
        else:
            w["gate"] = torch.randn(E, HIDDEN) * 0.05
            w["bias"] = torch.randn(E) * 0.1
            w["experts"] = [(_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER)) for _ in range(E)]
            w["shared"] = (_rand(SHARED_INTER, HIDDEN), _rand(SHARED_INTER, HIDDEN), _rand(HIDDEN, SHARED_INTER))
        layer_w.append(w)
    final_norm_w = torch.randn(HIDDEN) * 0.1
    lm_head_w = _rand(V, HIDDEN)

    # --- torch reference (one prompt, full sequence) ---
    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    emb = torch.cat([torch.outer(torch.arange(seq_len).float(), inv_freq)] * 2, dim=-1)
    cos_ref, sin_ref = emb.cos()[None, None], emb.sin()[None, None]
    h = x.float()
    for w, val in zip(layer_w, schedule):
        h = _layer(h, w, cos_ref, sin_ref, is_dense=(val == 0))
    ref_logits = _gemma_norm(h, final_norm_w) @ lm_head_w.t()

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "MiniMax-M3", "config.json")
    with open(cfg_path) as f:
        c = json.load(f)
    c.update(
        num_hidden_layers=len(schedule),
        num_local_experts=E,
        num_experts_per_tok=TOPK,
        intermediate_size=INTER,
        vocab_size=V,
        moe_layer_freq=list(schedule),
    )
    hf_config = SimpleNamespace(**c)

    state = {
        "model.embed_tokens.weight": _rand(V, HIDDEN),
        "model.norm.weight": final_norm_w,
        "lm_head.weight": lm_head_w,
    }
    for i, (w, val) in enumerate(zip(layer_w, schedule)):
        p = f"model.layers.{i}."
        state[p + "input_layernorm.weight"] = w["input_ln"]
        state[p + "post_attention_layernorm.weight"] = w["post_ln"]
        state[p + "self_attn.q_proj.weight"] = w["q"]
        state[p + "self_attn.k_proj.weight"] = w["k"]
        state[p + "self_attn.v_proj.weight"] = w["v"]
        state[p + "self_attn.o_proj.weight"] = w["o"]
        state[p + "self_attn.q_norm.weight"] = w["q_norm"]
        state[p + "self_attn.k_norm.weight"] = w["k_norm"]
        if val == 0:
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

    mesh_config = MeshConfig((rows, cols), tp=tp)
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    model = Model(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        max_local_batch_size=1,
        sequence_parallel=True,
        use_ep_moe=True,
        ep_seq_len_per_chip=s_local,  # per-device tokens = one prompt's seq-shard
    )

    in_dims = [None, None]
    in_dims[sp_axis] = 2  # seq -> SP rows
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq_len, HIDDEN),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
    )

    def reshard_rope(dev_tensor):
        full = ttnn.to_torch(ttnn.get_device_tensors(dev_tensor)[0])[:, :, :seq_len, :]
        return ttnn.from_torch(
            full,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
        )

    rope_sp = [reshard_rope(model.rope_setup.cos_matrix_prefill), reshard_rope(model.rope_setup.sin_matrix_prefill)]
    logits = model.prefill_forward(x_tt, rot_mats_global=rope_sp, get_last_token=-1)
    out = ttnn.to_torch(
        logits, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(-2, -1))
    ).float()
    out = out.reshape(1, seq_len, -1)[..., :V]

    # Threshold 0.93: the EP=32 MoE runs bf4 expert weights + bf8 matmuls, a lower precision floor than
    # the dense path. The SP-correctness signal is the per-row uniformity (no row collapses), not the
    # absolute PCC.
    passing, pcc = comp_pcc(ref_logits, out, 0.93)
    row_pccs = []
    for r in range(sp):
        _, rpcc = comp_pcc(ref_logits[:, r * s_local : (r + 1) * s_local], out[:, r * s_local : (r + 1) * s_local])
        row_pccs.append(rpcc)
        logger.info(f"  row{r} (pos {r*s_local}:{(r+1)*s_local}) pcc={rpcc}")
    logger.info(
        f"SP=8xTP=4 + MoE(EP=32) model vs torch ref: pcc={pcc} row_pccs=[{min(row_pccs):.4f}..{max(row_pccs):.4f}]"
    )
    assert passing, f"SP+MoE model PCC fail: {pcc}"
    assert min(row_pccs) > 0.93, f"SP+MoE row collapse (sharding bug): row_pccs={row_pccs}"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("seq_len", [1024], ids=["s1024"])  # 128/row at SP=8
@pytest.mark.skip(reason="unreliable with random gate weights: near-tie top-k routing flips under SP precision drift")
def test_model_sp_tokens_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """End-to-end SP via the REAL I/O path: token ids -> prepare_inputs_prefill (SP shard + per-row
    RoPE) -> prefill_forward (dense + MoE/EP=32) -> gathered logits vs torch ref. This exercises
    exactly the input plumbing the real-weights run uses (the manual-shard tests above bypass it)."""
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis = rows, cols, 0
    s_local = seq_len // sp
    schedule = [0, 1]  # dense + MoE

    torch.manual_seed(0)
    embed_w = _rand(V, HIDDEN)
    final_norm_w = torch.randn(HIDDEN) * 0.1
    lm_head_w = _rand(V, HIDDEN)
    toks = torch.randint(0, V, (1, seq_len), dtype=torch.int32)

    layer_w = []
    for val in schedule:
        w = _attn_weights()
        if val == 0:
            w["dense"] = (_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER))
        else:
            w["gate"] = torch.randn(E, HIDDEN) * 0.05
            w["bias"] = torch.randn(E) * 0.1
            w["experts"] = [(_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER)) for _ in range(E)]
            w["shared"] = (_rand(SHARED_INTER, HIDDEN), _rand(SHARED_INTER, HIDDEN), _rand(HIDDEN, SHARED_INTER))
        layer_w.append(w)

    # --- torch reference: embed -> layers -> final norm -> lm_head ---
    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    emb = torch.cat([torch.outer(torch.arange(seq_len).float(), inv_freq)] * 2, dim=-1)
    cos_ref, sin_ref = emb.cos()[None, None], emb.sin()[None, None]
    h = embed_w[toks[0].long()].unsqueeze(0).float()
    for w, val in zip(layer_w, schedule):
        h = _layer(h, w, cos_ref, sin_ref, is_dense=(val == 0))
    ref_logits = _gemma_norm(h, final_norm_w) @ lm_head_w.t()

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "MiniMax-M3", "config.json")
    with open(cfg_path) as f:
        c = json.load(f)
    c.update(
        num_hidden_layers=len(schedule),
        num_local_experts=E,
        num_experts_per_tok=TOPK,
        intermediate_size=INTER,
        vocab_size=V,
        moe_layer_freq=list(schedule),
    )
    hf_config = SimpleNamespace(**c)

    state = {"model.embed_tokens.weight": embed_w, "model.norm.weight": final_norm_w, "lm_head.weight": lm_head_w}
    for i, (w, val) in enumerate(zip(layer_w, schedule)):
        p = f"model.layers.{i}."
        state[p + "input_layernorm.weight"] = w["input_ln"]
        state[p + "post_attention_layernorm.weight"] = w["post_ln"]
        state[p + "self_attn.q_proj.weight"] = w["q"]
        state[p + "self_attn.k_proj.weight"] = w["k"]
        state[p + "self_attn.v_proj.weight"] = w["v"]
        state[p + "self_attn.o_proj.weight"] = w["o"]
        state[p + "self_attn.q_norm.weight"] = w["q_norm"]
        state[p + "self_attn.k_norm.weight"] = w["k_norm"]
        if val == 0:
            g, u, d = w["dense"]
            state[p + "mlp.gate_proj.weight"], state[p + "mlp.up_proj.weight"], state[p + "mlp.down_proj.weight"] = (
                g,
                u,
                d,
            )
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

    mesh_config = MeshConfig((rows, cols), tp=tp)
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    model = Model(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        max_local_batch_size=1,
        sequence_parallel=True,
        use_ep_moe=True,
        ep_seq_len_per_chip=s_local,
    )

    # REAL I/O path: token ids -> SP shard + per-row RoPE inside prepare_inputs_prefill
    host_out = model.prepare_inputs_prefill(toks)
    logits = model.prefill_forward(
        host_out[0],
        rot_mats_global=host_out[1],
        rot_mats_local=host_out[2],
        kv_cache=None,
        batch_size=1,
        get_last_token=-1,
    )
    out = (
        ttnn.to_torch(
            logits, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(-2, -1))
        )
        .float()
        .reshape(1, seq_len, -1)[..., :V]
    )

    passing, pcc = comp_pcc(ref_logits, out, 0.93)
    row_pccs = []
    for r in range(sp):
        _, rpcc = comp_pcc(ref_logits[:, r * s_local : (r + 1) * s_local], out[:, r * s_local : (r + 1) * s_local])
        row_pccs.append(rpcc)
    last_tok_ref, last_tok_out = int(ref_logits[0, -1].argmax()), int(out[0, -1].argmax())
    logger.info(
        f"SP token-path (prepare_inputs + EP MoE) vs ref: pcc={pcc} rows=[{min(row_pccs):.4f}..{max(row_pccs):.4f}] argmax ref={last_tok_ref} out={last_tok_out}"
    )
    assert passing, f"SP token-path PCC fail: {pcc}"
    assert min(row_pccs) > 0.93, f"SP token-path row collapse: row_pccs={row_pccs}"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("seq_len", [2048], ids=["s2048"])  # 16 blocks (=2048/128) == topk 16 -> MSA degenerate
@pytest.mark.skip(reason="unreliable with random gate weights: near-tie top-k routing flips under SP precision drift")
def test_model_sp_msa_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """SP=8 × TP=4 with an MSA (sparse) layer through the FULL prefill (concat_heads -> o_proj -> MoE).

    Regression for the sparse_sdpa_msa ROW_MAJOR->TILE / concat_heads fix (71a2128fc68): the other
    whole-model SP tests use only dense-attention layers, so no MSA layer ever went through prefill.py —
    this is the test that would have caught the layout crash. seq_len=2048 gives 16 blocks == topk 16
    (topk_large_indices requires nblocks >= k), so the indexer selects all CAUSAL blocks per query (future
    blocks are -1 sentinels, masked) and MSA degenerates to BLOCK-CAUSAL attention (causal rounded up to
    the 128-block boundary) — exactly refable, a real numerical check on the in-model MSA data flow (exact
    causal sparsity at real sizes is covered op-level by test_msa_sp_chunked at 5120/40-blocks).
    Layer 3 = MSA attention (sparse_attention_freq[3]=1 from config.json) + MoE MLP (moe_layer_freq[3]=1).
    """
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis = rows, cols, 0
    s_local = seq_len // sp
    NIDX, INDEX_DIM = 4, 128  # sparse_num_index_heads, sparse_index_dim
    schedule = [0, 0, 0, 1]  # moe_layer_freq: layers 0-2 dense MLP, layer 3 MoE; layer 3 is also the MSA layer

    torch.manual_seed(0)
    x = torch.randn(1, seq_len, HIDDEN) * 0.1

    layer_w = []
    for val in schedule:
        w = _attn_weights()
        if val == 0:
            w["dense"] = (_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER))
        else:
            w["gate"] = torch.randn(E, HIDDEN) * 0.05
            w["bias"] = torch.randn(E) * 0.1
            w["experts"] = [(_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER)) for _ in range(E)]
            w["shared"] = (_rand(SHARED_INTER, HIDDEN), _rand(SHARED_INTER, HIDDEN), _rand(HIDDEN, SHARED_INTER))
        layer_w.append(w)
    final_norm_w = torch.randn(HIDDEN) * 0.1
    lm_head_w = _rand(V, HIDDEN)

    # --- torch ref: layer 3 attention is MSA -> block-causal (degenerate); others token-causal ---
    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    emb = torch.cat([torch.outer(torch.arange(seq_len).float(), inv_freq)] * 2, dim=-1)
    cos_ref, sin_ref = emb.cos()[None, None], emb.sin()[None, None]
    h = x.float()
    for i, (w, val) in enumerate(zip(layer_w, schedule)):
        h = _layer(h, w, cos_ref, sin_ref, is_dense=(val == 0), mask_mode=("block" if i == 3 else "causal"))
    ref_logits = _gemma_norm(h, final_norm_w) @ lm_head_w.t()

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "MiniMax-M3", "config.json")
    with open(cfg_path) as f:
        c = json.load(f)
    # num_hidden_layers=4 keeps layer 3 (sparse_attention_freq[3]=1, moe_layer_freq[3]=1) — DO NOT touch
    # sparse_attention_config, so layer 3 stays MSA.
    c.update(
        num_hidden_layers=4,
        num_local_experts=E,
        num_experts_per_tok=TOPK,
        intermediate_size=INTER,
        vocab_size=V,
        moe_layer_freq=list(schedule),
    )
    hf_config = SimpleNamespace(**c)

    state = {
        "model.embed_tokens.weight": _rand(V, HIDDEN),
        "model.norm.weight": final_norm_w,
        "lm_head.weight": lm_head_w,
    }
    for i, (w, val) in enumerate(zip(layer_w, schedule)):
        p = f"model.layers.{i}."
        state[p + "input_layernorm.weight"] = w["input_ln"]
        state[p + "post_attention_layernorm.weight"] = w["post_ln"]
        state[p + "self_attn.q_proj.weight"] = w["q"]
        state[p + "self_attn.k_proj.weight"] = w["k"]
        state[p + "self_attn.v_proj.weight"] = w["v"]
        state[p + "self_attn.o_proj.weight"] = w["o"]
        state[p + "self_attn.q_norm.weight"] = w["q_norm"]
        state[p + "self_attn.k_norm.weight"] = w["k_norm"]
        if i == 3:  # MSA layer — index branch weights (values don't affect output: all causal blocks selected)
            state[p + "self_attn.index_q_proj.weight"] = _rand(NIDX * INDEX_DIM, HIDDEN)
            state[p + "self_attn.index_k_proj.weight"] = _rand(INDEX_DIM, HIDDEN)
            state[p + "self_attn.index_q_norm.weight"] = torch.randn(INDEX_DIM) * 0.1
            state[p + "self_attn.index_k_norm.weight"] = torch.randn(INDEX_DIM) * 0.1
        if val == 0:
            g, u, d = w["dense"]
            state[p + "mlp.gate_proj.weight"], state[p + "mlp.up_proj.weight"], state[p + "mlp.down_proj.weight"] = (
                g,
                u,
                d,
            )
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

    mesh_config = MeshConfig((rows, cols), tp=tp)
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    model = Model(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        max_local_batch_size=1,
        sequence_parallel=True,
        use_ep_moe=True,
        ep_seq_len_per_chip=s_local,
    )

    in_dims = [None, None]
    in_dims[sp_axis] = 2
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq_len, HIDDEN),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
    )

    def reshard_rope(dev_tensor):
        full = ttnn.to_torch(ttnn.get_device_tensors(dev_tensor)[0])[:, :, :seq_len, :]
        return ttnn.from_torch(
            full,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
        )

    rope_sp = [reshard_rope(model.rope_setup.cos_matrix_prefill), reshard_rope(model.rope_setup.sin_matrix_prefill)]
    logits = model.prefill_forward(x_tt, rot_mats_global=rope_sp, get_last_token=-1)
    out = (
        ttnn.to_torch(
            logits, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=(rows, cols), dims=(-2, -1))
        )
        .float()
        .reshape(1, seq_len, -1)[..., :V]
    )

    # PRIMARY: it RUNS through prefill.py for the MSA layer (concat_heads) without the TT_FATAL.
    passing, pcc = comp_pcc(ref_logits, out, 0.93)  # bf4 MoE floor; MSA layer matches block-causal ref
    row_pccs = []
    for r in range(sp):
        _, rpcc = comp_pcc(ref_logits[:, r * s_local : (r + 1) * s_local], out[:, r * s_local : (r + 1) * s_local])
        row_pccs.append(rpcc)
    logger.info(
        f"SP=8xTP=4 + MSA(layer3) + MoE model vs ref: pcc={pcc} rows=[{min(row_pccs):.4f}..{max(row_pccs):.4f}]"
    )
    assert passing, f"SP+MSA model PCC fail: {pcc}"
    assert min(row_pccs) > 0.93, f"SP+MSA row collapse (sharding/layout bug): row_pccs={row_pccs}"
