# tests/test_attn_v_outproj_pcc.py
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
DIAGNOSTIC -- not part of the real bounty test suite.

Context: self_attn PCC (full tst_self_attention, causal=True, via
tst_decoder_layer) is 0.82 vs HF reference. Already ruled out:
  - masking/softmax precision (fp32 test: PCC unchanged, 0.8295 vs 0.8294)
  - QKV projection + head-split (Q/K/V PCC vs HF: 0.99999/0.99999/0.99985)
  - scale factor bug (HEAD_DIM_PADDED->HEAD_DIM_TRUE fix applied: PCC
    unchanged, 0.8194 vs pre-fix 0.8294 -- within run-to-run noise)

This test isolates what's LEFT in tst_self_attention(causal=True):
    scores = ttnn.matmul(query, key)          [ALREADY VERIFIED: PCC .99999]
    scale, mask, softmax                      [ALREADY VERIFIED: not the cause]
    context = ttnn.matmul(probs, value)        <- testing this
    context = ttnn.transformer.concatenate_heads(context)  <- and this
    return ttnn.linear(context, out_proj_weight, bias=out_proj_bias)  <- and this

KEY DESIGN DECISION (flagged, not hidden):
HF's TimeSeriesTransformerAttention.forward computes attention probs as
plain tensor ops -- there's no hookable submodule for "post-softmax probs".
So this test RECONSTRUCTS HF's true probs by hand from:
  - HF's own q_proj/k_proj outputs (already verified PCC 0.99999/0.99999
    against TTNN's split Q/K in test_qkv_split_pcc.py)
  - HF's scale (13**-0.5) and the same causal mask convention already
    verified to produce 0% future leakage (build_causal_mask).
This is a composition of two already-trusted pieces, not new unverified
machinery -- but it IS an assumption, and it's stated here explicitly
rather than silently baked in: if HF's internal causal masking convention
differs subtly from build_causal_mask's (e.g. mask value, mask dtype,
upcast behavior), this reconstruction could be wrong in a way that looks
like a TTNN bug but isn't. Treat a PASS here as "consistent with no bug
in this composition", not "absolute proof".
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from tt.tst_attention import HEAD_DIM_TRUE, NUM_HEADS, build_causal_mask
from tt.tst_model import load_weights

import ttnn
from models.common.utility_functions import comp_pcc

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
PCC_THRESHOLD = 0.99


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def get_hf_decoder_input_and_qkv(hf_model, inputs):
    """Capture HF's decoder embedding (input to layer 0) and layer 0's
    raw q_proj/k_proj/v_proj outputs -- same hook pattern already used
    and verified in test_qkv_split_pcc.py."""
    captured = {}

    def dec_emb_hook(m, i, o):
        captured["dec_emb"] = o.detach()

    def q_hook(m, i, o):
        captured["hf_q"] = o.detach()

    def k_hook(m, i, o):
        captured["hf_k"] = o.detach()

    def v_hook(m, i, o):
        captured["hf_v"] = o.detach()

    h1 = hf_model.model.decoder.layernorm_embedding.register_forward_hook(dec_emb_hook)
    h2 = hf_model.model.decoder.layers[0].self_attn.q_proj.register_forward_hook(q_hook)
    h3 = hf_model.model.decoder.layers[0].self_attn.k_proj.register_forward_hook(k_hook)
    h4 = hf_model.model.decoder.layers[0].self_attn.v_proj.register_forward_hook(v_hook)

    with torch.no_grad():
        hf_model(
            past_values=inputs["input_past_values"],
            past_time_features=inputs["input_past_time_features"],
            past_observed_mask=inputs["input_past_observed_mask"],
            future_time_features=inputs["input_future_time_features"],
            future_values=inputs["input_future_values"],
            static_categorical_features=inputs["input_static_categorical_features"].long(),
            static_real_features=inputs["input_static_real_features"],
        )
    h1.remove()
    h2.remove()
    h3.remove()
    h4.remove()

    return captured["dec_emb"], captured["hf_q"], captured["hf_k"], captured["hf_v"]


def hf_reference_attn_v_and_outproj(hf_q, hf_k, hf_v, hf_model, T_dec):
    """
    Hand-build HF's true (causal) post-softmax probs from verified q/k,
    then compute probs@V and the real out_proj -- using HF's OWN
    out_proj weights, so the out_proj comparison is apples-to-apples.
    """
    B = hf_q.shape[0]
    head_dim = HEAD_DIM_TRUE  # 13

    def split_heads(x):
        return x.view(B, T_dec, NUM_HEADS, head_dim).transpose(1, 2)  # [B, H, T, 13]

    q = split_heads(hf_q)
    k = split_heads(hf_k)
    v = split_heads(hf_v)

    scale = head_dim**-0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T, T]

    causal = torch.full((T_dec, T_dec), float("-1e9"))
    causal = causal.masked_fill(
        torch.tril(torch.ones(T_dec, T_dec)).bool(), 0.0
    )  # 0 where j<=i, -1e9 where j>i -- matches build_causal_mask convention
    scores = scores + causal.unsqueeze(0).unsqueeze(0)

    probs = torch.softmax(scores, dim=-1)  # [B, H, T, T] -- HF-equivalent ground truth
    context = torch.matmul(probs, v)  # [B, H, T, 13]
    context = context.transpose(1, 2).contiguous().view(B, T_dec, NUM_HEADS * head_dim)  # [B, T, 26]

    out_proj = hf_model.model.decoder.layers[0].self_attn.out_proj
    final = out_proj(context)  # [B, T, 26] -- HF's real out_proj weights

    return probs, context, final


def ttnn_self_attention_internals(device, weights, dec_emb, T_dec):
    """Reproduces tst_self_attention(causal=True) but returns intermediate
    tensors (probs, context pre-out_proj, final) instead of just the final
    output -- so we can localize which stage diverges."""
    from tt.tst_attention import causal_softmax

    w = weights["decoder.layers.0"]["self_attn"]

    pad = 64 - dec_emb.shape[-1]
    h = F.pad(dec_emb.float(), (0, pad)) if pad > 0 else dec_emb.float()
    hidden = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    fused_qkv = ttnn.linear(hidden, w["qkv_weight"], bias=w["qkv_bias"])
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)

    scores = ttnn.matmul(query, key)
    scale = HEAD_DIM_TRUE**-0.5  # post-fix value
    mask = build_causal_mask(device, T_dec)
    probs = causal_softmax(scores, mask, scale)

    context = ttnn.matmul(probs, value)
    context_concat = ttnn.transformer.concatenate_heads(context)
    final = ttnn.linear(context_concat, w["out_proj_weight"], bias=w["out_proj_bias"])

    return probs, context, context_concat, final


def unpad_probs_real_heads(probs_ttnn_torch, T_dec):
    """probs from TTNN: [B, NUM_HEADS, T_dec, T_dec] -- no per-head-width
    padding here (probs is T_dec x T_dec, not head_dim-wide), so this is
    just a direct shape check, no unpadding needed."""
    return probs_ttnn_torch


def unpad_context_real_heads(context_ttnn_torch):
    """context: [B, NUM_HEADS, T_dec, HEAD_DIM_PADDED=32] -> strip each
    head's padding back to HEAD_DIM_TRUE=13 and reassemble contiguously,
    matching HF's [B, T, 26] layout. Same logic already verified correct
    in test_qkv_split_pcc.py's unpad routine."""
    B, H, T, _ = context_ttnn_torch.shape
    real_heads = [context_ttnn_torch[:, h, :, :HEAD_DIM_TRUE] for h in range(H)]
    return torch.cat(real_heads, dim=-1)  # [B, T, NUM_HEADS*13] = [B, T, 26]


@pytest.fixture(scope="module")
def setup():
    device = ttnn.open_device(device_id=0)
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    weights = load_weights(hf_model, device)
    inputs = load_ref("inputs.safetensors")
    yield device, hf_model, weights, inputs
    ttnn.close_device(device)


def test_attn_v_and_outproj_pcc(setup):
    device, hf_model, weights, inputs = setup

    dec_emb, hf_q, hf_k, hf_v = get_hf_decoder_input_and_qkv(hf_model, inputs)
    T_dec = dec_emb.shape[1]
    logger.info(f"dec_emb shape: {dec_emb.shape}, T_dec={T_dec}")

    hf_probs, hf_context, hf_final = hf_reference_attn_v_and_outproj(hf_q, hf_k, hf_v, hf_model, T_dec)

    tt_probs, tt_context, tt_context_concat, tt_final = ttnn_self_attention_internals(device, weights, dec_emb, T_dec)

    # [1] Compare post-softmax probs directly
    tt_probs_torch = ttnn.to_torch(tt_probs).float()
    passing_probs, pcc_probs = comp_pcc(tt_probs_torch, hf_probs, PCC_THRESHOLD)
    logger.info(f"[1] post-softmax probs PCC: {pcc_probs} (passing={passing_probs})")

    # [2] Compare attn@V context (pre-concatenate, pre-out_proj), unpadded to real heads
    tt_context_torch = ttnn.to_torch(tt_context).float()
    tt_context_real = unpad_context_real_heads(tt_context_torch)  # [B, T, 26]
    passing_ctx, pcc_ctx = comp_pcc(tt_context_real, hf_context, PCC_THRESHOLD)
    logger.info(f"[2] attn@V context (unpadded) PCC: {pcc_ctx} (passing={passing_ctx})")

    # [3] Compare concatenate_heads output, unpadded
    tt_concat_torch = ttnn.to_torch(tt_context_concat).float()
    tt_concat_real = tt_concat_torch[..., :26] if tt_concat_torch.shape[-1] != 26 else tt_concat_torch
    # NOTE: concatenate_heads' layout may not match a simple [:26] slice if
    # padding lanes are interleaved per-head rather than trailing -- flagging
    # this as a possible source of false signal in [3] specifically.
    logger.info(f"[3] concatenate_heads raw shape: {tt_concat_torch.shape}")

    # [4] Compare final out_proj output against HF's real out_proj on HF's own context
    tt_final_torch = ttnn.to_torch(tt_final).float()[..., :26]
    passing_final, pcc_final = comp_pcc(tt_final_torch, hf_final, PCC_THRESHOLD)
    logger.info(f"[4] final (out_proj) PCC: {pcc_final} (passing={passing_final})")

    logger.info(
        "=== Summary: [1] probs, [2] attn@V context, [4] final out_proj. "
        "Whichever drops first (reading 1->2->4) localizes the remaining bug. "
        "[3] is shape-only due to concatenate_heads layout uncertainty -- "
        "see note above, don't over-trust a PCC there without confirming layout."
    )
