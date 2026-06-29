# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic: does computing the causal mask-add + softmax in fp32 (instead of
bfloat16) recover PCC for decoder layer 0 self-attention?

Baseline measured: [1] self_attn PCC = 0.8294 (bfloat16 throughout).

This script reproduces tst_self_attention's causal path by hand, but with
scores/mask/softmax done in ttnn.float32, then casts back to bfloat16 only
at the very end (matching what the rest of the pipeline expects). Everything
else (QKV projection, head split, matmul for context, out_proj) stays in
bfloat16, identical to production -- we are isolating ONLY the mask-add +
softmax step.

Run from models/demos/time_series_transformer/, with this file copied into
tests/ first (so the REFERENCE_DIR relative path resolves correctly):

cp <this file> tests/test_fp32_softmax_pcc.py
PYTHONPATH=/root/tt-metal/ttnn:/root/tt-metal/tools:/root/tt-metal/build_Release/lib:. \
TT_METAL_HOME=/root/tt-metal \
LD_LIBRARY_PATH=/root/tt-metal/build_Release/lib \
ARCH_NAME=wormhole_b0 \
pytest tests/test_fp32_softmax_pcc.py -v -s --noconftest
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from tt.tst_attention import HEAD_DIM_PADDED, NEG_INF, NUM_HEADS
from tt.tst_model import load_weights

import ttnn
from models.common.utility_functions import comp_pcc

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
D_MODEL = 26
PADDED_WIDTH = NUM_HEADS * HEAD_DIM_PADDED  # 64


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def get_hf_dec_emb(hf_model, inputs):
    dec_out = {}

    def dec_hook(m, i, o):
        dec_out["emb"] = o.detach()

    h2 = hf_model.model.decoder.layernorm_embedding.register_forward_hook(dec_hook)
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
    h2.remove()
    return dec_out["emb"]


@pytest.fixture(scope="module")
def setup():
    device = ttnn.open_device(device_id=0)
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    weights = load_weights(hf_model, device)
    inputs = load_ref("inputs.safetensors")
    intermediates = load_ref("intermediates.safetensors")
    yield device, hf_model, weights, inputs, intermediates
    ttnn.close_device(device)


def _build_causal_mask_fp32(device, seq_len):
    """Same mask values as build_causal_mask, but stored as ttnn.float32."""
    mask = torch.zeros(seq_len, seq_len)
    mask = mask.masked_fill(torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), NEG_INF)
    mask = mask.unsqueeze(0).unsqueeze(0)
    return ttnn.from_torch(mask, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


def _self_attention_fp32_softmax(hidden_states, w, causal_mask_fp32):
    """
    Identical to tst_self_attention(causal=True), EXCEPT the scaled scores,
    the mask-add, and the softmax are done in ttnn.float32. QKV projection,
    head split, attn@V matmul, and out_proj all stay bfloat16 (unchanged
    from production) -- isolates the mask/softmax step only.
    """
    fused_qkv = ttnn.linear(hidden_states, w["qkv_weight"], bias=w["qkv_bias"])
    query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(fused_qkv, num_heads=NUM_HEADS)

    scores = ttnn.matmul(query, key)  # bfloat16 result
    scale = HEAD_DIM_PADDED**-0.5

    # ---- fp32 mask + softmax (the part under test) ----
    # NOTE: ttnn.typecast's existence/signature was not confirmed against
    # verified docs, so we avoid it. Instead we round-trip through torch,
    # which every file in this codebase already does safely via
    # ttnn.to_torch / ttnn.from_torch (e.g. ttnn_utils.py's pad_to_tile).
    device = scores.device()
    scores_torch = ttnn.to_torch(scores).float()  # fp32 on host
    scores_fp32 = ttnn.from_torch(scores_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    scaled_fp32 = ttnn.multiply(scores_fp32, scale)
    masked_fp32 = ttnn.add(scaled_fp32, causal_mask_fp32)
    probs_fp32 = ttnn.softmax(masked_fp32, dim=-1)

    probs_torch = ttnn.to_torch(probs_fp32).float()
    probs = ttnn.from_torch(probs_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    # ---- end fp32 section ----

    context = ttnn.matmul(probs, value)
    context = ttnn.transformer.concatenate_heads(context)
    return ttnn.linear(context, w["out_proj_weight"], bias=w["out_proj_bias"])


def test_self_attn_fp32_softmax_pcc(setup):
    """Compare fp32-softmax self-attention PCC against the bfloat16 baseline (0.8294)."""
    device, hf_model, weights, inputs, intermediates = setup

    dec_emb = get_hf_dec_emb(hf_model, inputs)
    logger.info(f"dec_emb shape: {dec_emb.shape}")

    h = dec_emb.float()
    pad = PADDED_WIDTH - h.shape[-1]
    if pad > 0:
        h = F.pad(h, (0, pad))
    hidden_states = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    T_dec = dec_emb.shape[1]
    causal_mask_fp32 = _build_causal_mask_fp32(device, T_dec)

    w0 = weights["decoder.layers.0"]["self_attn"]
    self_attn_out = _self_attention_fp32_softmax(hidden_states, w0, causal_mask_fp32)

    self_attn_torch = ttnn.to_torch(self_attn_out).float()[..., :D_MODEL]
    ref_self_attn = intermediates["decoder_layer0_self_attn"]
    assert (
        self_attn_torch.shape == ref_self_attn.shape
    ), f"shape mismatch: {self_attn_torch.shape} vs {ref_self_attn.shape}"

    passing, pcc_val = comp_pcc(self_attn_torch, ref_self_attn, 0.99)
    logger.info(f"[fp32 mask+softmax] self_attn PCC: {pcc_val} (passing={passing})")
    logger.info("Baseline (bfloat16 throughout) was: 0.8294046411958238")
    logger.info(
        "If PCC jumped close to 1.0 here, bfloat16+large-mask softmax is confirmed as the cause. "
        "If it barely moved, the precision loss is elsewhere in self-attention "
        "(e.g. the QKV split / fused projection itself)."
    )
