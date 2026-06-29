# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic: does cross-attention have the same scale-factor bug confirmed in
self-attention (scale = HEAD_DIM_PADDED**-0.5 = 32**-0.5, instead of the
correct HEAD_DIM_TRUE**-0.5 = 13**-0.5)?

This mirrors test_qk_scores_pcc.py's method exactly, adapted for
cross-attention's structure: Q from decoder hidden state, K/V from encoder
hidden state, via tst_cross_attention's actual production code path
(manual reshape/permute, not split_query_key_value_and_split_heads).

K's transpose is NOT auto-detected here (unlike the self-attention test) --
tst_cross_attention's source shows it explicitly: k is permuted to
[B, H, T_enc, 32] then permuted AGAIN to [B, H, 32, T_enc] before the
Q@K matmul. This is known directly from reading the code, not inferred.

Run (after copying into tests/):
PYTHONPATH=/root/tt-metal/ttnn:/root/tt-metal/tools:/root/tt-metal/build_Release/lib:. \
TT_METAL_HOME=/root/tt-metal \
LD_LIBRARY_PATH=/root/tt-metal/build_Release/lib \
ARCH_NAME=wormhole_b0 \
pytest tests/test_cross_attn_qk_scale_pcc.py -v -s --noconftest
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from loguru import logger
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from tt.tst_attention import HEAD_DIM_PADDED, NUM_HEADS
from tt.tst_model import load_weights, run_encoder

import ttnn
from models.common.utility_functions import comp_pcc

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
D_MODEL = 26
HEAD_DIM_TRUE = D_MODEL // NUM_HEADS  # 13
PADDED_WIDTH = NUM_HEADS * HEAD_DIM_PADDED  # 64


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def get_hf_tensors(hf_model, inputs):
    """Capture encoder layernorm_embedding output, decoder layernorm_embedding
    output, and decoder layer 0 cross-attn's raw q_proj/k_proj outputs."""
    captured = {}

    def make_hook(name):
        def fn(module, inp, out):
            captured[name] = out.detach()

        return fn

    cross_attn = hf_model.model.decoder.layers[0].encoder_attn
    handles = [
        hf_model.model.encoder.layernorm_embedding.register_forward_hook(make_hook("enc_emb")),
        hf_model.model.decoder.layernorm_embedding.register_forward_hook(make_hook("dec_emb")),
        cross_attn.q_proj.register_forward_hook(make_hook("hf_q")),
        cross_attn.k_proj.register_forward_hook(make_hook("hf_k")),
    ]
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
    for h in handles:
        h.remove()
    return captured["enc_emb"], captured["dec_emb"], captured["hf_q"], captured["hf_k"]


@pytest.fixture(scope="module")
def setup():
    device = ttnn.open_device(device_id=0)
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    weights = load_weights(hf_model, device)
    inputs = load_ref("inputs.safetensors")
    yield device, hf_model, weights, inputs
    ttnn.close_device(device)


def test_cross_attn_qk_scale(setup):
    device, hf_model, weights, inputs = setup

    enc_emb, dec_emb, hf_q, hf_k = get_hf_tensors(hf_model, inputs)
    B_enc, T_enc, _ = hf_k.shape
    B_dec, T_dec, _ = hf_q.shape
    logger.info(f"hf_q (decoder Q) shape: {hf_q.shape}, hf_k (encoder K) shape: {hf_k.shape}")

    # ---- Pure-torch reference: HF's own head convention, UNSCALED ----
    hf_q_heads = hf_q.view(B_dec, T_dec, NUM_HEADS, HEAD_DIM_TRUE).transpose(1, 2)
    hf_k_heads = hf_k.view(B_enc, T_enc, NUM_HEADS, HEAD_DIM_TRUE).transpose(1, 2)
    ref_scores_unscaled = hf_q_heads @ hf_k_heads.transpose(-2, -1)  # [B, H, T_dec, T_enc]
    logger.info(f"Reference (torch) unscaled cross-attn scores shape: {ref_scores_unscaled.shape}")

    # ---- Build TTNN production encoder_hidden and decoder hidden_states ----
    encoder_hidden = run_encoder(device, enc_emb, weights)  # [B, T_enc, PADDED_WIDTH]

    h = dec_emb.float()
    pad = PADDED_WIDTH - h.shape[-1]
    if pad > 0:
        h = F.pad(h, (0, pad))
    decoder_hidden = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    w0 = weights["decoder.layers.0"]["encoder_attn"]

    # ---- Reproduce tst_cross_attention's EXACT production path up to the
    # unscaled Q@K matmul (copied structure, not a guess) ----
    B = decoder_hidden.shape[0]
    T_dec_ttnn = decoder_hidden.shape[1]
    T_enc_ttnn = encoder_hidden.shape[1]

    query_proj = ttnn.linear(decoder_hidden, w0["q_proj_weight"], bias=w0["q_proj_bias"])
    fused_kv = ttnn.linear(encoder_hidden, w0["kv_weight"], bias=w0["kv_bias"])

    kv_half = NUM_HEADS * HEAD_DIM_PADDED  # 64
    k_proj = ttnn.slice(fused_kv, slice_start=[0, 0, 0], slice_end=[B, T_enc_ttnn, kv_half])

    q = ttnn.reshape(query_proj, (B, T_dec_ttnn, NUM_HEADS, HEAD_DIM_PADDED))
    q = ttnn.permute(q, (0, 2, 1, 3))  # [B, H, T_dec, 32]

    k = ttnn.reshape(k_proj, (B, T_enc_ttnn, NUM_HEADS, HEAD_DIM_PADDED))
    k = ttnn.permute(k, (0, 2, 1, 3))  # [B, H, T_enc, 32]
    k = ttnn.permute(k, (0, 1, 3, 2))  # [B, H, 32, T_enc] -- transposed, per production code

    scores = ttnn.matmul(q, k)  # UNSCALED, exactly mirrors production before scale/softmax
    scores_torch = ttnn.to_torch(scores).float()
    logger.info(f"TTNN raw cross-attn scores shape: {scores_torch.shape}")

    assert (
        scores_torch.shape == ref_scores_unscaled.shape
    ), f"scores shape mismatch: {scores_torch.shape} vs {ref_scores_unscaled.shape}"

    passing, pcc_val = comp_pcc(scores_torch, ref_scores_unscaled, 0.99)
    logger.info(f"[cross-attn Q@K^T unscaled] PCC: {pcc_val} (passing={passing})")

    prod_scale = HEAD_DIM_PADDED**-0.5
    hf_scale = HEAD_DIM_TRUE**-0.5
    logger.info(f"Production scale factor (uses HEAD_DIM_PADDED=32): {prod_scale:.6f}")
    logger.info(f"HF-equivalent scale factor (uses HEAD_DIM_TRUE=13): {hf_scale:.6f}")
    logger.info(
        f"Ratio (prod/hf): {prod_scale / hf_scale:.6f} -- "
        f"if not 1.0, cross-attention has the SAME systematic scale bug as self-attention."
    )

    max_abs_diff = (scores_torch - ref_scores_unscaled).abs().max().item()
    logger.info(f"Max abs diff (unscaled scores): {max_abs_diff:.6f}")
