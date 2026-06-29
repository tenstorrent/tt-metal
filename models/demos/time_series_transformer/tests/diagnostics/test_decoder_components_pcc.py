# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Diagnostic: per-sub-step PCC for decoder layer 0.

Goal: localize exactly which sub-step (self-attn, cross-attn, FFN) introduces
the PCC drift that pulls test_decoder_pcc below 0.99 (measured: 0.98995).

Method: call tst_self_attention / tst_cross_attention / tst_ffn directly on
layer 0's weights, feeding the SAME dec_emb / encoder_hidden used by
test_decoder_pcc, and diff each against the matching key in
intermediates.safetensors. This mirrors exactly what tst_decoder_layer does
internally for layer 0, just with PCC checks inserted between each call.

Run from models/demos/time_series_transformer/ with the same env vars as the
main suite:

PYTHONPATH=/root/tt-metal/ttnn:/root/tt-metal/tools:/root/tt-metal/build_Release/lib:. \
TT_METAL_HOME=/root/tt-metal \
LD_LIBRARY_PATH=/root/tt-metal/build_Release/lib \
ARCH_NAME=wormhole_b0 \
pytest /home/claude/tst_debug/test_decoder_components_pcc.py -v -s --noconftest
"""

from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from tt.tst_attention import build_causal_mask, tst_cross_attention, tst_self_attention
from tt.tst_model import load_weights, run_encoder
from tt.ttnn_utils import layer_norm_padded

import ttnn
from models.common.utility_functions import comp_pcc

# Assumes this file lives in models/demos/time_series_transformer/tests/,
# same convention as test_tst_pcc.py / test_tst_e2e.py / test_tst_perf.py.
REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
D_MODEL = 26


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def get_hf_embeddings(hf_model, inputs):
    enc_out, dec_out = {}, {}

    def enc_hook(m, i, o):
        enc_out["emb"] = o.detach()

    def dec_hook(m, i, o):
        dec_out["emb"] = o.detach()

    h1 = hf_model.model.encoder.layernorm_embedding.register_forward_hook(enc_hook)
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
    h1.remove()
    h2.remove()
    return enc_out["emb"], dec_out["emb"]


@pytest.fixture(scope="module")
def setup():
    device = ttnn.open_device(device_id=0)
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    weights = load_weights(hf_model, device)
    inputs = load_ref("inputs.safetensors")
    intermediates = load_ref("intermediates.safetensors")
    yield device, hf_model, weights, inputs, intermediates
    ttnn.close_device(device)


def test_decoder_layer0_components(setup):
    """Per-component PCC for decoder layer 0: self-attn, cross-attn, FFN."""
    device, hf_model, weights, inputs, intermediates = setup

    enc_emb, dec_emb = get_hf_embeddings(hf_model, inputs)
    logger.info(f"dec_emb shape: {dec_emb.shape}")

    # Encoder hidden state (same as test_decoder_pcc does)
    encoder_hidden = run_encoder(device, enc_emb, weights)

    T_dec = dec_emb.shape[1]
    causal_mask = build_causal_mask(device, T_dec)

    w0 = weights["decoder.layers.0"]
    PADDED_WIDTH = 64

    # Pad decoder input to PADDED_WIDTH, same as run_decoder_step does
    import torch.nn.functional as F

    h = dec_emb.float()
    pad = PADDED_WIDTH - h.shape[-1]
    if pad > 0:
        h = F.pad(h, (0, pad))
    hidden_states = ttnn.from_torch(h, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # ---- Step 1: self-attention (pre-residual, pre-layernorm) ----
    self_attn_out = tst_self_attention(hidden_states, w0["self_attn"], causal=True, causal_mask=causal_mask)
    self_attn_torch = ttnn.to_torch(self_attn_out).float()[..., :D_MODEL]
    ref_self_attn = intermediates["decoder_layer0_self_attn"]
    assert (
        self_attn_torch.shape == ref_self_attn.shape
    ), f"self_attn shape mismatch: {self_attn_torch.shape} vs {ref_self_attn.shape}"
    passing, pcc_val = comp_pcc(self_attn_torch, ref_self_attn, 0.99)
    logger.info(f"[1] self_attn PCC: {pcc_val} (passing={passing})")

    # Continue the real layer flow so cross-attn gets a realistic input
    residual = ttnn.add(hidden_states, self_attn_out)
    hidden_states_after_self = layer_norm_padded(
        residual, w0["self_attn_layer_norm_weight"], w0["self_attn_layer_norm_bias"], orig_dim=D_MODEL
    )

    # ---- Step 2: cross-attention (pre-residual, pre-layernorm) ----
    cross_attn_out = tst_cross_attention(hidden_states_after_self, encoder_hidden, w0["encoder_attn"])
    cross_attn_torch = ttnn.to_torch(cross_attn_out).float()[..., :D_MODEL]
    ref_cross_attn = intermediates["decoder_layer0_cross_attn"]
    assert (
        cross_attn_torch.shape == ref_cross_attn.shape
    ), f"cross_attn shape mismatch: {cross_attn_torch.shape} vs {ref_cross_attn.shape}"
    passing, pcc_val = comp_pcc(cross_attn_torch, ref_cross_attn, 0.99)
    logger.info(f"[2] cross_attn PCC: {pcc_val} (passing={passing})")

    residual2 = ttnn.add(hidden_states_after_self, cross_attn_out)
    hidden_states_after_cross = layer_norm_padded(
        residual2, w0["encoder_attn_layer_norm_weight"], w0["encoder_attn_layer_norm_bias"], orig_dim=D_MODEL
    )

    # ---- Step 3: FFN fc1 activation (gelu(fc1(x))) ----
    fc1_out = ttnn.linear(hidden_states_after_cross, w0["fc1_weight"], bias=w0["fc1_bias"], activation="gelu")
    fc1_torch = ttnn.to_torch(fc1_out).float()
    ref_fc1 = intermediates["decoder_layer0_fc1"]
    # fc1 output width is ffn_dim (32 for this config), not D_MODEL -- compare full width
    fc1_width = ref_fc1.shape[-1]
    fc1_torch_trim = fc1_torch[..., :fc1_width]
    assert fc1_torch_trim.shape == ref_fc1.shape, f"fc1 shape mismatch: {fc1_torch_trim.shape} vs {ref_fc1.shape}"
    passing, pcc_val = comp_pcc(fc1_torch_trim, ref_fc1, 0.99)
    logger.info(f"[3] fc1(+gelu) PCC: {pcc_val} (passing={passing})")

    # ---- Step 4: FFN fc2 ----
    fc2_out = ttnn.linear(fc1_out, w0["fc2_weight"], bias=w0["fc2_bias"])
    fc2_torch = ttnn.to_torch(fc2_out).float()[..., :D_MODEL]
    ref_fc2 = intermediates["decoder_layer0_fc2"]
    assert fc2_torch.shape == ref_fc2.shape, f"fc2 shape mismatch: {fc2_torch.shape} vs {ref_fc2.shape}"
    passing, pcc_val = comp_pcc(fc2_torch, ref_fc2, 0.99)
    logger.info(f"[4] fc2 PCC: {pcc_val} (passing={passing})")

    logger.info("=== Summary: see [1]-[4] above to localize the PCC drop ===")
