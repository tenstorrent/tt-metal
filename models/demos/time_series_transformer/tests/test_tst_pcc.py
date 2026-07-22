# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC validation: TTNN TST vs HuggingFace reference tensors.
"""
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from tt.attention import build_causal_mask
from tt.tst_model import run_decoder_step, run_encoder
from tt.tst_weights import load_weights

import ttnn
from models.common.utility_functions import comp_pcc

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
# 0.99: full per-layer comparison (attention + layer norm + FFN chained),
# so bfloat16 rounding accumulates across many ops. See ../CHANGELOG.md
# "PCC threshold policy" for the 0.99 vs 0.999 rule.
PCC_THRESHOLD = 0.99
D_MODEL = 26


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def get_hf_embeddings(hf_model, inputs):
    enc_out, dec_out, dec_final = {}, {}, {}

    def enc_hook(m, i, o):
        enc_out["emb"] = o.detach()

    def dec_hook(m, i, o):
        dec_out["emb"] = o.detach()

    def dec_final_hook(m, i, o):
        dec_final["out"] = (o[0] if isinstance(o, tuple) else o).detach()

    h1 = hf_model.model.encoder.layernorm_embedding.register_forward_hook(enc_hook)
    h2 = hf_model.model.decoder.layernorm_embedding.register_forward_hook(dec_hook)
    h3 = hf_model.model.decoder.layers[-1].register_forward_hook(dec_final_hook)

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
    return enc_out["emb"], dec_out["emb"], dec_final["out"]


@pytest.fixture(scope="module")
def setup():
    device = ttnn.open_device(device_id=0)
    hf_model = TimeSeriesTransformerForPrediction.from_pretrained(MODEL_ID).eval()
    weights = load_weights(hf_model, device)
    inputs = load_ref("inputs.safetensors")
    intermediates = load_ref("intermediates.safetensors")
    outputs = load_ref("outputs.safetensors")
    yield device, hf_model, weights, inputs, intermediates, outputs
    ttnn.close_device(device)


def test_encoder_pcc(setup):
    """Encoder output PCC >= 0.99 vs HF encoder_last_hidden_state.

    PORT NOTE: run_encoder() now requires a ttnn tensor input, not the raw
    torch tensor captured by the HF hook (enc_emb below). The hook captures
    HF's OWN layernorm_embedding output, which is the reference target for
    PCC -- it must be converted to ttnn explicitly here since this test
    intentionally bypasses prepare_encoder_input() to isolate the
    encoder LAYER's correctness from the embedding layer's correctness
    (those are two separate things now that the embedding layer is its
    own ttnn port -- see test_tst_embedding_pcc.py for that layer's own
    isolated check)."""
    device, hf_model, weights, inputs, intermediates, outputs = setup
    enc_emb, _, _ = get_hf_embeddings(hf_model, inputs)
    logger.info(f"Encoder input shape: {enc_emb.shape}")

    enc_emb_padded = torch.nn.functional.pad(enc_emb, (0, 64 - D_MODEL))
    enc_emb_tt = ttnn.from_torch(enc_emb_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    result = run_encoder(device, enc_emb_tt, weights)
    result_torch = ttnn.to_torch(result).float()[..., : enc_emb.shape[-1]]
    ref = outputs["encoder_last_hidden_state"]
    assert result_torch.shape == ref.shape, f"Shape mismatch: {result_torch.shape} vs {ref.shape}"
    passing, pcc_val = comp_pcc(result_torch, ref, PCC_THRESHOLD)
    logger.info(f"Encoder PCC: {pcc_val}")
    assert passing, f"Encoder PCC {pcc_val} < {PCC_THRESHOLD}"


def test_decoder_pcc(setup):
    """Decoder output PCC >= 0.99 vs HF full decoder stack output."""
    device, hf_model, weights, inputs, intermediates, outputs = setup
    enc_emb, dec_emb, dec_ref = get_hf_embeddings(hf_model, inputs)
    logger.info(f"Decoder input shape: {dec_emb.shape}")

    enc_emb_padded = torch.nn.functional.pad(enc_emb, (0, 64 - D_MODEL))
    enc_emb_tt = ttnn.from_torch(enc_emb_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    enc_hidden = run_encoder(device, enc_emb_tt, weights)

    dec_emb_padded = torch.nn.functional.pad(dec_emb, (0, 64 - D_MODEL))
    dec_emb_tt = ttnn.from_torch(dec_emb_padded, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    T_dec = dec_emb.shape[1]
    causal_mask = build_causal_mask(device, T_dec, batch_size=dec_emb_tt.shape[0])

    result = run_decoder_step(device, dec_emb_tt, enc_hidden, weights, causal_mask=causal_mask)
    result_torch = ttnn.to_torch(result).float()[..., : dec_emb.shape[-1]]
    assert result_torch.shape == dec_ref.shape, f"Shape mismatch: {result_torch.shape} vs {dec_ref.shape}"
    passing, pcc_val = comp_pcc(result_torch, dec_ref, PCC_THRESHOLD)
    logger.info(f"Decoder PCC: {pcc_val}")
    assert passing, f"Decoder PCC {pcc_val} < {PCC_THRESHOLD}"
