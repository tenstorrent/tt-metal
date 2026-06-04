# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PCC validation: TTNN TST vs HuggingFace reference tensors.
"""

import pytest
import torch
import ttnn
from pathlib import Path
from safetensors import safe_open
from transformers import TimeSeriesTransformerForPrediction
from loguru import logger
from models.utility_functions import comp_pcc

from tt.tst_model import load_weights, run_encoder, run_decoder_step

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
MODEL_ID = "huggingface/time-series-transformer-tourism-monthly"
PCC_THRESHOLD = 0.99


def load_ref(filename):
    tensors = {}
    with safe_open(str(REFERENCE_DIR / filename), framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def get_hf_embeddings(hf_model, inputs):
    enc_out, dec_out = {}, {}
    def enc_hook(m, i, o): enc_out["emb"] = o.detach()
    def dec_hook(m, i, o): dec_out["emb"] = o.detach()
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
    outputs = load_ref("outputs.safetensors")
    yield device, hf_model, weights, inputs, intermediates, outputs
    ttnn.close_device(device)


def test_encoder_pcc(setup):
    """Encoder output PCC >= 0.99 vs HF encoder_last_hidden_state."""
    device, hf_model, weights, inputs, intermediates, outputs = setup
    enc_emb, _ = get_hf_embeddings(hf_model, inputs)
    logger.info(f"Encoder input shape: {enc_emb.shape}")
    result = run_encoder(device, enc_emb, weights)
    result_torch = ttnn.to_torch(result).float()[..., :enc_emb.shape[-1]]
    ref = outputs["encoder_last_hidden_state"]
    assert result_torch.shape == ref.shape, f"Shape mismatch: {result_torch.shape} vs {ref.shape}"
    passing, pcc_val = comp_pcc(result_torch, ref, PCC_THRESHOLD)
    logger.info(f"Encoder PCC: {pcc_val}")
    assert passing, f"Encoder PCC {pcc_val} < {PCC_THRESHOLD}"


def test_decoder_pcc(setup):
    """Decoder output PCC >= 0.99 vs HF decoder_layer1_fc2."""
    device, hf_model, weights, inputs, intermediates, outputs = setup
    enc_emb, dec_emb = get_hf_embeddings(hf_model, inputs)
    logger.info(f"Decoder input shape: {dec_emb.shape}")
    enc_hidden = run_encoder(device, enc_emb, weights)
    result = run_decoder_step(device, dec_emb, enc_hidden, weights)
    result_torch = ttnn.to_torch(result).float()[..., :dec_emb.shape[-1]]
    ref = intermediates["decoder_layer1_out"]
    assert result_torch.shape == ref.shape, f"Shape mismatch: {result_torch.shape} vs {ref.shape}"
    passing, pcc_val = comp_pcc(result_torch, ref, PCC_THRESHOLD)
    logger.info(f"Decoder PCC: {pcc_val}")
    assert passing, f"Decoder PCC {pcc_val} < {PCC_THRESHOLD}"
