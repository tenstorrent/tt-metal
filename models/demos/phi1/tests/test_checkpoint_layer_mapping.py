# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# CPU-only regression check for the microsoft/phi-1 checkpoint layer mapping.
# Does not import ttnn or open a device: only validates that demo.py's layer
# truncation and base_address detection agree with the real HF checkpoint
# structure, and that TTPhi1Model/TTPhi1DecoderLayer's key-resolution candidates
# (models/demos/phi1/tt/phi1_model.py) actually resolve against it.

import pytest
import torch
from transformers import AutoModelForCausalLM

from models.demos.phi1.demo.demo import detect_base_address, truncate_hf_layers

MODEL_ID = "microsoft/phi-1"
NUM_HIDDEN_LAYERS = 24


@pytest.fixture(scope="module")
def fresh_hf_model():
    def _load():
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)
        model.eval()
        return model

    return _load


def test_native_hf_phi_checkpoint_has_no_mixformer_container(fresh_hf_model):
    """microsoft/phi-1's current checkpoint loads as native transformers PhiForCausalLM,
    not the legacy MixFormerSequentialForCausalLM single-`layers`-container format."""
    model = fresh_hf_model()
    assert type(model).__name__ == "PhiForCausalLM"
    assert hasattr(model, "model") and hasattr(model.model, "layers")
    assert len(model.model.layers) == NUM_HIDDEN_LAYERS

    state_dict = model.state_dict()
    assert detect_base_address(state_dict) == "model"
    assert "model.embed_tokens.weight" in state_dict
    assert "model.final_layernorm.weight" in state_dict
    assert "lm_head.weight" in state_dict


@pytest.mark.parametrize("num_layers", [1, 5, NUM_HIDDEN_LAYERS])
def test_truncation_preserves_embedding_and_head(fresh_hf_model, num_layers):
    """Truncating to `num_layers` must keep the embedding, final norm, and LM head
    byte-for-byte, keep exactly `num_layers` decoder blocks, and keep decoder block
    identity (layer i after truncation == layer i before truncation) — not silently
    substitute the embedding or head in as a "decoder layer"."""
    reference = fresh_hf_model()
    ref_embed = reference.model.embed_tokens.weight.clone()
    ref_final_norm_w = reference.model.final_layernorm.weight.clone()
    ref_final_norm_b = reference.model.final_layernorm.bias.clone()
    ref_lm_head = reference.lm_head.weight.clone()
    ref_layer_weights = [layer.input_layernorm.weight.clone() for layer in reference.model.layers[:num_layers]]

    model = fresh_hf_model()
    truncate_hf_layers(model, num_layers)

    assert len(model.model.layers) == num_layers, "truncation did not produce the requested layer count"

    assert torch.equal(model.model.embed_tokens.weight, ref_embed), "truncation altered the token embedding"
    assert torch.equal(model.model.final_layernorm.weight, ref_final_norm_w)
    assert torch.equal(model.model.final_layernorm.bias, ref_final_norm_b)
    assert torch.equal(model.lm_head.weight, ref_lm_head), "truncation altered the LM head"

    for i in range(num_layers):
        assert torch.equal(
            model.model.layers[i].input_layernorm.weight, ref_layer_weights[i]
        ), f"decoder layer {i} identity changed after truncation"

    # layer 0 must be an actual decoder block, never the embedding or head reinterpreted as one.
    assert model.model.layers[0].input_layernorm.weight.shape == (2048,)
    assert not torch.equal(model.model.layers[0].input_layernorm.weight, ref_embed[0])


@pytest.mark.parametrize("num_layers", [1, 5, NUM_HIDDEN_LAYERS])
def test_tt_key_resolution_candidates_match_truncated_checkpoint(fresh_hf_model, num_layers):
    """Exercises the exact fallback key-resolution chains TTPhi1Model /
    TTPhi1DecoderLayer use (phi1_model.py) against a truncated state_dict, without
    ever constructing ttnn tensors or opening a device."""
    model = fresh_hf_model()
    truncate_hf_layers(model, num_layers)
    state_dict = model.state_dict()
    base_address = detect_base_address(state_dict)
    assert base_address == "model"

    # TTPhi1Model's embedding candidates (phi1_model.py ~line 629)
    assert f"{base_address}.embed_tokens.weight" in state_dict

    # TTPhi1Model's final-norm candidates (phi1_model.py ~line 679)
    assert f"{base_address}.final_layernorm.weight" in state_dict
    assert f"{base_address}.final_layernorm.bias" in state_dict

    # TTPhi1ForCausalLM's lm_head candidates (phi1_model.py ~line 818)
    assert "lm_head.weight" in state_dict

    # TTPhi1DecoderLayer's per-layer address + input_layernorm candidates
    # (phi1_model.py ~line 517-543) must resolve for every kept layer index,
    # and must NOT resolve for any index beyond the truncated count.
    for i in range(num_layers):
        assert f"{base_address}.layers.{i}.input_layernorm.weight" in state_dict
        assert f"{base_address}.layers.{i}.self_attn.q_proj.weight" in state_dict

    if num_layers < NUM_HIDDEN_LAYERS:
        assert f"{base_address}.layers.{num_layers}.input_layernorm.weight" not in state_dict
