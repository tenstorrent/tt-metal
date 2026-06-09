# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 — Weight loading smoke test.

Verifies that all expected key prefixes exist in the loaded state dict and
that tensor shapes match the architecture defined in vibevoice_config.py.
"""


import pytest

from models.experimental.vibevoice.common.config import MODEL_PATH
from models.experimental.vibevoice.tt.load_weights import (
    load_vibevoice_state_dict,
    split_submodule_weights,
    remap_lm_keys_to_tt_transformers,
)
from models.experimental.vibevoice.tt.vibevoice_config import load_vibevoice_model_config


@pytest.fixture(scope="module")
def state_dict():
    return load_vibevoice_state_dict(MODEL_PATH)


@pytest.fixture(scope="module")
def submodules(state_dict):
    return split_submodule_weights(state_dict)


@pytest.fixture(scope="module")
def vv_config():
    return load_vibevoice_model_config(MODEL_PATH)


def test_state_dict_not_empty(state_dict):
    assert len(state_dict) > 0, "state dict is empty"


def test_expected_prefixes_present(state_dict):
    expected_prefixes = [
        "model.language_model.",
        "model.acoustic_connector.",
        "model.semantic_connector.",
        "model.prediction_head.",
        "model.acoustic_tokenizer.",
        "model.semantic_tokenizer.",
    ]
    for prefix in expected_prefixes:
        matching = [k for k in state_dict if k.startswith(prefix)]
        assert len(matching) > 0, f"No keys found with prefix {prefix!r}"


def test_submodule_split(submodules):
    required_submodules = [
        "lm",
        "acoustic_connector",
        "semantic_connector",
        "diffusion_head",
        "acoustic_tokenizer",
        "semantic_tokenizer",
    ]
    for name in required_submodules:
        assert name in submodules, f"Submodule {name!r} missing from split"
        assert len(submodules[name]) > 0, f"Submodule {name!r} is empty"
    # lm_head uses tied weights (embed_tokens) — no separate keys expected
    assert "lm_head" in submodules, "lm_head key missing from split result"


def test_lm_layer_count(submodules, vv_config):
    lm = submodules["lm"]
    n_layers = vv_config.decoder.num_hidden_layers
    for layer_idx in range(n_layers):
        key = f"layers.{layer_idx}.self_attn.q_proj.weight"
        assert key in lm, f"Missing LM layer key: {key}"


def test_lm_hidden_size(submodules, vv_config):
    lm = submodules["lm"]
    hidden = vv_config.decoder.hidden_size
    embed_weight = lm.get("embed_tokens.weight")
    assert embed_weight is not None, "embed_tokens.weight missing from LM weights"
    assert embed_weight.shape[1] == hidden, f"embed dim {embed_weight.shape[1]} != config hidden_size {hidden}"


def test_connector_shapes(submodules, vv_config):
    ac = submodules["acoustic_connector"]
    sc = submodules["semantic_connector"]
    hidden = vv_config.connector_output_dim

    # fc1 projects input_dim → hidden; fc2 projects hidden → hidden
    assert "fc1.weight" in ac, "acoustic_connector fc1.weight missing"
    assert "fc2.weight" in ac, "acoustic_connector fc2.weight missing"
    assert ac["fc2.weight"].shape == (
        hidden,
        hidden,
    ), f"acoustic_connector fc2 shape {ac['fc2.weight'].shape} != ({hidden},{hidden})"

    assert "fc1.weight" in sc, "semantic_connector fc1.weight missing"
    assert "fc2.weight" in sc, "semantic_connector fc2.weight missing"
    assert sc["fc2.weight"].shape == (
        hidden,
        hidden,
    ), f"semantic_connector fc2 shape {sc['fc2.weight'].shape} != ({hidden},{hidden})"


def test_diffusion_head_keys(submodules):
    dh = submodules["diffusion_head"]
    expected = [
        "noisy_images_proj.weight",
        "cond_proj.weight",
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.2.weight",
        "layers.0.ffn.gate_proj.weight",
        "final_layer.linear.weight",
    ]
    for key in expected:
        assert key in dh, f"diffusion_head missing key: {key}"


def test_lm_key_remap(submodules):
    remapped = remap_lm_keys_to_tt_transformers(submodules["lm"])
    assert "tok_embeddings.weight" in remapped, "tok_embeddings.weight missing after remap"
    assert "layers.0.attention.wq.weight" in remapped, "wq remap failed"
    assert "layers.0.feed_forward.w1.weight" in remapped, "w1 remap failed"
    assert "norm.weight" in remapped, "norm.weight missing after remap"
