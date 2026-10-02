# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free checks of the checkpoint selector (``tt/model_spec.py``)."""
from __future__ import annotations

import json

import pytest

from models.demos.laguna.tt import model_spec as M


def test_default_is_s():
    assert M.resolve_model_id({}) == "poolside/Laguna-S-2.1"


def test_env_selects_xs():
    assert M.resolve_model_id({M.MODEL_ENV: "poolside/Laguna-XS-2.1"}) == "poolside/Laguna-XS-2.1"


def test_rejects_unknown_checkpoint():
    with pytest.raises(ValueError, match="not a supported Laguna checkpoint"):
        M.resolve_model_id({M.MODEL_ENV: "poolside/Laguna-M-2.1"})


def test_slugs_are_distinct():
    # Distinct slugs keep the converted-weight caches of the two checkpoints apart.
    assert len(set(M.SUPPORTED_MODELS.values())) == len(M.SUPPORTED_MODELS)
    assert set(M.MAX_POSITION_EMBEDDINGS) == set(M.SUPPORTED_MODELS)


@pytest.mark.parametrize("model_id", sorted(M.SUPPORTED_MODELS))
def test_max_context_matches_downloaded_config(model_id):
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(model_id, "config.json")
    if not isinstance(path, str):
        pytest.skip(f"{model_id} config.json is not in the local HF cache")
    with open(path) as f:
        declared = json.load(f)["max_position_embeddings"]
    assert M.MAX_POSITION_EMBEDDINGS[model_id] == declared


def test_check_hf_config_accepts_the_selected_checkpoint_and_names_the_fix():
    from types import SimpleNamespace

    s_cfg = SimpleNamespace(num_hidden_layers=48, hidden_size=3072)
    xs_cfg = SimpleNamespace(num_hidden_layers=40, hidden_size=2048)
    M.check_hf_config(s_cfg, "poolside/Laguna-S-2.1")
    M.check_hf_config(xs_cfg, "poolside/Laguna-XS-2.1")
    with pytest.raises(ValueError, match="TT_LAGUNA_MODEL=poolside/Laguna-XS-2.1"):
        M.check_hf_config(xs_cfg, "poolside/Laguna-S-2.1")
    with pytest.raises(ValueError, match="served config has 7 layers"):
        M.check_hf_config(SimpleNamespace(num_hidden_layers=7, hidden_size=64), "poolside/Laguna-S-2.1")


@pytest.mark.parametrize("model_id", sorted(M.SUPPORTED_MODELS))
def test_checkpoint_shapes_match_downloaded_config(model_id):
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(model_id, "config.json")
    if not isinstance(path, str):
        pytest.skip(f"{model_id} config.json is not in the local HF cache")
    with open(path) as f:
        cfg = json.load(f)
    assert M.CHECKPOINT_SHAPES[model_id] == (cfg["num_hidden_layers"], cfg["hidden_size"])
