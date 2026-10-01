# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free checks of the checkpoint selector (``tt/model_spec.py``)."""
from __future__ import annotations

import json

import pytest

from models.autoports.poolside_laguna_xs_2_1.tt import model_spec as M


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
