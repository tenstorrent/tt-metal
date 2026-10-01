# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Which Laguna checkpoint this checkout serves.

Laguna-S-2.1 and Laguna-XS-2.1 share one HF modeling file; only config sizes differ. The
TTNN code in this directory reads every size from the HF config, so the checkpoint is a
runtime choice. ``TT_LAGUNA_MODEL`` selects it (default: Laguna-S-2.1). Everything that
names a checkpoint, a reference, or an on-disk weight cache must go through this module so
two checkpoints can never read each other's converted tensors.
"""
from __future__ import annotations

import os
from typing import Mapping

MODEL_ENV = "TT_LAGUNA_MODEL"
DEFAULT_MODEL_ID = "poolside/Laguna-S-2.1"

# HF repo id -> cache-safe slug. The slug names the converted-weight cache directory.
SUPPORTED_MODELS: Mapping[str, str] = {
    "poolside/Laguna-S-2.1": "laguna_s_2_1",
    "poolside/Laguna-XS-2.1": "laguna_xs_2_1",
}

# HF ``max_position_embeddings`` per checkpoint: the RoPE horizon the published config declares.
# Kept here (not read from the config at import) so serving limits are known without transformers;
# tests/test_model_spec.py checks these against the downloaded config.
MAX_POSITION_EMBEDDINGS: Mapping[str, int] = {
    "poolside/Laguna-S-2.1": 1048576,
    "poolside/Laguna-XS-2.1": 262144,
}


def resolve_model_id(environ: Mapping[str, str] | None = None) -> str:
    """Return the selected HF repo id, rejecting anything this port was not built for."""
    env = os.environ if environ is None else environ
    model_id = env.get(MODEL_ENV, DEFAULT_MODEL_ID).strip()
    if model_id not in SUPPORTED_MODELS:
        choices = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"{MODEL_ENV}={model_id!r} is not a supported Laguna checkpoint; expected one of: {choices}")
    return model_id


def model_slug(model_id: str) -> str:
    """Cache-safe name for ``model_id`` (raises for unsupported ids)."""
    try:
        return SUPPORTED_MODELS[model_id]
    except KeyError as exc:
        raise ValueError(f"unsupported Laguna checkpoint {model_id!r}") from exc


MODEL_ID = resolve_model_id()
MODEL_SLUG = model_slug(MODEL_ID)
MODEL_MAX_CONTEXT = MAX_POSITION_EMBEDDINGS[MODEL_ID]
