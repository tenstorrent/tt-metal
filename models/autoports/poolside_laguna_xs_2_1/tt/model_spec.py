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

# (num_hidden_layers, hidden_size) per checkpoint: enough to tell the two apart from any HF config.
CHECKPOINT_SHAPES: Mapping[str, tuple[int, int]] = {
    "poolside/Laguna-S-2.1": (48, 3072),
    "poolside/Laguna-XS-2.1": (40, 2048),
}


def check_hf_config(hf_config, model_id: str | None = None) -> None:
    """Fail fast when a caller's HF config is not the checkpoint ``TT_LAGUNA_MODEL`` selects.

    vLLM loads its config from the served model name, while the TTNN code reads weights, references and
    caches for ``MODEL_ID``. If they disagree (e.g. an XS server without TT_LAGUNA_MODEL, now that S is
    the default) the model would be built from the wrong checkpoint."""
    model_id = model_id or MODEL_ID
    got = (int(hf_config.num_hidden_layers), int(hf_config.hidden_size))
    want = CHECKPOINT_SHAPES[model_id]
    if got != want:
        match = [mid for mid, shape in CHECKPOINT_SHAPES.items() if shape == got]
        hint = f"; it looks like {match[0]}, so set {MODEL_ENV}={match[0]}" if match else ""
        raise ValueError(
            f"{MODEL_ENV} selects {model_id} ({want[0]} layers, hidden {want[1]}) but the served config has "
            f"{got[0]} layers, hidden {got[1]}{hint}"
        )


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
