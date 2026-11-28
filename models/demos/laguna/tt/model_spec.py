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
from dataclasses import dataclass
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


@dataclass(frozen=True)
class DFlashSpec:
    """The published DFlash draft (speculator) checkpoint paired with one target checkpoint.

    Every value here is what Poolside's ``config.json`` at ``revision`` declares;
    ``tt/dflash_reference.py`` validates a downloaded config against it, and
    ``tests/test_dflash_reference.py`` checks it against the downloaded file. The draft owns no
    embedding or LM head: it reads the target's post-layer hidden states at ``target_layer_ids``
    (0-based decoder layers; ``aux_hidden_state_layer_ids`` is the same list in the +1
    "hidden_states[i]" convention) and uses the target's embedding table and LM head.
    """

    repo_id: str
    revision: str
    num_draft_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    sliding_window: int
    block_size: int
    mask_token_id: int
    num_target_layers: int
    target_layer_ids: tuple[int, ...]
    aux_hidden_state_layer_ids: tuple[int, ...]
    # The single served topology (1×D mesh) DFlash is scoped to for this checkpoint: the profile
    # the target streams its prefill on. XS: p150x2 (D=2, qualified); S: p150x4 (D=4, the only
    # profile that holds S).
    serving_device_count: int
    serving_profile: str

    @property
    def geometry(self) -> tuple[int, ...]:
        """Ordered like ``LagunaDFlashConfig.validate``'s published-geometry tuple."""
        return (
            self.hidden_size,
            self.intermediate_size,
            self.num_draft_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
            self.max_position_embeddings,
            self.sliding_window,
            self.block_size,
            self.mask_token_id,
        )


# Target HF repo id -> its published DFlash draft.
DFLASH_MODELS: Mapping[str, DFlashSpec] = {
    "poolside/Laguna-S-2.1": DFlashSpec(
        repo_id="poolside/Laguna-S-2.1-DFlash",
        revision="1334981872c58f5023614307b4536f8b23c262e5",
        num_draft_layers=6,
        hidden_size=3072,
        intermediate_size=12288,
        num_attention_heads=72,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=100352,
        max_position_embeddings=1048576,
        sliding_window=512,
        block_size=16,
        mask_token_id=12,
        num_target_layers=48,
        target_layer_ids=(1, 10, 19, 29, 38, 47),
        aux_hidden_state_layer_ids=(2, 11, 20, 30, 39, 48),
        serving_device_count=4,
        serving_profile="p150x4",
    ),
    "poolside/Laguna-XS-2.1": DFlashSpec(
        repo_id="poolside/Laguna-XS-2.1-DFlash",
        revision="5c36361aab23c8ed3afbd079c10c426b677bc607",
        num_draft_layers=5,
        hidden_size=2048,
        intermediate_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=100352,
        max_position_embeddings=262144,
        sliding_window=512,
        block_size=16,
        mask_token_id=12,
        num_target_layers=40,
        target_layer_ids=(1, 13, 25, 33, 39),
        aux_hidden_state_layer_ids=(2, 14, 26, 34, 40),
        serving_device_count=2,
        serving_profile="p150x2",
    ),
}


def dflash_spec(model_id: str | None = None) -> DFlashSpec:
    """The DFlash draft published for ``model_id`` (default: the selected checkpoint)."""
    model_id = model_id or MODEL_ID
    try:
        return DFLASH_MODELS[model_id]
    except KeyError as exc:
        raise ValueError(f"no published DFlash draft is registered for {model_id!r}") from exc


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
DFLASH_SPEC = dflash_spec(MODEL_ID)
