# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Drafter config + weights straight from the HF cache, with no ``transformers`` import.

``reference_dflash`` is the *correctness* reference and deliberately imports
``MuseGlimmerAssistantModel`` so the port is graded against real HF math.  That
makes it unusable for perf work on a machine whose ``transformers`` predates the
architecture -- ``transformers.models.muse_glimmer_assistant`` does not exist in
5.11.0, so importing ``reference_dflash`` raises at module scope and
``AutoConfig`` raises ``KeyError('muse_glimmer_assistant')``.

Perf work does not need HF: the drafter's weights are a plain safetensors file and
its config is a plain JSON document.  This module reads both directly, so a
benchmark or an old-vs-new PCC comparison runs anywhere the checkpoint is cached.

It is **not** a correctness reference and must not become one -- grading the port
against a config this module parsed would only prove the parser agrees with
itself.  ``reference_dflash`` remains the authority on HF math; the pinned-field
check below exists only so a checkpoint swap fails loudly here too.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch

DRAFT_MODEL_ID = "meta-models/Muse-Glimmer-30B-assistant"

#: Mirrors ``reference_dflash.EXPECTED_CONFIG``.  Duplicated rather than imported
#: because importing that module is exactly what this one exists to avoid.
EXPECTED_CONFIG = {
    "hidden_size": 6656,
    "intermediate_size": 19968,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_hidden_layers": 5,
    "sliding_window": 2048,
    "block_size": 16,
    "target_layer_ids": [1, 13, 25, 37, 49],
    "max_position_embeddings": 131072,
}


def snapshot_dir(model_id: str = DRAFT_MODEL_ID) -> Path:
    """Cache snapshot holding the weights.

    Located by finding the weight file rather than by trusting ``refs/main``, for
    the reason the target's loader gives: a repo's default revision can be
    metadata-only.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    repo = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    candidates = sorted(repo.glob("snapshots/*/model.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"no cached weights for {model_id} under {repo}")
    return candidates[0].parent


@lru_cache(maxsize=1)
def draft_config() -> SimpleNamespace:
    """The drafter config as a duck-typed stand-in for the HF config object.

    ``config_from_hf`` reads plain attributes plus ``rope_parameters["rope_theta"]``,
    so a namespace over the JSON is a faithful substitute for its purposes.
    """
    raw = json.loads((snapshot_dir() / "config.json").read_text())
    for key, expected in EXPECTED_CONFIG.items():
        actual = raw[key]
        if isinstance(expected, list):
            actual = list(actual)
        if actual != expected:
            raise AssertionError(f"drafter config drifted: {key} is {actual!r}, port assumes {expected!r}")
    if raw["rope_parameters"]["rope_theta"] != 500000.0:
        raise AssertionError(f"drafter rope_theta drifted: {raw['rope_parameters']['rope_theta']}")
    return SimpleNamespace(**raw)


@lru_cache(maxsize=1)
def draft_state_dict() -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    path = snapshot_dir() / "model.safetensors"
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as handle:
        for key in handle.keys():
            state_dict[key] = handle.get_tensor(key)
    for banned in ("embed_tokens", "lm_head"):
        if any(banned in key for key in state_dict):
            raise AssertionError(f"drafter unexpectedly ships {banned}; the port assumes it reuses the target's")
    return state_dict


def synthetic_inputs(*, context_len: int, seed: int = 20260816, dtype: torch.dtype = torch.bfloat16) -> dict:
    """Byte-identical to ``reference_dflash.synthetic_inputs`` so goldens stay comparable."""
    config = draft_config()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    fan_in = len(config.target_layer_ids) * config.hidden_size
    return {
        "noise_embeds": torch.normal(
            0.0, 0.02, (1, config.block_size, config.hidden_size), generator=generator, dtype=torch.float32
        ).to(dtype),
        "context_hidden_states": torch.normal(
            0.0, 1.0, (1, context_len, fan_in), generator=generator, dtype=torch.float32
        ).to(dtype),
    }
