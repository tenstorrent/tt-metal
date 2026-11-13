# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Model providers: the only place that decides whether a model can be trained with a config.

Bringing up a model family means adding its provider to ``PROVIDERS``, keyed by the ``model_type`` of the
HF ``config.json``; the trainer and its hosts don't change.
"""

from __future__ import annotations

from typing import Any

from ..configs import TrainerConfig
from .base import AdapterState, ModelProvider, UnsupportedTrainingConfig
from .hub import load_hf_config, model_dir
from .llama import LlamaProvider

PROVIDERS: dict[str, ModelProvider] = {"llama": LlamaProvider()}


def get_provider(config: TrainerConfig) -> tuple[ModelProvider, dict[str, Any]]:
    """The provider for ``config.model.name``, and the model's HF config."""
    hf_config = load_hf_config(config.model.name)
    model_type = hf_config.get("model_type")
    provider = PROVIDERS.get(model_type)
    if provider is None:
        raise UnsupportedTrainingConfig(
            f"model.name: no provider for model_type {model_type!r} ({config.model.name}); "
            f"available: {sorted(PROVIDERS)}"
        )
    return provider, hf_config


__all__ = [
    "AdapterState",
    "LlamaProvider",
    "ModelProvider",
    "PROVIDERS",
    "UnsupportedTrainingConfig",
    "get_provider",
    "load_hf_config",
    "model_dir",
]
