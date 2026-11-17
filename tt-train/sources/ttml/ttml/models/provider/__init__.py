# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Model providers: build one family of HF models in ttml, and load and export its weights.

Bringing up a family means subclassing ``ModelProvider`` (``base.py``) in ``ttml/models/provider/<family>.py`` and
adding it to ``providers``, keyed by the ``model_type`` of the HF ``config.json``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils.errors import UnsupportedTrainingConfig
from ..utils.hub import load_hf_config
from .base import ModelProvider
from .llama import LlamaProvider

if TYPE_CHECKING:
    from ttml.unified_trainer.configs import TrainerConfig


def providers() -> dict[str, type[ModelProvider]]:
    """The provider of each model family, keyed by the ``model_type`` of the HF ``config.json``."""
    return {"llama": LlamaProvider}


def get_provider(config: TrainerConfig) -> ModelProvider:
    """The provider of the family of ``config.model.name``, for the run ``config`` describes."""
    model_type = load_hf_config(config.model.name).get("model_type")
    provider = providers().get(model_type)
    if provider is None:
        raise UnsupportedTrainingConfig(
            f"model.name: no provider for model_type {model_type!r} ({config.model.name}); "
            f"available: {sorted(providers())}"
        )
    return provider(config)


__all__ = [
    "LlamaProvider",
    "ModelProvider",
    "UnsupportedTrainingConfig",
    "get_provider",
    "providers",
]
