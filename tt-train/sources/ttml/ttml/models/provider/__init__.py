# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Model providers: build HF models in ttml, and load and export their weights.

Bringing up a model means subclassing ``ModelProvider`` (``base.py``) in ``ttml/models/provider/<family>.py``, or
reusing its family's, and adding the model to ``providers``, keyed by its HF hub id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils.errors import UnsupportedTrainingConfig
from ..utils.hub import same_model
from .base import ModelProvider
from .llama import LlamaProvider

if TYPE_CHECKING:
    from ttml.experimental.trainer.configs import TrainerConfig


def providers() -> dict[str, type[ModelProvider]]:
    """The provider of each supported model, keyed by its HF hub id."""
    return {
        "meta-llama/Llama-3.2-1B": LlamaProvider,
        "meta-llama/Llama-3.2-1B-Instruct": LlamaProvider,
        "meta-llama/Llama-3.1-8B": LlamaProvider,
        "meta-llama/Llama-3.1-8B-Instruct": LlamaProvider,
    }


def get_provider(config: TrainerConfig) -> ModelProvider:
    """The provider of ``config.model.name``: a hub id, or a local copy of a hub model, named as its directory."""
    name = config.model.name
    provider = next((provider for model, provider in providers().items() if same_model(model, name)), None)
    if provider is None:
        raise UnsupportedTrainingConfig(f"model.name: {name!r} is not one of {sorted(providers())}")
    return provider(config)


__all__ = [
    "LlamaProvider",
    "ModelProvider",
    "UnsupportedTrainingConfig",
    "get_provider",
    "providers",
]
