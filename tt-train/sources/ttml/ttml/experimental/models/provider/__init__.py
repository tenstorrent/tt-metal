# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Model providers: build HF models in ttml, and load and export their weights.

``model.name`` is an HF hub id, and the provider registered for it builds the model; where the model's files are is
the provider's (``ModelProvider.model_source``). Bringing up a model means subclassing ``ModelProvider``
(``base.py``) in ``ttml/experimental/models/provider/<family>.py``, or reusing its family's, and registering it for
the model's hub id: below for the models ttml ships, or with ``register_provider`` from outside ttml.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...errors import UnsupportedTrainingConfig
from .base import ModelProvider
from .llama import LlamaProvider

if TYPE_CHECKING:
    from ttml.experimental.trainer.configs import TrainerConfig

# The provider of each model, keyed by its HF hub id.
_PROVIDERS: dict[str, type[ModelProvider]] = {}


def register_provider(hub_id: str, provider: type[ModelProvider]) -> None:
    """Build the model ``model.name: <hub_id>`` names with ``provider``. Raises ``ValueError`` if another provider is
    registered for ``hub_id``."""
    registered = _PROVIDERS.get(hub_id)
    if registered is not None and registered is not provider:
        raise ValueError(f"{hub_id!r} already has a provider, {registered.__name__}")
    _PROVIDERS[hub_id] = provider


for _hub_id in (
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
):
    register_provider(_hub_id, LlamaProvider)


def providers() -> dict[str, type[ModelProvider]]:
    """The provider of each registered model, keyed by its HF hub id."""
    return dict(_PROVIDERS)


def get_provider(config: TrainerConfig) -> ModelProvider:
    """The provider registered for ``config.model.name``."""
    name = config.model.name
    provider = _PROVIDERS.get(name)
    if provider is None:
        raise UnsupportedTrainingConfig(f"model.name: {name!r} is not one of {sorted(_PROVIDERS)}")
    return provider(config)


__all__ = [
    "LlamaProvider",
    "ModelProvider",
    "get_provider",
    "providers",
    "register_provider",
]
