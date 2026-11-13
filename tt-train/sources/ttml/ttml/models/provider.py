# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Model providers: the only place that decides whether a model can be trained with a config.

A provider builds one family of HF models in ttml for ``ttml.unified_trainer``. Bringing up a family means writing its
provider in ``ttml/models/<family>/provider.py`` and adding it to ``providers``, keyed by the ``model_type`` of the HF
``config.json``; the trainer and its hosts don't change.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from ttml.unified_trainer.configs import TrainerConfig

# What a training run reads from a hub model: the weights, the configs and the tokenizer.
_MODEL_FILES = ["*.safetensors", "*.json", "tokenizer.model", "chat_template.jinja"]


class UnsupportedTrainingConfig(ValueError):
    """This model can't be trained with this config. The message names the setting and says why."""


@dataclass
class AdapterState:
    """A LoRA adapter in PEFT's form: tensors by PEFT name, and the contents of ``adapter_config.json``."""

    tensors: dict[str, np.ndarray]
    config: dict[str, Any]


class ModelProvider(Protocol):
    """Builds one family of HF models in ttml, and maps the LoRA adapters a run trains to and from PEFT's format.

    The only place that decides whether a model can train a config: the trainer and its hosts never inspect
    LoRA targets, dtypes, sequence lengths or the device mesh. Every config a provider gets has ``model.lora``
    set, as the trainer rejects a run without it.
    """

    def check(self, config: TrainerConfig, hf_config: dict[str, Any]) -> None:
        """Raise ``UnsupportedTrainingConfig`` if ``config`` can't be trained on this model.

        Runs before any weights are downloaded and before the device is opened.
        """

    def build(self, config: TrainerConfig, hf_config: dict[str, Any]) -> Any:
        """Load the pretrained model, wrapped in ``ttml.modules.LoraModel`` as ``config.model.lora`` describes.

        The model takes the standard ttml causal-LM input the trainer assumes:
        ``logits = model(input_ids, None)``, with ``input_ids`` ``[B, 1, 1, T]`` uint32, ``logits``
        ``[B, 1, T, V]``, and ``None`` for a causal mask.
        """

    def export_adapter(self, model: Any, config: TrainerConfig) -> AdapterState:
        """The LoRA adapters of ``model`` as a PEFT adapter of the HF model."""

    def import_adapter(self, model: Any, state: AdapterState, config: TrainerConfig) -> None:
        """Load a PEFT adapter into ``model``; raise ``UnsupportedTrainingConfig`` if it doesn't fit."""


@cache
def providers() -> dict[str, ModelProvider]:
    """The provider of each model family, keyed by the ``model_type`` of the HF ``config.json``."""
    # Imported here, not with the module: every provider imports its contract from this module.
    from .llama.provider import LlamaProvider

    return {"llama": LlamaProvider()}


def get_provider(config: TrainerConfig) -> tuple[ModelProvider, dict[str, Any]]:
    """The provider for ``config.model.name``, and the model's HF config."""
    hf_config = load_hf_config(config.model.name)
    model_type = hf_config.get("model_type")
    provider = providers().get(model_type)
    if provider is None:
        raise UnsupportedTrainingConfig(
            f"model.name: no provider for model_type {model_type!r} ({config.model.name}); "
            f"available: {sorted(providers())}"
        )
    return provider, hf_config


def load_hf_config(model_name: str) -> dict[str, Any]:
    """The ``config.json`` of a local model directory, or of a hub model (downloads only that file)."""
    path = Path(model_name) / "config.json"
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(model_name, "config.json"))
    return json.loads(path.read_text())


def model_dir(model_name: str) -> str:
    """A local directory with the model's weights, configs and tokenizer; downloads a hub model once."""
    if Path(model_name).is_dir():
        return model_name
    from huggingface_hub import snapshot_download

    return snapshot_download(model_name, allow_patterns=_MODEL_FILES)


__all__ = [
    "AdapterState",
    "ModelProvider",
    "UnsupportedTrainingConfig",
    "get_provider",
    "load_hf_config",
    "model_dir",
    "providers",
]
