# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The contract every model provider implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from ..configs import TrainerConfig


class UnsupportedTrainingConfig(ValueError):
    """This model can't be trained with this config. The message names the setting and says why."""


@dataclass
class AdapterState:
    """A LoRA adapter in PEFT's form: tensors by PEFT name, and the contents of ``adapter_config.json``."""

    tensors: dict[str, np.ndarray]
    config: dict[str, Any]


class ModelProvider(Protocol):
    """Builds one family of HF models in ttml, and maps what a run trains to and from HF's formats.

    The only place that decides whether a config can be trained: the trainer and its hosts never inspect
    LoRA targets, dtypes or sequence lengths.
    """

    def check(self, config: TrainerConfig, hf_config: dict[str, Any]) -> None:
        """Raise ``UnsupportedTrainingConfig`` if ``config`` can't be trained on this model.

        Runs before any weights are downloaded and before the device is opened.
        """

    def build(self, config: TrainerConfig, hf_config: dict[str, Any]) -> Any:
        """Load the pretrained model, wrapped in ``ttml.modules.LoraModel`` when ``config.model.lora`` is set.

        The model takes the standard ttml causal-LM input the trainer assumes:
        ``logits = model(input_ids, None)``, with ``input_ids`` ``[B, 1, 1, T]`` uint32, ``logits``
        ``[B, 1, T, V]``, and ``None`` for a causal mask.
        """

    def export_adapter(self, model: Any, config: TrainerConfig) -> AdapterState:
        """The LoRA adapters of ``model`` as a PEFT adapter of the HF model."""

    def import_adapter(self, model: Any, state: AdapterState, config: TrainerConfig) -> None:
        """Load a PEFT adapter into ``model``; raise ``UnsupportedTrainingConfig`` if it doesn't fit."""

    def export_weights(self, model: Any, config: TrainerConfig, directory: str) -> None:
        """Write all weights of ``model`` to ``directory`` as an HF model directory (``from_pretrained`` loads it)."""

    def import_weights(self, model: Any, directory: str, config: TrainerConfig) -> None:
        """Load all weights of ``model`` from an HF model directory."""
