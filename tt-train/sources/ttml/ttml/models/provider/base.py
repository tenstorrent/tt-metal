# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..utils.checkpoint import PathLike

if TYPE_CHECKING:
    from ttml.unified_trainer.configs import TrainerConfig


class ModelProvider(ABC):
    """One family of HF models in ttml, for one config: whether it supports the config, the model, and its weights.

    Used as::

        provider = get_provider(config)
        provider.check()
        model = provider.build()
        provider.load_weights(model, directory)  # optional: weights a previous export_weights wrote
        ...
        provider.export_weights(model, directory)

    Which weights ``export_weights`` writes, and how it lays them out, is up to the provider; ``load_weights`` reads
    them back.
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    @abstractmethod
    def check(self) -> None:
        """Raise ``UnsupportedTrainingConfig`` if the model can't be built and trained as ``config`` describes.

        Must not download weights or open the device.
        """

    @abstractmethod
    def build(self) -> Any:
        """The model, with the pretrained weights of ``config.model.name``.

        A causal LM: ``logits = model(input_ids, None)``, with ``input_ids`` ``[B, 1, 1, T]`` uint32, ``logits``
        ``[B, 1, T, V]``, and ``None`` for a causal mask.
        """

    @abstractmethod
    def load_weights(self, model: Any, directory: PathLike) -> None:
        """Load into ``model``, which ``build`` made, the weights ``export_weights`` wrote to ``directory``.

        Raise ``UnsupportedTrainingConfig`` if they are not of this model.
        """

    @abstractmethod
    def export_weights(self, model: Any, directory: PathLike) -> None:
        """Read the weights of ``model`` from the device and write them to the existing ``directory``."""
