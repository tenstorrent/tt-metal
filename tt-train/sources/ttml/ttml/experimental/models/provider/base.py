# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..utils.checkpoint import PathLike

if TYPE_CHECKING:
    from ttml.experimental.trainer.configs import TrainerConfig


class ModelProvider(ABC):
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def model_source(self) -> str:
        """Where the pretrained model's files are: an HF hub id, or a local directory laid out as ``save_pretrained``
        writes one. ``config.model.name``, the hub id, by default; a provider whose model isn't on the hub returns
        where it is."""
        return self.config.model.name

    @abstractmethod
    def check(self) -> None:
        """Raise ``UnsupportedTrainingConfig`` if the model can't be built and trained as ``config`` describes."""

    @abstractmethod
    def build(self) -> Any:
        """Build the model and its optimizations, with the pretrained weights at ``model_source()``."""

    @abstractmethod
    def load_weights(self, model: Any, directory: PathLike) -> None:
        """Load weights into ``model``."""

    @abstractmethod
    def export_weights(self, model: Any, directory: PathLike) -> None:
        """Write the weights of ``model`` to the existing ``directory``."""
