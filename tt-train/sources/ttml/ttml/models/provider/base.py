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

    @abstractmethod
    def check(self) -> None:
        """Raise ``UnsupportedTrainingConfig`` if the model can't be built and trained as ``config`` describes."""

    @abstractmethod
    def build(self) -> Any:
        """Build the model and its optimizations, with the pretrained weights of ``config.model.name``."""

    @abstractmethod
    def load_weights(self, model: Any, directory: PathLike) -> None:
        """Load weights into ``model``."""

    @abstractmethod
    def export_weights(self, model: Any, directory: PathLike) -> None:
        """Write the weights of ``model`` to the existing ``directory``."""
