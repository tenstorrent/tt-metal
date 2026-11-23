# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from ..utils.checkpoint import PathLike, tt_metal_version

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
    def export_weights(self, model: Any, directory: PathLike, metadata: Optional[dict[str, Any]] = None) -> None:
        """Write the weights of ``model`` and ``trainer_state.json`` to the existing ``directory``.

        ``metadata`` is what the caller of ``Trainer.save_checkpoint`` passes: the step, epoch and metrics at the
        save. ``trainer_state.json`` holds ``trainer_state(metadata)`` and what the provider records about the model
        for ``load_weights`` to read back, written with ``write_trainer_state``. For example::

            {
                "format": {"type": "hf"},
                "config": {...},
                "tt_metal_version": "0.64.0",
                "step": 100,
                "epoch": 0,
                "metrics": {"train/loss": 1.23, "val/loss": 1.31},
                "base_model": {"name_or_path": "meta-llama/Llama-3.2-1B-Instruct"}
            }
        """

    def trainer_state(self, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """What every checkpoint records about the run: the format, the run's full config and the tt-metal version
        (``tt_metal_version()``), then ``metadata``, whose keys win."""
        return {
            "format": self.config.checkpoint.format.model_dump(),
            "config": self.config.model_dump(mode="json"),
            "tt_metal_version": tt_metal_version(),
            **(metadata or {}),
        }
