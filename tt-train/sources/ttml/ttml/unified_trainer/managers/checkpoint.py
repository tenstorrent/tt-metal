# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Callable, Optional

from ..configs import CheckpointConfig

HISTORY_FILE = "checkpoint_history.json"


class CheckpointManager:
    """When a run saves checkpoints, and which of them it keeps.

    Owns ``<project_dir>/checkpoints/``: the history file and the rotation ``rotation_strategy`` picks. Writing a
    checkpoint directory is delegated to ``save(path, metadata, optimizer=...)``, which
    ``Trainer.create_checkpoint_manager`` binds to the trainer's ``save_checkpoint``.
    """

    def __init__(self, config: CheckpointConfig, save: Callable[..., None], logger: Any = None) -> None:
        self.config = config
        self._save = save
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.checkpoint_dir = os.path.join(config.project_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.history_file = os.path.join(self.checkpoint_dir, HISTORY_FILE)
        # A run only rotates away the checkpoints it saved itself: an earlier run's in the same project_dir, such as
        # the one a resumed run started from, are never deleted.
        if os.path.exists(self.history_file):
            self.logger.warning(f"Starting a new checkpoint history; {self.history_file} is replaced on the first save")
        # The checkpoints of this run that rotation kept, oldest first.
        self.history: list[dict] = []

    def should_save_checkpoint(self, step: int, epoch: Optional[int] = None) -> bool:
        """At the end of an epoch with ``epoch`` set, else after optimizer step ``step``."""
        strategy = self.config.save_strategy
        if epoch is not None:
            return strategy.type == "epoch" and (epoch + 1) % strategy.frequency == 0
        return strategy.type == "step" and step % strategy.frequency == 0

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        metrics: Optional[dict[str, float]] = None,
        name: Optional[str] = None,
    ) -> str:
        """Save a checkpoint named ``name`` (by default from the step, epoch and time); return its path."""
        metrics = metrics or {}
        rotation = self.config.rotation_strategy
        if rotation.type == "best" and rotation.metric not in metrics:
            raise ValueError(
                f"checkpoint.rotation_strategy: best ranks checkpoints by {rotation.metric}, and the one at step "
                f"{step} has only {sorted(metrics)}"
            )
        timestamp = datetime.now()
        if name is None:
            name = f"checkpoint_step{step}_epoch{epoch}_{timestamp:%Y%m%d_%H%M%S}"
        path = os.path.join(self.checkpoint_dir, name)
        metadata = {"step": step, "epoch": epoch, "metrics": metrics, "timestamp": timestamp.isoformat()}

        self._save(path, metadata, optimizer=self.config.save_optim)

        # Saving under a name that is already tracked overwrote that checkpoint.
        self.history = [entry for entry in self.history if entry["path"] != path]
        self.history.append({"path": path, "name": name, **metadata})
        self._rotate()
        with open(self.history_file, "w") as f:
            json.dump({"checkpoints": self.history}, f, indent=2)

        self.logger.info(f"Saved checkpoint: {path}")
        return path

    def _rotate(self) -> None:
        """Delete the checkpoints ``rotation_strategy`` doesn't keep. The latest one is always kept."""
        rotation = self.config.rotation_strategy
        if rotation.type == "none":
            return
        if rotation.type == "last":
            kept = self.history[-rotation.keep_n :]
        else:
            # A stable sort: of equally good checkpoints, the earlier ones rank first.
            ranked = sorted(
                self.history, key=lambda entry: entry["metrics"][rotation.metric], reverse=rotation.mode == "max"
            )
            kept = ranked[: rotation.keep_n] + self.history[-1:]
        kept_paths = {entry["path"] for entry in kept}
        for entry in self.history:
            if entry["path"] not in kept_paths:
                self._remove(entry["path"])
        self.history = [entry for entry in self.history if entry["path"] in kept_paths]

    def _remove(self, path: str) -> None:
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.exists(path):
            os.remove(path)
        else:
            return
        self.logger.info(f"Removed old checkpoint: {path}")
