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
    """Which checkpoints a run keeps, and which one it starts from.

    Owns ``<project_dir>/checkpoints/``: the history file, last-N / best-N rotation, and init-from-checkpoint.
    Writing and reading a checkpoint directory is delegated to ``save(path, metadata)`` and
    ``load(path) -> metadata``, which ``Trainer.create_checkpoint_manager`` binds to the trainer's
    ``save_checkpoint`` / ``load_checkpoint``.
    """

    def __init__(
        self,
        config: CheckpointConfig,
        save: Callable[[str, dict], None],
        load: Callable[[str], dict],
        logger: Any = None,
    ) -> None:
        if config.save_optim:
            raise NotImplementedError(
                "checkpoint.save_optim: checkpoints hold model weights only (there is no resume); set it to false"
            )
        self.config = config
        self._save = save
        self._load = load
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.checkpoint_dir = os.path.join(config.project_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.history_file = os.path.join(self.checkpoint_dir, HISTORY_FILE)
        # A run never inherits an earlier run's entries from the same project_dir: they would compete for
        # best-N, get rotated away by this run and mix their steps with this run's. The earlier history is
        # only read to find the "last" or "best" checkpoint to initialize from.
        if os.path.exists(self.history_file):
            self.logger.warning(f"Starting a new checkpoint history; {self.history_file} is replaced on the first save")
        self.history: dict[str, list] = {"checkpoints": [], "best_checkpoints": []}

    def should_save_checkpoint(self, step: int, epoch: Optional[int] = None) -> bool:
        """At the end of an epoch with ``epoch`` set, else after optimizer step ``step``."""
        if epoch is not None:
            return self.config.save_strategy == "epoch" and (epoch + 1) % self.config.epoch_freq == 0
        return self.config.save_strategy == "step" and step % self.config.steps_freq == 0

    def save_checkpoint(
        self,
        step: int,
        epoch: int,
        metrics: Optional[dict[str, float]] = None,
        name: Optional[str] = None,
    ) -> str:
        """Save a checkpoint named ``name`` (by default from the step, epoch and time); return its path."""
        metrics = metrics or {}
        timestamp = datetime.now()
        if name is None:
            name = f"checkpoint_step{step}_epoch{epoch}_{timestamp:%Y%m%d_%H%M%S}"
        path = os.path.join(self.checkpoint_dir, name)
        metadata = {"step": step, "epoch": epoch, "metrics": metrics, "timestamp": timestamp.isoformat()}

        self._save(path, metadata)

        entry = {"path": path, "name": name, **metadata}
        tracked_before = self._tracked_paths() | {path}
        # Saving under a name that is already tracked overwrote that checkpoint.
        self._drop_entries(path)
        self.history["checkpoints"].append(entry)
        if self.config.checkpoint_metric in metrics:
            self._update_best(entry)
        if not self.config.keep_all_checkpoints:
            self.history["checkpoints"] = self.history["checkpoints"][-self.config.keep_last_n :]
            self._remove_untracked(tracked_before)
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

        self.logger.info(f"Saved checkpoint: {path}")
        return path

    def _update_best(self, entry: dict) -> None:
        best = self.history["best_checkpoints"]
        best.append({**entry, "metric_value": entry["metrics"][self.config.checkpoint_metric]})
        best.sort(key=lambda e: e["metric_value"], reverse=self.config.checkpoint_metric_mode == "max")
        self.history["best_checkpoints"] = best[: self.config.keep_best_n]

    def _drop_entries(self, path: str) -> None:
        for key in ("checkpoints", "best_checkpoints"):
            self.history[key] = [e for e in self.history[key] if e["path"] != path]

    def _tracked_paths(self) -> set[str]:
        return {e["path"] for key in ("checkpoints", "best_checkpoints") for e in self.history[key]}

    def _remove_untracked(self, tracked_before: set[str]) -> None:
        """Delete checkpoints that were tracked before this save and are in neither list now.

        A checkpoint best-N still holds outlives last-N, and is deleted once best-N drops it too.
        """
        for path in tracked_before - self._tracked_paths():
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
            else:
                continue
            self.logger.info(f"Removed old checkpoint: {path}")

    def load_checkpoint(self) -> dict:
        """Load the checkpoint ``init_option`` selects into the model; return its metadata."""
        option = self.config.init_option
        if option == "path":
            if not self.config.checkpoint_path:
                raise ValueError("checkpoint.checkpoint_path must be set when init_option is 'path'")
            path = self.config.checkpoint_path
        else:
            key = "checkpoints" if option == "last" else "best_checkpoints"
            entries = self._saved_history()[key]
            if not entries:
                raise FileNotFoundError(f"No {option} checkpoint recorded in {self.history_file}")
            path = entries[-1]["path"] if option == "last" else entries[0]["path"]
        metadata = self._load(path)
        self.logger.info(f"Loaded checkpoint: {path}")
        return metadata

    def _saved_history(self) -> dict[str, list]:
        """The history file, which holds the previous run's history until this run first saves."""
        if not os.path.exists(self.history_file):
            return {"checkpoints": [], "best_checkpoints": []}
        with open(self.history_file) as f:
            saved = json.load(f)
        return {"checkpoints": saved.get("checkpoints", []), "best_checkpoints": saved.get("best_checkpoints", [])}
