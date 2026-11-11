# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Optional

from ..configs import LoggingConfig


class TrainingLogger:
    """Logs to stdout and, with ``use_wandb``, to a Weights & Biases run.

    W&B failures are logged as warnings and never fail training.
    """

    def __init__(self, config: LoggingConfig, run_config: Optional[dict[str, Any]] = None) -> None:
        """``run_config`` is uploaded as the W&B run config; defaults to ``config``."""
        self.config = config
        self._use_wandb = config.use_wandb
        self._finished = False
        self._wandb = None
        self._run = None

        self.std_logger = logging.getLogger(config.wandb_run_name)
        level = getattr(logging, config.log_level.upper())
        self.std_logger.setLevel(level)
        self.std_logger.handlers.clear()
        self.std_logger.propagate = False
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        self.std_logger.addHandler(handler)

        if self._use_wandb:
            self._start_wandb(run_config if run_config is not None else config.model_dump())

    def _start_wandb(self, run_config: dict[str, Any]) -> None:
        try:
            import wandb

            self._run = wandb.init(
                project=self.config.wandb_project or None,
                name=self.config.wandb_run_name,
                tags=self.config.wandb_tags,
                config=run_config,
            )
            self._wandb = wandb
        except Exception as e:  # noqa: BLE001
            self.std_logger.error(f"Failed to initialize W&B, logging to stdout only: {e}")
            self._use_wandb = False

    def info(self, message: str) -> None:
        self.std_logger.info(message)

    def warning(self, message: str) -> None:
        self.std_logger.warning(message)

    def debug(self, message: str) -> None:
        self.std_logger.debug(message)

    def error(self, message: str, traceback_str: Optional[str] = None) -> None:
        self.std_logger.error(message)
        if traceback_str:
            self.std_logger.error(traceback_str)
        if self._use_wandb:
            try:
                self._run.alert(title="Training failed", text=message, level=self._wandb.AlertLevel.ERROR)
            except Exception as e:  # noqa: BLE001
                self.std_logger.warning(f"Failed to send the W&B alert: {e}")

    def log_metrics(self, metrics: dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
        """Log ``metrics`` at ``step``. With ``commit=False`` W&B batches them with the next commit."""
        # An empty dict only commits the batched W&B logs.
        if metrics:
            text = " | ".join(f"{k}: {v:.6g}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items())
            self.std_logger.info(f"Step {step} | {text}" if step is not None else text)
        if self._use_wandb:
            try:
                self._run.log(metrics, step=step, commit=commit)
            except Exception as e:  # noqa: BLE001
                self.std_logger.warning(f"Failed to log to W&B: {e}")

    def log_model_info(self, info: dict[str, Any]) -> None:
        self.std_logger.info("Model information:")
        for key, value in info.items():
            self.std_logger.info(f"  {key}: {value}")
        if self._use_wandb:
            try:
                self._run.config.update({"model": info})
            except Exception as e:  # noqa: BLE001
                self.std_logger.warning(f"Failed to log model info to W&B: {e}")

    def log_artifact(self, path: str, artifact_type: str = "model", name: Optional[str] = None) -> None:
        """Upload a file or directory to W&B; a no-op without W&B."""
        if not self._use_wandb:
            return
        try:
            artifact = self._wandb.Artifact(name or os.path.basename(path), type=artifact_type)
            if os.path.isdir(path):
                artifact.add_dir(path)
            else:
                artifact.add_file(path)
            self._run.log_artifact(artifact)
        except Exception as e:  # noqa: BLE001
            self.std_logger.warning(f"Failed to log artifact {path} to W&B: {e}")

    def finish(self) -> None:
        """Close the W&B run. Safe to call more than once."""
        if self._finished:
            return
        self._finished = True
        if self._use_wandb:
            try:
                self._run.finish()
            except Exception as e:  # noqa: BLE001
                self.std_logger.warning(f"Failed to finish the W&B run: {e}")
