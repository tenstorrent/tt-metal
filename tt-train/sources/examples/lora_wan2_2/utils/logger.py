# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""wandb logger with a stdout fallback."""

from __future__ import annotations

try:
    import wandb  # noqa: F401

    _WANDB_OK = True
except ImportError:
    _WANDB_OK = False


class Logger:
    def __init__(self, enabled: bool, project: str, run_name: str, config: dict):
        self.enabled = enabled and _WANDB_OK
        if self.enabled:
            import wandb

            self.run = wandb.init(project=project, name=run_name, config=config)
        elif enabled and not _WANDB_OK:
            print("[logger] wandb not installed; stdout only")

    def log(self, data: dict, step: int | None = None):
        if self.enabled:
            import wandb

            wandb.log(data, step=step)
        else:
            scal = {k: (float(v) if isinstance(v, (int, float)) else type(v).__name__) for k, v in data.items()}
            print(f"[step {step}] " + " ".join(f"{k}={v}" for k, v in scal.items()))

    def finish(self):
        if self.enabled:
            import wandb

            wandb.finish()
