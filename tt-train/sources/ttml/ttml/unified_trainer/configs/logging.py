# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, Field


class LoggingConfig(BaseModel):
    """Console and Weights & Biases logging, used by ``TrainingLogger``."""

    model_config = ConfigDict(extra="forbid")

    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: str = ""
    # Also names the console logger.
    wandb_run_name: str = "ttml-unified-trainer"
    wandb_tags: list[str] = Field(default_factory=list)
