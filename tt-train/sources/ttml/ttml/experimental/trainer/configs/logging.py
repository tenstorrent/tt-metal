# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LoggingConfig(BaseModel):
    """Console and Weights & Biases logging, used by ``TrainingLogger``."""

    model_config = ConfigDict(extra="forbid")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    use_wandb: bool = False
    wandb_project: str = ""
    # Also names the console logger.
    wandb_run_name: str = "ttml-experimental-trainer"
    wandb_tags: list[str] = Field(default_factory=list)

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper_log_level(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value
