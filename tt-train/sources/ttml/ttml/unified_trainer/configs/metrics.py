# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, Field


class MetricsConfig(BaseModel):
    """Which metrics ``MetricsCallback`` logs, and how often."""

    model_config = ConfigDict(extra="forbid")

    # Train metrics are averaged over, and logged every, this many optimizer steps.
    steps_freq: int = Field(default=10, ge=1)
    # Names the trainer can't provide are ignored, so a recipe can list metrics a later trainer adds.
    # Available: "loss" and "learning_rate" for train, "loss" for validation.
    train_metrics: list[str] = Field(default_factory=lambda: ["loss"])
    val_metrics: list[str] = Field(default_factory=lambda: ["loss"])
