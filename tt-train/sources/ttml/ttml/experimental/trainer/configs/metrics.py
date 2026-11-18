# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, Field

# A name the trainer can't provide is ignored, so a recipe can list metrics a later trainer adds.


class TrainMetricsConfig(BaseModel):
    """The metrics of training, which ``MetricsCallback`` logs."""

    model_config = ConfigDict(extra="forbid")

    # The metrics are averaged over, and logged every, this many optimizer steps.
    steps_freq: int = Field(default=10, ge=1)
    # Available: "loss" and "learning_rate".
    metric_names: list[str] = Field(default_factory=lambda: ["loss"])


class ValidationMetricsConfig(BaseModel):
    """Validation, and the metrics of it ``MetricsCallback`` logs."""

    model_config = ConfigDict(extra="forbid")

    # The trainer validates before training and every this many optimizer steps; 0 disables validation.
    steps_freq: int = Field(default=0, ge=0)
    # Available: "loss".
    metric_names: list[str] = Field(default_factory=lambda: ["loss"])


class MetricsConfig(BaseModel):
    """What a run measures and how often: the metrics of training, and validation."""

    model_config = ConfigDict(extra="forbid")

    train: TrainMetricsConfig = Field(default_factory=TrainMetricsConfig)
    validation: ValidationMetricsConfig = Field(default_factory=ValidationMetricsConfig)
