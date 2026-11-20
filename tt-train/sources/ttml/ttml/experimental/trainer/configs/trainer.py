# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .checkpoint import CheckpointConfig
from .dataset import DatasetConfig
from .device import DeviceConfig
from .logging import LoggingConfig
from .metrics import MetricsConfig
from .model import ModelConfig
from .optimizer import OptimizerConfigs
from .stopping import StoppingCriteria


class TrainerConfig(BaseModel):
    """What every trainer reads. Unknown fields are rejected, so a typo fails instead of being ignored."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig
    dataset: DatasetConfig
    # One optimizer, or a list of them that each train the weights their `modules` match. Every trainable weight
    # must get an optimizer. `type` picks one (see ``OPTIMIZERS``), and with it the other fields it takes.
    optimizer: OptimizerConfigs
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    dtype: str = "bfloat16"

    batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    # Clip the global gradient norm to this before every optimizer step; 0 disables clipping.
    max_grad_norm: float = Field(default=0.0, ge=0.0)
    seed: int = Field(default=0, ge=0)
    # When the run ends: after a number of epochs or of optimizer steps. One epoch unless set.
    stopping_criteria: StoppingCriteria = Field(default_factory=lambda: StoppingCriteria(type="epoch", limit=1))

    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    @model_validator(mode="after")
    def _check_best_rotation(self) -> "TrainerConfig":
        # The metrics a checkpoint carries come from validation.
        if self.checkpoint.rotation_strategy.type == "best" and self.metrics.validation.frequency == 0:
            raise ValueError(
                "checkpoint.rotation_strategy: best ranks checkpoints by validation, so set "
                "metrics.validation.frequency"
            )
        return self


class SFTTrainerConfig(TrainerConfig):
    """``SFTTrainer``: supervised fine-tuning of a causal LM on prompt/response pairs. ``model.lora`` trains those
    adapters; without it, every weight is trained."""

    # Every batch is padded to this many tokens; longer examples are dropped.
    max_length: int = Field(gt=0)
