# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .checkpoint import CheckpointConfig
from .dataset import CustomDatasetConfig
from .logging import LoggingConfig
from .metrics import MetricsConfig
from .model import ModelConfig
from .optimizer import OptimizerConfigs


class TrainerConfig(BaseModel):
    """What every trainer reads. Unknown fields are rejected, so a typo fails instead of being ignored."""

    model_config = ConfigDict(extra="forbid")

    # "sst2", "alpaca" or "custom" (see ``ttml.unified_trainer.data``).
    dataset_id: str
    model: ModelConfig
    dtype: str = "bfloat16"

    # One optimizer, or a list of them that each train the weights their `modules` match. Every trainable weight
    # must get an optimizer. `type` picks one (see ``OPTIMIZERS``), and with it the other fields it takes.
    optimizer: OptimizerConfigs
    batch_size: int = Field(gt=0)
    num_epochs: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    # Validate before training and every this many optimizer steps; 0 disables validation.
    val_steps_freq: int = Field(default=0, ge=0)
    # End the run after this many optimizer steps; 0 runs every epoch to the end.
    max_steps: int = Field(default=0, ge=0)
    seed: int = Field(default=0, ge=0)

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    custom_dataset: Optional[CustomDatasetConfig] = None

    @model_validator(mode="after")
    def _check_custom_dataset(self) -> "TrainerConfig":
        if self.dataset_id == "custom" and self.custom_dataset is None:
            raise ValueError("custom_dataset is required when dataset_id is 'custom'")
        return self

    @model_validator(mode="after")
    def _check_best_rotation(self) -> "TrainerConfig":
        # The metrics a checkpoint carries come from validation.
        if self.checkpoint.rotation_strategy.type == "best" and self.val_steps_freq == 0:
            raise ValueError(
                "checkpoint.rotation_strategy: best ranks checkpoints by validation, so set val_steps_freq"
            )
        return self


class SFTTrainerConfig(TrainerConfig):
    """``SFTTrainer``: supervised fine-tuning of a causal LM on prompt/response pairs."""

    # Every batch is padded to this many tokens; longer examples are dropped.
    max_length: int = Field(gt=0)
    # Clip the global gradient norm to this; 0 disables clipping.
    max_grad_norm: float = Field(default=0.0, ge=0.0)
    # Recompute activations in the backward pass instead of keeping them.
    gradient_checkpointing: bool = False
    # What the trainable weights are stored in. float32 keeps master weights for AdamW, and the model computes on
    # their bf16 view: full fine-tuning needs it, as updates at fine-tuning learning rates round away in bf16.
    master_dtype: Literal["bfloat16", "float32"] = "bfloat16"
