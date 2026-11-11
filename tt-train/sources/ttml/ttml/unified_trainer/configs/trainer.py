# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .checkpoint import CheckpointConfig
from .dataset import CustomDatasetConfig
from .logging import LoggingConfig
from .lora import LoraConfig
from .metrics import MetricsConfig


class TrainerConfig(BaseModel):
    """What every trainer reads. Unknown fields are rejected, so a typo fails instead of being ignored."""

    # No protected namespaces: `model_name` is a field, not pydantic's.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # "sst2", "alpaca" or "custom" (see ``ttml.unified_trainer.data``).
    dataset_id: str
    # An HF hub id or a local HF model directory.
    model_name: str
    dtype: str = "bfloat16"

    learning_rate: float = Field(ge=0.0)
    batch_size: int = Field(gt=0)
    num_epochs: int = Field(default=1, gt=0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    # Validate before training and every this many optimizer steps; 0 disables validation.
    val_steps_freq: int = Field(default=0, ge=0)
    # End the run after this many optimizer steps; 0 runs every epoch to the end.
    max_steps: int = Field(default=0, ge=0)
    seed: int = Field(default=0, ge=0)

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)

    # Train LoRA adapters; unset trains every weight (full fine-tuning).
    lora: Optional[LoraConfig] = None
    custom_dataset: Optional[CustomDatasetConfig] = None

    @model_validator(mode="after")
    def _check_custom_dataset(self) -> "TrainerConfig":
        if self.dataset_id == "custom" and self.custom_dataset is None:
            raise ValueError("custom_dataset is required when dataset_id is 'custom'")
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
