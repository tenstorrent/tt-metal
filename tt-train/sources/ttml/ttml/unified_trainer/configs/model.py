# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .lora import LoraConfig


class ModelConfig(BaseModel):
    """The model a run trains: the pretrained model it builds and how, and what of it trains."""

    model_config = ConfigDict(extra="forbid")

    # An HF hub id or a local HF model directory.
    name: str = Field(min_length=1)
    # The LoRA adapters to train. Unset would train every weight (full fine-tuning), which no trainer supports yet.
    lora: Optional[LoraConfig] = None
    # Build the model to recompute activations in the backward pass instead of keeping them.
    gradient_checkpointing: bool = False
