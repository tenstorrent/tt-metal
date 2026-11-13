# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .lora import LoraConfig


class InitStrategy(BaseModel):
    """The checkpoint a run starts from."""

    model_config = ConfigDict(extra="forbid")

    # "init" loads the trained weights only: training starts at step 0 with a fresh optimizer. "resume" also loads
    # the optimizer state, which a checkpoint holds only if saved with `checkpoint.save_optim`, and continues the run
    # after the checkpoint's step, on the batches it would have trained on next.
    type: Literal["init", "resume"]
    # A checkpoint directory.
    path: str = Field(min_length=1)


class ModelConfig(BaseModel):
    """The model a run trains: the pretrained model it builds, what of it trains, and the checkpoint it starts from."""

    model_config = ConfigDict(extra="forbid")

    # An HF hub id or a local HF model directory.
    name: str = Field(min_length=1)
    # Train LoRA adapters; unset trains every weight (full fine-tuning).
    lora: Optional[LoraConfig] = None
    # Unset starts from the pretrained weights.
    init_strategy: Optional[InitStrategy] = None
