# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CheckpointConfig(BaseModel):
    """When ``CheckpointCallback`` saves, how many checkpoints ``CheckpointManager`` keeps, and what a run
    starts from.

    A checkpoint is a directory holding what the model trains: a PEFT adapter for a LoRA run, an HF model
    directory for a full fine-tune.
    """

    model_config = ConfigDict(extra="forbid")

    save_strategy: Literal["epoch", "step", "none"] = "none"
    steps_freq: int = Field(default=100, ge=1)
    epoch_freq: int = Field(default=1, ge=1)

    # Checkpoints go to <project_dir>/checkpoints/.
    project_dir: str = "runs/unified_trainer"
    # Saved at the end of the run unless save_strategy is "none".
    final_checkpoint_name: str = "final"
    # Optimizer state is not saved (there is no resume); True is rejected rather than ignored.
    save_optim: bool = False
    # At least 1, so the checkpoint just saved (e.g. the final one) is never rotated away.
    keep_last_n: int = Field(default=3, ge=1)
    keep_best_n: int = Field(default=0, ge=0)
    # Disables rotation: no checkpoint is ever deleted.
    keep_all_checkpoints: bool = False
    checkpoint_metric: str = "val/loss"
    checkpoint_metric_mode: Literal["min", "max"] = "min"

    # Initialize the trained weights from a checkpoint. Not a resume: training still starts at step 0.
    init_from_checkpoint: bool = False
    init_option: Literal["last", "best", "path"] = "last"
    # Used when init_option is "path".
    checkpoint_path: str = ""
