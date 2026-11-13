# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class StepSave(BaseModel):
    """Save every ``frequency`` optimizer steps."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["step"] = "step"
    frequency: int = Field(default=100, ge=1)


class EpochSave(BaseModel):
    """Save at the end of every ``frequency``-th epoch."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["epoch"] = "epoch"
    frequency: int = Field(default=1, ge=1)


class NoSave(BaseModel):
    """Save no checkpoint, not even the final one."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["none"] = "none"


class LastRotation(BaseModel):
    """Keep the ``keep_n`` most recent checkpoints."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["last"] = "last"
    keep_n: int = Field(default=3, ge=1)


class BestRotation(BaseModel):
    """Keep the ``keep_n`` checkpoints with the best ``metric``, and the latest one."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["best"] = "best"
    keep_n: int = Field(default=3, ge=1)
    # Checkpoints carry the "val/loss" of the latest validation.
    metric: str = "val/loss"
    # "min" keeps the lowest values, "max" the highest.
    mode: Literal["min", "max"] = "min"


class NoRotation(BaseModel):
    """Keep every checkpoint."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["none"] = "none"


class InitStrategy(BaseModel):
    """The checkpoint a run starts from."""

    model_config = ConfigDict(extra="forbid")

    # "init" loads the trained weights only: training starts at step 0 with a fresh optimizer. "resume" also loads
    # the optimizer state, which a checkpoint holds only if saved with `save_optim`, and continues the run after the
    # checkpoint's step, on the batches it would have trained on next.
    type: Literal["init", "resume"]
    # A checkpoint directory.
    path: str = Field(min_length=1)


# `type` picks one, and with it the other fields it takes.
SaveStrategy = Annotated[Union[StepSave, EpochSave, NoSave], Field(discriminator="type")]
RotationStrategy = Annotated[Union[LastRotation, BestRotation, NoRotation], Field(discriminator="type")]


class CheckpointConfig(BaseModel):
    """The checkpoint ``CheckpointCallback`` starts the run from, when it saves, and which checkpoints
    ``CheckpointManager`` keeps.

    A checkpoint is a directory holding what the model trains, as a PEFT adapter.
    """

    model_config = ConfigDict(extra="forbid")

    save_strategy: SaveStrategy = Field(default_factory=NoSave)
    # Never deletes the checkpoint just saved, such as the final one.
    rotation_strategy: RotationStrategy = Field(default_factory=LastRotation)
    # Unset starts from the pretrained weights.
    init_strategy: Optional[InitStrategy] = None

    # Checkpoints go to <project_dir>/checkpoints/.
    project_dir: str = "runs/unified_trainer"
    # Saved at the end of the run unless save_strategy is "none".
    final_checkpoint_name: str = "final"
    # Also save the optimizer state, which a run resuming from the checkpoint needs.
    save_optim: bool = False
