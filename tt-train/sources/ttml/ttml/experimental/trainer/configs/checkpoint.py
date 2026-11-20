# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class StepSave(BaseModel):
    """Save every ``frequency`` optimizer steps, and the final checkpoint on the run's last step."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["step"] = "step"
    # A multiple of metrics.train.frequency, and of metrics.validation.frequency when the run validates, so
    # every checkpoint carries the metrics of its own step. `CheckpointCallback` checks it when training starts.
    frequency: int = Field(default=100, ge=1)
    # Checkpoints go to <project_dir>/checkpoints/.
    project_dir: str = "runs"
    # The name of the checkpoint of the run's last step.
    final_checkpoint_name: str = "final"


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
    # Checkpoints carry the metrics MetricsCallback logs, of their own step: "val/loss" needs "loss" in
    # metrics.validation.metric_names, and a run that validates.
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

    # "init" loads the checkpoint's weights: training starts at step 0 with a fresh optimizer.
    type: Literal["init"] = "init"
    # A checkpoint directory, in the format ``CheckpointConfig.format`` picks.
    path: str = Field(min_length=1)


class CheckpointFormat(BaseModel):
    """How a checkpoint stores the weights the run trains; a run loads its ``init_strategy`` checkpoint in this
    format too.

    ``hf`` writes HF layouts: a LoRA run saves a PEFT adapter, full fine-tuning the model as ``save_pretrained`` lays
    one out. ``tt`` writes ttml's layout to one ``checkpoint.pkl`` with ``ttml.checkpointing``: the LoRA adapters of
    a LoRA run, every parameter for full fine-tuning.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["hf", "tt"] = "hf"


# `type` picks one, and with it the other fields it takes.
SaveStrategy = Annotated[Union[StepSave, NoSave], Field(discriminator="type")]
RotationStrategy = Annotated[Union[LastRotation, BestRotation, NoRotation], Field(discriminator="type")]


class CheckpointConfig(BaseModel):
    """The checkpoint the run starts from, which ``setup`` loads, and the checkpoints the run saves: when
    ``CheckpointCallback`` saves them, which ones ``CheckpointManager`` keeps, and how they store the model.

    A checkpoint is a directory holding the model's weights, in the layout ``format`` picks.
    """

    model_config = ConfigDict(extra="forbid")

    save_strategy: SaveStrategy = Field(default_factory=NoSave)
    # Never deletes the checkpoint just saved, such as the final one.
    rotation_strategy: RotationStrategy = Field(default_factory=LastRotation)
    # Unset starts from the pretrained weights.
    init_strategy: Optional[InitStrategy] = None
    format: CheckpointFormat = Field(default_factory=CheckpointFormat)
