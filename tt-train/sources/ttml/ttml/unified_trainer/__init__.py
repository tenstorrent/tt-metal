# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fine-tuning of HF models on ttml, driven by one validated config, for scripts and services alike.

A host builds a trainer once with its callbacks and runs any number of jobs on it::

    trainer = SFTTrainer(callbacks=[MetricsCallback(), CheckpointCallback()])
    try:
        trainer.setup(load_config("recipe.yaml"))
        trainer.train()
    finally:
        trainer.cleanup()

``python -m ttml.unified_trainer --config recipe.yaml`` does the same from the command line. See
``tt-train/docs/UNIFIED_TRAINER.md``.
"""

from .callbacks import Callback, CallbackHandler, CheckpointCallback, MetricsCallback
from .configs import (
    OPTIMIZERS,
    AdamWConfig,
    BestRotation,
    CheckpointConfig,
    CustomDatasetConfig,
    DatasetConfig,
    EpochSave,
    InitStrategy,
    LastRotation,
    LoggingConfig,
    LoraConfig,
    MetricsConfig,
    ModelConfig,
    NoRotation,
    NoSave,
    OptimizerConfig,
    SFTTrainerConfig,
    StepSave,
    TrainerConfig,
    load_config,
)
from .providers import UnsupportedTrainingConfig
from .sft_trainer import SFTTrainer
from .trainer import Trainer

__all__ = [
    "OPTIMIZERS",
    "AdamWConfig",
    "BestRotation",
    "Callback",
    "CallbackHandler",
    "CheckpointCallback",
    "CheckpointConfig",
    "CustomDatasetConfig",
    "DatasetConfig",
    "EpochSave",
    "InitStrategy",
    "LastRotation",
    "LoggingConfig",
    "LoraConfig",
    "MetricsCallback",
    "MetricsConfig",
    "ModelConfig",
    "NoRotation",
    "NoSave",
    "OptimizerConfig",
    "SFTTrainer",
    "SFTTrainerConfig",
    "StepSave",
    "Trainer",
    "TrainerConfig",
    "UnsupportedTrainingConfig",
    "load_config",
]
