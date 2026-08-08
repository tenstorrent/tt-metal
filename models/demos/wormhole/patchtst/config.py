# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

PUBLIC_TASK_MODE = Literal["forecast", "regression", "pretraining", "classification", "multi_task"]
CHANNEL_MODE = Literal["independent", "attention"]
DATASET_NAME = Literal[
    "etth1",
    "weather",
    "traffic",
    "electricity",
    "exchange_rate",
    "heartbeat_cls",
    "flood_modeling1_reg",
]
_ATTENTION_CHECKPOINT_KEY_BY_TASK = {
    "forecast": "forecast_attention",
    "regression": "regression_attention",
    "pretraining": "channel_attention",
    "classification": "classification_attention",
}


@dataclass
class PatchTSTDemoConfig:
    task: PUBLIC_TASK_MODE = "forecast"
    channel_mode: CHANNEL_MODE = "independent"
    share_embedding: bool = True
    strict_fallback: bool = True
    use_trace: bool = False

    batch_size: int = 1
    context_length: int = 512
    prediction_length: int = 96
    patch_length: int = 12
    patch_stride: int = 12

    dataset: DATASET_NAME = "etth1"
    dataset_root: Path = Path("data/patchtst")
    split: Literal["train", "val", "test"] = "test"
    max_windows: int = 64
    window_offset: int = 0
    masking_seed: int = 1337

    required_latency_ms: float = 40.0
    required_throughput_sps: float = 150.0
    required_mse_delta_ratio: float = 0.05
    required_mae_delta_ratio: float = 0.05
    required_rmse_delta_ratio: float = 0.05
    required_correlation: float = 0.90
    required_quality_correlation: float = 0.65
    required_quality_correlation_delta: float = 0.05
    required_classification_accuracy_delta: float = 0.02
    required_classification_f1_delta: float = 0.02

    stretch_latency_ms: float = 15.0
    stretch_throughput_sps: float = 800.0
    stretch_context_length: int = 4096
    stretch_required_batch_size: int = 16

    output_dir: Path = Path("generated/patchtst/report")

    required_versions: dict[str, str] = field(
        default_factory=lambda: {
            "transformers": "4.53.0",
            "datasets": "2.9.0",
            "evaluate": "0.4.0",
        }
    )
    checkpoint_ids: dict[str, str] = field(
        default_factory=lambda: {
            "forecast": "ibm-research/test-patchtst",
            "forecast_attention": "models/demos/wormhole/patchtst/artifacts/finetune/forecast_attention_etth1_ckpt",
            "regression": "models/demos/wormhole/patchtst/artifacts/finetune/regression_flood_modeling1_ckpt",
            "regression_attention": "models/demos/wormhole/patchtst/artifacts/finetune/regression_flood_modeling1_attention_ckpt",
            "pretraining": "ibm-research/testing-patchtst_etth1_pretrain",
            "classification": "models/demos/wormhole/patchtst/artifacts/finetune/classification_heartbeat_ckpt",
            "classification_attention": "models/demos/wormhole/patchtst/artifacts/finetune/classification_heartbeat_attention_ckpt",
            "channel_attention": "AminiTech/amini-28M-v1",
        }
    )
    checkpoint_revisions: dict[str, str] = field(
        default_factory=lambda: {
            "forecast": "a8c54ceeaf3c9c4d864a33d650b156cc18b3ab53",
            "forecast_attention": "local-generated",
            "regression": "local-generated",
            "regression_attention": "local-generated",
            "pretraining": "5dcaab0b40fa2b8a3843e2197e1f6b6e53ecd363",
            "classification": "local-generated",
            "classification_attention": "local-generated",
            "channel_attention": "5ce4e875ad358f4a3f842f33c7d36b8c7dd433ac",
        }
    )
    checkpoint_id_override: str | None = None
    checkpoint_revision_override: str | None = None
    allow_reference_context_adaptation: bool = True
    allow_reference_channel_adaptation: bool = True
    multi_task_checkpoint_ids: dict[str, str] = field(
        default_factory=lambda: {
            "forecast": "models/demos/wormhole/patchtst/artifacts/finetune/forecast_heartbeat_multitask_ckpt",
            "classification": "models/demos/wormhole/patchtst/artifacts/finetune/classification_heartbeat_multitask_ckpt",
        }
    )
    multi_task_checkpoint_revisions: dict[str, str] = field(
        default_factory=lambda: {
            "forecast": "local-generated",
            "classification": "local-generated",
        }
    )

    def __post_init__(self) -> None:
        # PatchTST demo runs in strict no-fallback mode.
        self.strict_fallback = True
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.context_length <= 0:
            raise ValueError("context_length must be > 0")
        if self.prediction_length <= 0:
            raise ValueError("prediction_length must be > 0")
        if self.window_offset < 0:
            raise ValueError("window_offset must be >= 0")
        if self.patch_length <= 0:
            raise ValueError("patch_length must be > 0")
        if self.patch_stride <= 0:
            raise ValueError("patch_stride must be > 0")
        if self.context_length <= self.patch_length:
            raise ValueError(f"context_length ({self.context_length}) must be > patch_length ({self.patch_length})")
        if self.channel_mode == "attention" and self.task == "multi_task":
            raise ValueError("channel_mode='attention' is not supported for multi_task runs.")

    def _attention_checkpoint_key_for_task(self, task: str) -> str:
        if task not in _ATTENTION_CHECKPOINT_KEY_BY_TASK:
            raise ValueError(f"channel_mode='attention' is not supported for task: {task}")
        return _ATTENTION_CHECKPOINT_KEY_BY_TASK[task]

    def checkpoint_for_task(self, task: str | None = None) -> str:
        key = task or self.task
        if key == "multi_task":
            if self.channel_mode == "attention":
                raise ValueError("channel_mode='attention' is not supported for multi_task runs.")
            key = "forecast"
        if self.channel_mode == "attention":
            key = self._attention_checkpoint_key_for_task(key)
        if key not in self.checkpoint_ids:
            raise ValueError(f"No checkpoint configured for task: {key}")
        return self.checkpoint_ids[key]

    def checkpoint_revision_for_task(self, task: str | None = None) -> str:
        key = task or self.task
        if key == "multi_task":
            if self.channel_mode == "attention":
                raise ValueError("channel_mode='attention' is not supported for multi_task runs.")
            key = "forecast"
        if self.channel_mode == "attention":
            key = self._attention_checkpoint_key_for_task(key)
        if key not in self.checkpoint_revisions:
            raise ValueError(f"No checkpoint revision configured for task: {key}")
        return self.checkpoint_revisions[key]

    def effective_checkpoint_id(self, task: str | None = None) -> str:
        # Validation presets may pin a task-matched local checkpoint for a stretched workload instead of the default task checkpoint.
        return self.checkpoint_id_override or self.checkpoint_for_task(task)

    def effective_checkpoint_revision(self, task: str | None = None) -> str:
        # When a local stretch checkpoint is pinned, its revision metadata must travel with it for reproducible reporting.
        return self.checkpoint_revision_override or self.checkpoint_revision_for_task(task)


def merge_demo_config(base: PatchTSTDemoConfig, **overrides) -> PatchTSTDemoConfig:
    return PatchTSTDemoConfig(**(base.__dict__ | overrides))


def resolve_runtime_config(base: PatchTSTDemoConfig, reference_config) -> PatchTSTDemoConfig:
    defaults = PatchTSTDemoConfig(task=base.task)
    return merge_demo_config(
        base,
        context_length=int(reference_config.context_length)
        if base.context_length == defaults.context_length
        else base.context_length,
        prediction_length=int(reference_config.prediction_length)
        if base.prediction_length == defaults.prediction_length
        else base.prediction_length,
        patch_length=int(reference_config.patch_length),
        patch_stride=int(reference_config.patch_stride),
    )
