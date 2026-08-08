# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Based on HuggingFace PatchTST and the PatchTST paper:
# https://huggingface.co/docs/transformers/en/model_doc/patchtst
# https://github.com/huggingface/transformers/tree/main/src/transformers/models/patchtst
# https://arxiv.org/abs/2211.14730

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import PatchTSTForClassification, PatchTSTForPrediction, PatchTSTForPretraining, PatchTSTForRegression
from transformers.models.patchtst.modeling_patchtst import PatchTSTPositionalEncoding

from models.demos.wormhole.patchtst.config import PatchTSTDemoConfig


def _model_cls(task: str):
    if task == "forecast":
        return PatchTSTForPrediction
    if task == "regression":
        return PatchTSTForRegression
    if task == "pretraining":
        return PatchTSTForPretraining
    if task == "classification":
        return PatchTSTForClassification
    raise ValueError(f"Unsupported task: {task}")


@dataclass
class ReferenceArtifacts:
    model: torch.nn.Module
    config: object
    checkpoint_id: str
    task: str


def adapt_reference_for_runtime_context(artifacts: ReferenceArtifacts, runtime_context_length: int) -> bool:
    reference_context = int(artifacts.config.context_length)
    runtime_context_length = int(runtime_context_length)
    if runtime_context_length == reference_context:
        return False

    if str(artifacts.config.positional_encoding_type) != "sincos":
        raise ValueError(
            "Runtime context differs from checkpoint context and positional encoding is not sincos. "
            f"runtime_context={runtime_context_length}, checkpoint_context={reference_context}"
        )

    patchifier = artifacts.model.model.patchifier
    patch_length = int(patchifier.patch_length)
    patch_stride = int(patchifier.patch_stride)
    if runtime_context_length <= patch_length:
        raise ValueError(
            f"runtime_context_length ({runtime_context_length}) must be greater than patch_length ({patch_length})."
        )

    num_patches = (runtime_context_length - patch_length) // patch_stride + 1
    new_sequence_length = patch_length + patch_stride * (num_patches - 1)
    sequence_start = runtime_context_length - new_sequence_length

    artifacts.config.context_length = runtime_context_length
    patchifier.sequence_length = runtime_context_length
    patchifier.sequence_start = sequence_start
    patchifier.num_patches = num_patches

    encoder = artifacts.model.model.encoder
    old_positional = encoder.positional_encoder
    replacement_positional = PatchTSTPositionalEncoding(artifacts.config, num_patches=num_patches).to(
        old_positional.position_enc.device
    )
    if hasattr(old_positional, "cls_token") and hasattr(replacement_positional, "cls_token"):
        with torch.no_grad():
            replacement_positional.cls_token.copy_(old_positional.cls_token)
    encoder.positional_encoder = replacement_positional
    return True


def adapt_reference_for_runtime_channels(artifacts: ReferenceArtifacts, runtime_num_channels: int) -> bool:
    reference_num_channels = int(artifacts.config.num_input_channels)
    runtime_num_channels = int(runtime_num_channels)
    if runtime_num_channels == reference_num_channels:
        return False

    if artifacts.task not in {"forecast", "pretraining"}:
        raise ValueError(
            "Runtime channel count differs from checkpoint channel count, and this task does not support safe "
            f"channel adaptation. task={artifacts.task}, runtime_channels={runtime_num_channels}, "
            f"checkpoint_channels={reference_num_channels}"
        )

    config_dict = artifacts.model.config.to_dict()
    config_dict["num_input_channels"] = runtime_num_channels
    adapted_config = artifacts.model.config.__class__(**config_dict)
    adapted_model = _model_cls(artifacts.task)(adapted_config)

    missing_keys, unexpected_keys = adapted_model.load_state_dict(artifacts.model.state_dict(), strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Channel adaptation produced incompatible parameter mapping. "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )

    adapted_model.eval()
    artifacts.model = adapted_model
    artifacts.config = adapted_model.config
    return True


def load_reference_model(
    task: str,
    checkpoint_id: str | None = None,
    config: PatchTSTDemoConfig | None = None,
    revision: str | None = None,
) -> ReferenceArtifacts:
    if config is None:
        config = PatchTSTDemoConfig(task=task)
    if task == "multi_task":
        raise ValueError("Use load_multi_task_references for multi_task wiring.")

    ckpt = checkpoint_id or config.checkpoint_for_task(task)
    resolved_revision = revision or config.checkpoint_revision_for_task(task)
    # Classification and regression use locally generated task-matched checkpoints. Fail with an explicit
    # setup error here instead of falling through to a confusing Hugging Face repo-id lookup.
    if resolved_revision == "local-generated" and not Path(ckpt).exists():
        channel_mode_hint = " --channel-mode attention" if config.channel_mode == "attention" else ""
        override_hint = ""
        if config.checkpoint_id_override is not None:
            override_hint = (
                " This run pins a local checkpoint override, so generate the exact dataset/context-matched checkpoint "
                "with the documented `finetune` command before rerunning."
            )
        raise FileNotFoundError(
            "Required local PatchTST checkpoint was not found: "
            f"{ckpt}. Generate it with `python -m models.demos.wormhole.patchtst.demo.demo finetune --task {task}{channel_mode_hint}` "
            f"after downloading the matching labeled dataset.{override_hint}"
        )
    model_cls = _model_cls(task)
    model = (
        model_cls.from_pretrained(ckpt)
        if Path(ckpt).exists()
        else model_cls.from_pretrained(ckpt, revision=resolved_revision)
    )
    model.eval()
    return ReferenceArtifacts(model=model, config=model.config, checkpoint_id=ckpt, task=task)


def reference_forward(
    artifacts: ReferenceArtifacts,
    past_values: torch.Tensor,
    future_values: torch.Tensor | None = None,
    target_values: torch.Tensor | None = None,
    past_observed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs = {
        "past_values": past_values,
        "past_observed_mask": past_observed_mask,
        "return_dict": True,
    }
    if artifacts.task == "forecast" and future_values is not None:
        kwargs["future_values"] = future_values
    if target_values is not None:
        kwargs["target_values"] = target_values

    with torch.no_grad():
        outputs = artifacts.model(**kwargs)
    if artifacts.task == "forecast":
        return outputs.prediction_outputs
    if artifacts.task == "regression":
        return outputs.regression_outputs
    if artifacts.task == "pretraining":
        return outputs.prediction_output
    return outputs.prediction_logits
