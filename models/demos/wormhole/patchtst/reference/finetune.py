# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import PatchTSTConfig, PatchTSTForClassification, PatchTSTForPrediction, PatchTSTForRegression

from models.demos.wormhole.patchtst.config import PatchTSTDemoConfig, merge_demo_config, resolve_runtime_config
from models.demos.wormhole.patchtst.demo.data_utils import ARCHIVE_DATASET_FILES, build_observed_mask, load_task_dataset
from models.demos.wormhole.patchtst.reference.hf_reference import (
    adapt_reference_for_runtime_channels,
    adapt_reference_for_runtime_context,
    load_reference_model,
)

FORECAST_FINETUNE_TRAIN_WINDOWS_CAP = 256
FORECAST_FINETUNE_PROGRESS_EVERY = 50


def _validate_state_mapping(
    missing_keys: list[str],
    unexpected_keys: list[str],
    *,
    allowed_missing_prefix: str | None = None,
    allowed_unexpected_prefix: str | None = None,
    error_prefix: str,
) -> None:
    disallowed_missing = [
        key for key in missing_keys if not allowed_missing_prefix or allowed_missing_prefix not in key
    ]
    disallowed_unexpected = [
        key for key in unexpected_keys if not allowed_unexpected_prefix or not key.startswith(allowed_unexpected_prefix)
    ]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            f"{error_prefix} produced an unexpected state mapping. "
            f"missing_keys={disallowed_missing}, unexpected_keys={disallowed_unexpected}"
        )


def _initialize_model_from_attention_base(
    task: str,
    model: torch.nn.Module,
    config: PatchTSTDemoConfig,
) -> tuple[torch.nn.Module, str]:
    if config.channel_mode != "attention":
        return model, "scratch"

    # Channel attention reuses the same attention weights and only adds the second norm block,
    # so initializing from the matching independent checkpoint preserves task semantics.
    base_cfg = PatchTSTDemoConfig(task=task)
    if task in {"classification", "regression"}:
        base_cfg = PatchTSTDemoConfig(task=task, dataset=config.dataset)
    base_ckpt = base_cfg.checkpoint_for_task(task)
    base_revision = base_cfg.checkpoint_revision_for_task(task)
    if not Path(base_ckpt).exists() and base_revision == "local-generated":
        return model, "scratch"

    if Path(base_ckpt).exists():
        base_model = (PatchTSTForClassification if task == "classification" else PatchTSTForRegression).from_pretrained(
            base_ckpt
        )
    else:
        base_model = (PatchTSTForClassification if task == "classification" else PatchTSTForRegression).from_pretrained(
            base_ckpt, revision=base_revision
        )
    missing_keys, unexpected_keys = model.load_state_dict(base_model.state_dict(), strict=False)
    _validate_state_mapping(
        list(missing_keys),
        list(unexpected_keys),
        allowed_missing_prefix=".norm_sublayer2.batchnorm.",
        error_prefix="Channel-attention initialization",
    )
    return model, base_ckpt


def _load_forecast_bootstrap_reference(config: PatchTSTDemoConfig):
    if config.checkpoint_id_override is None:
        return None
    bootstrap_cfg = merge_demo_config(
        PatchTSTDemoConfig(task="forecast"),
        task="forecast",
        channel_mode=config.channel_mode,
        dataset=config.dataset,
        dataset_root=config.dataset_root,
    )
    return load_reference_model(
        task="forecast",
        checkpoint_id=config.checkpoint_id_override,
        revision=config.checkpoint_revision_override or "local-generated",
        config=bootstrap_cfg,
    )


def _build_supervised_patchtst_config(
    task: str,
    sequence_length: int,
    num_channels: int,
    num_targets: int,
    channel_mode: str,
    bootstrap_reference_config: PatchTSTConfig | None = None,
) -> PatchTSTConfig:
    if bootstrap_reference_config is not None:
        common = bootstrap_reference_config.to_dict()
        common.update(
            context_length=sequence_length,
            prediction_length=1,
            num_input_channels=num_channels,
            num_targets=num_targets,
            channel_attention=(channel_mode == "attention"),
        )
        if task == "regression":
            common.update(distribution_output=None, loss="mse")
        return PatchTSTConfig(**common)

    patch_length = min(16, max(4, sequence_length // 8))
    patch_stride = max(1, patch_length // 2)
    if patch_length >= sequence_length:
        patch_length = max(2, sequence_length - 1)
        patch_stride = max(1, patch_length // 2)
    common = dict(
        context_length=sequence_length,
        patch_length=patch_length,
        patch_stride=patch_stride,
        prediction_length=1,
        num_input_channels=num_channels,
        num_targets=num_targets,
        d_model=128,
        ffn_dim=512,
        num_hidden_layers=3,
        num_attention_heads=4,
        share_embedding=True,
        share_projection=True,
        positional_encoding_type="sincos",
        pooling_type="mean",
        norm_type="batchnorm",
        pre_norm=True,
        scaling="std",
        use_cls_token=False,
        channel_attention=(channel_mode == "attention"),
    )
    if task == "regression":
        common.update(distribution_output=None, loss="mse")
    return PatchTSTConfig(**common)


def _fit_supervised_reference(
    task: str,
    config: PatchTSTDemoConfig,
    steps: int,
    batch_size: int,
    learning_rate: float,
    output_dir: Path,
    seed: int,
) -> Path:
    if config.dataset not in ARCHIVE_DATASET_FILES or ARCHIVE_DATASET_FILES[config.dataset].task != task:
        raise ValueError(f"Task {task!r} requires a matching real-task dataset, got {config.dataset!r}.")

    default_context_length = PatchTSTDemoConfig(task=task, dataset=config.dataset).context_length
    requested_context_length = (
        0 if int(config.context_length) == int(default_context_length) else int(config.context_length)
    )
    torch.manual_seed(seed)
    train_batch = load_task_dataset(
        dataset_root=config.dataset_root,
        dataset_name=config.dataset,
        split="train",
        task=task,
        context_length=requested_context_length,
        prediction_length=1,
        max_windows=0,
    )
    if train_batch.target_values is None:
        raise ValueError(f"Task {task!r} dataset loader did not return labels/targets.")

    sequence_length = int(train_batch.past_values.shape[1])
    num_channels = int(train_batch.past_values.shape[2])
    num_targets = (
        int(train_batch.target_values.max().item()) + 1
        if task == "classification"
        else int(train_batch.target_values.shape[-1] if train_batch.target_values.ndim > 1 else 1)
    )
    bootstrap_reference = _load_forecast_bootstrap_reference(config)
    model_config = _build_supervised_patchtst_config(
        task=task,
        sequence_length=sequence_length,
        num_channels=num_channels,
        num_targets=num_targets,
        channel_mode=config.channel_mode,
        bootstrap_reference_config=(bootstrap_reference.config if bootstrap_reference is not None else None),
    )
    model_cls = PatchTSTForClassification if task == "classification" else PatchTSTForRegression
    model = model_cls(model_config)
    if bootstrap_reference is not None:
        missing_keys, unexpected_keys = model.load_state_dict(bootstrap_reference.model.state_dict(), strict=False)
        _validate_state_mapping(
            list(missing_keys),
            list(unexpected_keys),
            allowed_missing_prefix="head.",
            allowed_unexpected_prefix="head.",
            error_prefix="Forecast-to-supervised encoder bootstrap",
        )
    else:
        model, _ = _initialize_model_from_attention_base(
            task=task,
            model=model,
            config=config,
        )
    encoder_frozen = False
    if bootstrap_reference is not None and task == "classification":
        # Public multi_task reuses the forecast encoder exactly and swaps only the classifier head.
        # Freeze non-head parameters here so the exported classification checkpoint remains byte-identical
        # on the shared encoder weights and passes the explicit compatibility check at load time.
        for name, parameter in model.named_parameters():
            if not name.startswith("head."):
                parameter.requires_grad_(False)
        encoder_frozen = True

    model.train()
    if encoder_frozen:
        # BatchNorm running statistics would still drift in train() even when the parameters are frozen.
        # Hold the shared backbone in eval mode so only the classifier head learns and the saved encoder
        # remains exactly identical to the forecast checkpoint used for public multi_task runs.
        model.model.patchifier.eval()
        model.model.scaler.eval()
        model.model.encoder.eval()
        model.head.train()
    optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad), lr=learning_rate)
    observed_mask = build_observed_mask(train_batch.past_values)

    for step in range(steps):
        start = (step * batch_size) % max(int(train_batch.past_values.shape[0]) - batch_size + 1, 1)
        end = start + batch_size
        outputs = model(
            past_values=train_batch.past_values[start:end],
            target_values=train_batch.target_values[start:end],
            past_observed_mask=observed_mask[start:end],
            return_dict=True,
        )
        loss = outputs.loss
        if loss is None:
            prediction_attr = "prediction_logits" if task == "classification" else "regression_outputs"
            prediction = getattr(outputs, prediction_attr)
            if task == "classification":
                loss = F.cross_entropy(
                    prediction.to(torch.float32), train_batch.target_values[start:end].to(torch.long)
                )
            else:
                loss = F.mse_loss(prediction.to(torch.float32), train_batch.target_values[start:end].to(torch.float32))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    return output_dir


def run_reference_finetune(
    config: PatchTSTDemoConfig,
    steps: int,
    batch_size: int,
    learning_rate: float,
    output_dir: Path,
    seed: int = 1337,
) -> Path:
    if steps < 0:
        raise ValueError("steps must be >= 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")

    if config.task in {"classification", "regression"}:
        return _fit_supervised_reference(
            task=config.task,
            config=config,
            steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
            seed=seed,
        )

    if config.task != "forecast":
        raise ValueError("Reference finetune supports only forecast, classification, and regression tasks.")

    torch.manual_seed(seed)

    if config.channel_mode == "attention":
        base_cfg = merge_demo_config(config, channel_mode="independent")
        reference = load_reference_model(
            task="forecast",
            checkpoint_id=base_cfg.checkpoint_for_task("forecast"),
            revision=base_cfg.checkpoint_revision_for_task("forecast"),
            config=base_cfg,
        )
        attention_config = reference.model.config.to_dict()
        attention_config["channel_attention"] = True
        attention_model = PatchTSTForPrediction(reference.model.config.__class__(**attention_config))
        missing_keys, unexpected_keys = attention_model.load_state_dict(reference.model.state_dict(), strict=False)
        _validate_state_mapping(
            list(missing_keys),
            list(unexpected_keys),
            allowed_missing_prefix=".norm_sublayer2.batchnorm.",
            error_prefix="Channel-attention initialization",
        )
        reference.model = attention_model
        reference.config = attention_model.config
    else:
        reference = load_reference_model(
            task="forecast",
            checkpoint_id=config.effective_checkpoint_id("forecast"),
            revision=config.effective_checkpoint_revision("forecast"),
            config=config,
        )
    runtime_cfg = resolve_runtime_config(config, reference.config)
    if int(reference.config.context_length) != int(runtime_cfg.context_length):
        # Forecast stretch checkpoints are created by adapting the base reference once and then saving a real checkpoint at the target context.
        adapt_reference_for_runtime_context(reference, runtime_context_length=runtime_cfg.context_length)

    # Forecast checkpoint generation reuses a bounded rolling training pool instead of materializing
    # `steps * batch_size` full windows. For 4096-step ETTh1 runs that naive strategy eagerly allocates
    # thousands of 4096x7 windows and turns a simple local finetune command into a multi-hundred-MB preload.
    # A capped pool keeps the command practical while the modulo step schedule still cycles the same real windows.
    train_windows = min(max(batch_size * 8, batch_size), FORECAST_FINETUNE_TRAIN_WINDOWS_CAP)
    train_batch = load_task_dataset(
        dataset_root=runtime_cfg.dataset_root,
        dataset_name=runtime_cfg.dataset,
        split="train",
        task="forecast",
        context_length=runtime_cfg.context_length,
        prediction_length=runtime_cfg.prediction_length,
        max_windows=train_windows,
    )
    past_values = train_batch.past_values
    future_values = train_batch.future_values
    if future_values is None:
        raise ValueError("Forecast fine-tuning requires future_values.")

    runtime_dataset_channels = int(past_values.shape[-1])
    if runtime_dataset_channels != int(reference.config.num_input_channels):
        adapt_reference_for_runtime_channels(reference, runtime_num_channels=runtime_dataset_channels)
    expected_channels = int(reference.config.num_input_channels)
    assert int(past_values.shape[-1]) == expected_channels, (
        "Input channel count does not match checkpoint channel count. "
        f"input_channels={int(past_values.shape[-1])}, checkpoint_channels={expected_channels}"
    )
    assert int(future_values.shape[-1]) == expected_channels, (
        "Input channel count does not match checkpoint channel count. "
        f"input_channels={int(future_values.shape[-1])}, checkpoint_channels={expected_channels}"
    )
    observed_mask = build_observed_mask(past_values)

    model = reference.model
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(steps):
        start = (step * batch_size) % max(int(past_values.shape[0]) - batch_size + 1, 1)
        end = start + batch_size
        outputs = model(
            past_values=past_values[start:end],
            future_values=future_values[start:end],
            past_observed_mask=observed_mask[start:end],
            return_dict=True,
        )
        prediction = outputs.prediction_outputs
        loss = F.mse_loss(prediction.to(torch.float32), future_values[start:end].to(torch.float32))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % FORECAST_FINETUNE_PROGRESS_EVERY == 0 or step == 0 or step + 1 == steps:
            print(
                f"[patchtst-finetune] step={step + 1}/{steps} "
                f"loss={loss.item():.6f} train_windows={int(past_values.shape[0])}",
                flush=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    return output_dir
