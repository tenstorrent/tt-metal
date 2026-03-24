# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import copy
from argparse import ArgumentParser, Namespace
import random
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import torch
from torch import nn

import ttnn
from models.experimental.mole.reference.config import MoLEConfig
from models.experimental.mole.reference.mole import MixtureOfLinearExperts, create_reference_expert
from models.experimental.mole.tt.dlinear import TtDLinearExpert
from models.experimental.mole.tt.mole import TtMoLE
from models.experimental.mole.tt.rlinear import TtRLinearExpert
from models.experimental.mole.tt.rmlp import TtRMLPExpert
from models.experimental.mole.tt.common import (
    TtRuntimeOptions,
    upload_timeseries_and_marks_to_device,
    to_torch_with_cached_host,
)
from models.experimental.mole.utils.datasets import (
    create_real_dataset_loaders,
    RegressionMetricTotals,
    finalize_regression_metric_totals,
    update_regression_metric_totals,
)


DEFAULT_TTNN_L1_SMALL_SIZE = 24576
BASE_MODEL_TYPES = ("dlinear", "rlinear", "rmlp")
TT_EXPERT_CLASSES = {
    "dlinear": TtDLinearExpert,
    "rlinear": TtRLinearExpert,
    "rmlp": TtRMLPExpert,
}


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-3
    steps: int = 80
    eval_batch_size: int = 32
    validation_interval: int = 10
    max_validation_batches: int | None = 32


@dataclass
class TraceState:
    model: object
    device: object
    trace_id: int
    trace_output: object  # single Tensor or tuple of Tensors


def add_dataset_arguments(
    parser: ArgumentParser,
    *,
    dataset_help: str,
    dataset_path_help: str,
) -> None:
    parser.add_argument("--dataset-name", type=str, required=True, help=dataset_help)
    parser.add_argument("--dataset-path", type=str, default=None, help=dataset_path_help)


def add_model_arguments(
    parser: ArgumentParser,
    *,
    include_input_dim: bool = False,
) -> None:
    parser.add_argument("--base-model-type", choices=BASE_MODEL_TYPES, default="dlinear")
    for name, default in (("num-experts", 4), ("seq-len", 96), ("pred-len", 24)):
        parser.add_argument(f"--{name}", type=int, default=default)
    if include_input_dim:
        parser.add_argument("--input-dim", type=int, default=7)
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="Time-feature layout for marks: hourly-style (ends with 'h', e.g. h) uses 4 features; otherwise 5 (minute-style)",
    )


def add_training_arguments(
    parser: ArgumentParser,
    *,
    default_batch_size: int,
    default_eval_batch_size: int | None = None,
    default_steps: int | None = None,
    include_seed: bool = True,
) -> None:
    if default_steps is not None:
        parser.add_argument("--steps", type=int, default=default_steps)
    parser.add_argument("--batch-size", type=int, default=default_batch_size)
    if default_eval_batch_size is not None:
        parser.add_argument("--eval-batch-size", type=int, default=default_eval_batch_size)
        parser.add_argument("--learning-rate", type=float, default=1e-3)
    if include_seed:
        parser.add_argument("--seed", type=int, default=0)


def model_config_from_args(args: Namespace) -> MoLEConfig:
    model_kwargs = {
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "base_model_type": args.base_model_type,
        "num_experts": args.num_experts,
        "freq": args.freq,
    }
    for name in ("input_dim", "head_dropout"):
        if hasattr(args, name):
            model_kwargs[name] = getattr(args, name)
    return MoLEConfig(**model_kwargs)


def training_config_from_args(args: Namespace) -> TrainingConfig:
    training_kwargs = {"batch_size": args.batch_size}
    if hasattr(args, "eval_batch_size"):
        training_kwargs.update(eval_batch_size=args.eval_batch_size, learning_rate=args.learning_rate)
    if hasattr(args, "steps"):
        training_kwargs["steps"] = args.steps
    return TrainingConfig(**training_kwargs)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_ttnn_memory_config(config: MoLEConfig):
    if config.input_dim >= 256 or config.pred_len >= 336 or config.seq_len >= 336:
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.L1_MEMORY_CONFIG


def build_reference_mole(config: MoLEConfig) -> MixtureOfLinearExperts:
    return MixtureOfLinearExperts(config).eval()


def build_reference_expert(config: MoLEConfig):
    return create_reference_expert(config).eval()


def _runtime_options(config: MoLEConfig) -> TtRuntimeOptions:
    return TtRuntimeOptions(memory_config=select_ttnn_memory_config(config), dtype=ttnn.bfloat16)


def build_ttnn_mole(device, config: MoLEConfig, reference_model: MixtureOfLinearExperts) -> TtMoLE:
    return TtMoLE(config, reference_model=reference_model, device=device, runtime_options=_runtime_options(config))


def build_ttnn_expert(
    device: Any,
    config: MoLEConfig,
    reference_model: Any,
) -> TtDLinearExpert | TtRLinearExpert | TtRMLPExpert:
    expert_class = TT_EXPERT_CLASSES.get(config.base_model_type)
    if expert_class is None:
        raise ValueError(f"unsupported base_model_type: {config.base_model_type}")
    return expert_class(config, reference_model=reference_model, runtime_options=_runtime_options(config))


def open_ttnn_device():
    return ttnn.open_device(device_id=0, l1_small_size=DEFAULT_TTNN_L1_SMALL_SIZE)


def upload_mole_inputs(
    *, model, device, torch_input: torch.Tensor, torch_input_mark: torch.Tensor
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    return upload_timeseries_and_marks_to_device(
        model=model,
        device=device,
        torch_input=torch_input,
        torch_input_mark=torch_input_mark,
        memory_config=model.memory_config,
    )


def predict_mole_from_torch(
    *,
    model,
    device,
    torch_input: torch.Tensor,
    torch_input_mark: torch.Tensor | None,
    return_router_output: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if torch_input_mark is None:
        raise ValueError(
            "TT MoLE evaluation requires time marks (x_mark); the dataloader must return input marks."
        )
    tt_input, tt_marks = upload_mole_inputs(
        model=model,
        device=device,
        torch_input=torch_input,
        torch_input_mark=torch_input_mark,
    )
    if return_router_output:
        prediction, router = model.forward(tt_input, tt_marks)
        return (
            to_torch_with_cached_host(model=model, device_tensor=prediction, cache_name="mole_prediction").squeeze(0),
            to_torch_with_cached_host(model=model, device_tensor=router, cache_name="mole_router").squeeze(0),
        )
    prediction = model.forward_prediction(tt_input, tt_marks)
    return to_torch_with_cached_host(model=model, device_tensor=prediction, cache_name="mole_prediction").squeeze(0)


def predict_expert_from_torch(
    *, model, device, torch_input: torch.Tensor, torch_input_mark: torch.Tensor | None
) -> torch.Tensor:
    if torch_input_mark is None:
        raise ValueError(
            "TT expert evaluation requires time marks (x_mark); the dataloader must return input marks."
        )
    tt_input, tt_marks = upload_mole_inputs(
        model=model,
        device=device,
        torch_input=torch_input,
        torch_input_mark=torch_input_mark,
    )
    output = model.forward_prediction(tt_input, tt_marks)
    if isinstance(output, tuple):
        output = output[0]
    return to_torch_with_cached_host(model=model, device_tensor=output, cache_name="expert_prediction").squeeze(0)


def capture_trace(*, model, device, tt_input, tt_marks, prediction_only: bool = False) -> TraceState:
    forward_fn = (
        model.forward_prediction_no_trace
        if prediction_only and hasattr(model, "forward_prediction_no_trace")
        else model.forward_no_trace
        if hasattr(model, "forward_no_trace")
        else model.forward
    )
    forward_fn(tt_input, tt_marks)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    trace_output = forward_fn(tt_input, tt_marks)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    return TraceState(model=model, device=device, trace_id=trace_id, trace_output=trace_output)


def execute_trace(state: TraceState, *, blocking: bool):
    ttnn.execute_trace(state.device, state.trace_id, cq_id=0, blocking=blocking)
    return state.trace_output


def release_trace(state: TraceState) -> None:
    ttnn.release_trace(state.device, state.trace_id)


def resolve_dataset_config(
    config: MoLEConfig,
    *,
    input_dim: int,
    freq: str | None = None,
) -> MoLEConfig:
    if freq is not None:
        return replace(config, input_dim=input_dim, freq=freq)
    return replace(config, input_dim=input_dim)


def resolve_eval_input(
    config: MoLEConfig,
    *,
    batch_size: int,
    dataset_name: str,
    dataset_path: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, MoLEConfig]:
    loaders, input_dim, resolved_freq = create_real_dataset_loaders(
        dataset_name,
        dataset_path,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        batch_size=batch_size,
        eval_batch_size=batch_size,
        freq=config.freq,
    )
    batch = next(iter(loaders["test"]))
    torch_input, _, torch_input_mark, _ = unpack_batch(batch)
    return torch_input, torch_input_mark, resolve_dataset_config(config, input_dim=input_dim, freq=resolved_freq)


def unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if len(batch) == 2:
        inputs, targets = batch
        return inputs, targets, None, None
    if len(batch) == 4:
        inputs, targets, input_marks, target_marks = batch
        return inputs, targets, input_marks, target_marks
    raise ValueError(f"unsupported batch format with {len(batch)} elements")


def _forward_predictions(
    model: nn.Module, inputs: torch.Tensor, input_marks: torch.Tensor | None = None
) -> torch.Tensor:
    if input_marks is not None:
        try:
            predictions = model(inputs, input_marks)
        except TypeError:
            predictions = model(inputs)
    else:
        predictions = model(inputs)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    return predictions


def _regression_metric_totals_on_loader(
    model: nn.Module, data_loader, *, max_batches: int | None = None
) -> RegressionMetricTotals:
    totals = RegressionMetricTotals()
    with torch.no_grad():
        for batch_index, batch in enumerate(data_loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs, targets, input_marks, _ = unpack_batch(batch)
            predictions = _forward_predictions(model, inputs, input_marks)
            update_regression_metric_totals(totals, predictions, targets)

    if totals.numel == 0:
        raise ValueError("evaluation produced no batches; increase the batch limit or verify the dataset split")

    return totals


def _evaluate_model_on_dataloader(
    model: nn.Module, loaders, *, split_name: str = "test", max_batches: int | None = None
) -> dict[str, float]:
    totals = _regression_metric_totals_on_loader(model, loaders[split_name], max_batches=max_batches)
    return finalize_regression_metric_totals(totals)


def _mse_on_dataloader(model: nn.Module, loaders, *, split_name: str, max_batches: int | None = None) -> float:
    totals = _regression_metric_totals_on_loader(model, loaders[split_name], max_batches=max_batches)
    return totals.squared_error_sum / totals.numel


def train_model_on_dataloader(
    model: nn.Module,
    loaders,
    training: TrainingConfig,
    *,
    max_eval_batches: int | None = None,
    return_summary: bool = False,
) -> dict[str, float] | dict[str, object]:
    """Train the PyTorch reference model used for validation and TTNN weight export."""
    optimizer = torch.optim.Adam(model.parameters(), lr=training.learning_rate)
    model.train()
    best_state_dict = None
    best_validation_mse = None

    train_iterator = iter(loaders["train"])
    for step_index in range(training.steps):
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(loaders["train"])
            batch = next(train_iterator)

        inputs, targets, input_marks, _ = unpack_batch(batch)

        predictions = _forward_predictions(model, inputs, input_marks)

        loss = torch.nn.functional.mse_loss(predictions, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        should_validate = training.validation_interval > 0 and (
            (step_index + 1) % training.validation_interval == 0 or step_index + 1 == training.steps
        )
        if should_validate:
            model.eval()
            validation_mse = _mse_on_dataloader(
                model,
                loaders,
                split_name="val",
                max_batches=training.max_validation_batches,
            )
            if best_validation_mse is None or validation_mse < best_validation_mse:
                best_validation_mse = validation_mse
                best_state_dict = copy.deepcopy(model.state_dict())
            model.train()

    model.eval()
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    test_metrics = _evaluate_model_on_dataloader(model, loaders, split_name="test", max_batches=max_eval_batches)
    if return_summary:
        return {
            "metrics": test_metrics,
            "best_validation_mse": best_validation_mse,
            "trained_model": model,
        }
    return test_metrics
