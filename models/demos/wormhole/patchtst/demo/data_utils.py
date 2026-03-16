# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

FORECAST_DATASET_FILES = {
    "etth1": Path("ETT-small/ETTh1.csv"),
    "weather": Path("weather/weather.csv"),
    "traffic": Path("traffic/traffic.csv"),
    "electricity": Path("electricity/electricity.csv"),
    "exchange_rate": Path("exchange_rate/exchange_rate.csv"),
}


@dataclass(frozen=True)
class ArchiveDatasetSpec:
    task: Literal["classification", "regression"]
    train_file: Path
    test_file: Path


ARCHIVE_DATASET_FILES = {
    "heartbeat_cls": ArchiveDatasetSpec(
        task="classification",
        train_file=Path("Heartbeat/Heartbeat_TRAIN.ts"),
        test_file=Path("Heartbeat/Heartbeat_TEST.ts"),
    ),
    "flood_modeling1_reg": ArchiveDatasetSpec(
        task="regression",
        train_file=Path("FloodModeling1/FloodModeling1_TRAIN.ts"),
        test_file=Path("FloodModeling1/FloodModeling1_TEST.ts"),
    ),
}


@dataclass
class TaskDatasetBatch:
    past_values: torch.Tensor
    future_values: torch.Tensor | None
    target_values: torch.Tensor | None


def _load_csv_matrix(path: Path, skip_column: str = "date") -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {path}")

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")

        feature_names = [name for name in reader.fieldnames if name != skip_column]
        rows = []
        for row_idx, row in enumerate(reader, start=2):
            parsed_row: list[float] = []
            for name in feature_names:
                raw_value = row.get(name)
                if raw_value is None or raw_value == "":
                    raise ValueError(f"Malformed CSV value at row={row_idx}, column={name!r}: empty value")
                try:
                    parsed_row.append(float(raw_value))
                except ValueError as exc:
                    raise ValueError(f"Malformed CSV value at row={row_idx}, column={name!r}: {raw_value!r}") from exc
            rows.append(parsed_row)

    if not rows:
        raise ValueError(f"CSV has no data rows: {path}")

    return torch.tensor(rows, dtype=torch.float32)


def _split_bounds(length: int, split: str) -> tuple[int, int]:
    train_end = int(length * 0.7)
    val_end = int(length * 0.8)
    if split == "train":
        return 0, train_end
    if split == "val":
        return train_end, val_end
    if split == "test":
        return val_end, length
    raise ValueError(f"Unsupported split: {split}")


def load_dataset_matrix(dataset_root: Path, dataset_name: str, split: str) -> torch.Tensor:
    if dataset_name not in FORECAST_DATASET_FILES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    matrix = _load_csv_matrix(dataset_root / FORECAST_DATASET_FILES[dataset_name])
    start, end = _split_bounds(matrix.shape[0], split)
    return matrix[start:end]


def make_windows(
    matrix: torch.Tensor,
    context_length: int,
    prediction_length: int,
    max_windows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = context_length + prediction_length
    if matrix.shape[0] < total:
        raise ValueError(
            "Dataset split does not contain enough rows for requested workload. "
            f"rows={matrix.shape[0]}, required={total}, context={context_length}, prediction={prediction_length}"
        )

    starts = list(range(0, matrix.shape[0] - total + 1))[:max_windows]
    past = [matrix[start : start + context_length] for start in starts]
    future = [matrix[start + context_length : start + total] for start in starts]
    return torch.stack(past, dim=0), torch.stack(future, dim=0)


def _split_archive_series_for_forecast(
    series_tensor: torch.Tensor,
    *,
    context_length: int,
    prediction_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = int(context_length) + int(prediction_length)
    sequence_length = int(series_tensor.shape[1])
    if total <= 0:
        raise ValueError(
            f"context_length + prediction_length must be > 0 for archive-backed forecast windows, got total={total}"
        )
    if sequence_length < total:
        raise ValueError(
            "Archive dataset sequence length does not contain enough timesteps for the requested forecast window. "
            f"dataset_seq_len={sequence_length}, required={total}, context_length={context_length}, "
            f"prediction_length={prediction_length}"
        )
    return (
        series_tensor[:, :context_length, :].contiguous(),
        series_tensor[:, context_length:total, :].contiguous(),
    )


def build_observed_mask(values: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(values, dtype=torch.float32)


def _parse_ts_metadata_line(metadata: dict[str, object], line: str) -> None:
    lowered = line.lower()
    if lowered.startswith("@problemname"):
        metadata["problem_name"] = line.split(maxsplit=1)[1].strip()
    elif lowered.startswith("@timestamps"):
        metadata["timestamps"] = line.split(maxsplit=1)[1].strip().lower() == "true"
    elif lowered.startswith("@missing"):
        metadata["missing"] = line.split(maxsplit=1)[1].strip().lower() == "true"
    elif lowered.startswith("@univariate"):
        metadata["univariate"] = line.split(maxsplit=1)[1].strip().lower() == "true"
    elif lowered.startswith("@dimensions"):
        metadata["dimensions"] = int(line.split(maxsplit=1)[1].strip())
    elif lowered.startswith("@equallength"):
        metadata["equal_length"] = line.split(maxsplit=1)[1].strip().lower() == "true"
    elif lowered.startswith("@serieslength"):
        metadata["series_length"] = int(line.split(maxsplit=1)[1].strip())
    elif lowered.startswith("@classlabel"):
        parts = line.split()
        metadata["class_label"] = parts[1].lower() == "true"
        metadata["class_names"] = parts[2:]
    elif lowered.startswith("@targetlabel"):
        parts = line.split()
        metadata["target_label"] = parts[1].lower() == "true"


def _parse_ts_series_token(token: str) -> list[float]:
    values = []
    for raw_value in token.split(","):
        raw_value = raw_value.strip()
        if raw_value == "":
            continue
        if raw_value == "?":
            raise ValueError("Missing values in .ts archives are not supported in this PatchTST demo.")
        values.append(float(raw_value))
    if not values:
        raise ValueError("Encountered empty series token in .ts archive.")
    return values


def _load_archive_ts_dataset(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, Literal["classification", "regression"]]:
    if not path.exists():
        raise FileNotFoundError(f"Task dataset file not found: {path}")

    metadata: dict[str, object] = {
        "timestamps": False,
        "missing": False,
        "univariate": True,
        "equal_length": False,
        "class_label": False,
        "target_label": False,
        "class_names": [],
    }
    data_started = False
    series_rows: list[list[list[float]]] = []
    targets_raw: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if not data_started:
                if line.lower().startswith("@data"):
                    data_started = True
                    continue
                _parse_ts_metadata_line(metadata, line)
                continue

            parts = [part.strip() for part in line.split(":")]
            has_class_label = bool(metadata["class_label"])
            has_target_label = bool(metadata["target_label"])
            if has_class_label == has_target_label:
                raise ValueError(
                    "Task .ts archive must declare exactly one of @classLabel or @targetLabel for this PatchTST demo."
                )
            if len(parts) < 2:
                raise ValueError(f"Malformed .ts sample in {path}: expected data and label/target")

            targets_raw.append(parts[-1])
            dimension_tokens = parts[:-1]
            if bool(metadata["univariate"]) and len(dimension_tokens) != 1:
                raise ValueError(f"Univariate .ts sample in {path} provided {len(dimension_tokens)} dimensions")
            series_rows.append([_parse_ts_series_token(token) for token in dimension_tokens])

    if not data_started or not series_rows:
        raise ValueError(f"Task .ts archive {path} did not contain any samples after @data")
    if bool(metadata["timestamps"]):
        raise ValueError("Timestamped .ts archives are not supported in this PatchTST demo.")
    if not bool(metadata["equal_length"]):
        raise ValueError("Variable-length .ts archives are not supported in this PatchTST demo.")

    num_channels = len(series_rows[0])
    sequence_length = len(series_rows[0][0])
    for sample_idx, sample in enumerate(series_rows):
        if len(sample) != num_channels:
            raise ValueError(
                f"Inconsistent channel count in {path}: sample 0 has {num_channels}, sample {sample_idx} has {len(sample)}"
            )
        for channel_idx, channel_values in enumerate(sample):
            if len(channel_values) != sequence_length:
                raise ValueError(
                    f"Inconsistent sequence length in {path}: expected {sequence_length}, "
                    f"sample={sample_idx}, channel={channel_idx}, got {len(channel_values)}"
                )

    series_tensor = torch.tensor(series_rows, dtype=torch.float32).permute(0, 2, 1).contiguous()

    if bool(metadata["class_label"]):
        label_names = list(metadata.get("class_names", [])) or sorted(set(targets_raw))
        label_to_index = {name: idx for idx, name in enumerate(label_names)}
        try:
            targets = torch.tensor([label_to_index[label] for label in targets_raw], dtype=torch.long)
        except KeyError as error:
            raise ValueError(f"Encountered unseen class label {error.args[0]!r} in {path}") from error
        return series_tensor, targets, "classification"

    targets = torch.tensor([[float(value)] for value in targets_raw], dtype=torch.float32)
    return series_tensor, targets, "regression"


def load_task_dataset(
    dataset_root: Path,
    dataset_name: str,
    split: str,
    task: str,
    context_length: int,
    prediction_length: int,
    max_windows: int,
) -> TaskDatasetBatch:
    if task in {"forecast", "pretraining"} and dataset_name in FORECAST_DATASET_FILES:
        matrix = load_dataset_matrix(dataset_root=dataset_root, dataset_name=dataset_name, split=split)
        past_values, future_values = make_windows(
            matrix=matrix,
            context_length=context_length,
            prediction_length=prediction_length,
            max_windows=max_windows,
        )
        return TaskDatasetBatch(
            past_values=past_values,
            future_values=future_values,
            target_values=None,
        )

    if task == "forecast" and dataset_name in ARCHIVE_DATASET_FILES:
        if split not in {"train", "test"}:
            raise ValueError(f"Archive-backed forecast split must be 'train' or 'test', got {split!r}")
        spec = ARCHIVE_DATASET_FILES[dataset_name]
        archive_path = dataset_root / (spec.train_file if split == "train" else spec.test_file)
        series_tensor, _, _ = _load_archive_ts_dataset(archive_path)
        past_values, future_values = _split_archive_series_for_forecast(
            series_tensor,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        if max_windows > 0:
            past_values = past_values[:max_windows]
            future_values = future_values[:max_windows]
        return TaskDatasetBatch(
            past_values=past_values,
            future_values=future_values,
            target_values=None,
        )

    if task == "multi_task":
        if dataset_name not in ARCHIVE_DATASET_FILES or ARCHIVE_DATASET_FILES[dataset_name].task != "classification":
            raise ValueError(
                f"Task {task!r} requires a real classification dataset so the same input carries both forecast and class targets, got {dataset_name!r}"
            )
        if split not in {"train", "test"}:
            raise ValueError(f"Archive-backed multi_task split must be 'train' or 'test', got {split!r}")
        spec = ARCHIVE_DATASET_FILES[dataset_name]
        archive_path = dataset_root / (spec.train_file if split == "train" else spec.test_file)
        series_tensor, target_values, _ = _load_archive_ts_dataset(archive_path)
        past_values, future_values = _split_archive_series_for_forecast(
            series_tensor,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        if max_windows > 0:
            past_values = past_values[:max_windows]
            future_values = future_values[:max_windows]
            target_values = target_values[:max_windows]
        return TaskDatasetBatch(
            past_values=past_values,
            future_values=future_values,
            target_values=target_values,
        )

    if task not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task: {task}")
    if dataset_name not in ARCHIVE_DATASET_FILES or ARCHIVE_DATASET_FILES[dataset_name].task != task:
        raise ValueError(f"Task {task!r} requires a {task} dataset, got {dataset_name!r}")
    if split not in {"train", "test"}:
        raise ValueError(f"Task dataset split must be 'train' or 'test', got {split!r}")

    spec = ARCHIVE_DATASET_FILES[dataset_name]
    archive_path = dataset_root / (spec.train_file if split == "train" else spec.test_file)
    past_values, target_values, _ = _load_archive_ts_dataset(archive_path)
    if context_length > 0 and int(past_values.shape[1]) < context_length:
        raise ValueError(
            "Task dataset sequence length does not match the checkpoint/runtime context length. "
            f"dataset_seq_len={past_values.shape[1]}, context_length={context_length}"
        )
    if context_length > 0:
        # Classification/regression checkpoints used in the public demo may intentionally consume
        # a prefix window of a longer labeled sequence so they can share the same input contract as
        # the forecast path. Truncate explicitly here instead of requiring exact sequence-length equality.
        past_values = past_values[:, :context_length, :].contiguous()
    if max_windows > 0:
        past_values = past_values[:max_windows]
        target_values = target_values[:max_windows]
    return TaskDatasetBatch(
        past_values=past_values,
        future_values=None,
        target_values=target_values,
    )
