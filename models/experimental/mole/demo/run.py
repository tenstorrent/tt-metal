# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from argparse import ArgumentParser, Namespace
import csv
import json
import random
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
import re
from urllib.parse import quote
import urllib.request
import warnings
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import ttnn
from models.experimental.mole.reference.config import MoLEConfig
from models.experimental.mole.reference.mole import MixtureOfLinearExperts
from models.experimental.mole.tt.mole import TtMoLE
from models.experimental.mole.tt.common import (
    TtRuntimeOptions,
    release_active_traces_for_device,
    upload_timeseries_and_marks_to_device,
    to_torch_with_cached_host,
)


DEFAULT_TTNN_L1_SMALL_SIZE = 24576
DEFAULT_TTNN_TRACE_REGION_SIZE = 128 << 20
BASE_MODEL_TYPES = ("dlinear", "rlinear", "rmlp")
TRAIN_SPLIT_RATIO = 0.7
TEST_CONTEXT_SPLIT_RATIO = 0.8
CHECKPOINT_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model", "net", "weights")
CHECKPOINT_KEY_PREFIXES = ("module.", "model.", "net.")
BASELINE_DLINEAR_PATTERN = re.compile(r"(Linear_Seasonal|Linear_Trend)\.(weight|bias)")
EXPERT_STATE_KEY_PATTERN = re.compile(r"experts\.(\d+)\.(.+)")
ROUTER_KEY_ALIASES = {
    "router.0.weight": ("Linear_Temporal.0.weight", "router.0.weight"),
    "router.0.bias": ("Linear_Temporal.0.bias", "router.0.bias"),
    "router.2.weight": ("Linear_Temporal.2.weight", "router.2.weight"),
    "router.2.bias": ("Linear_Temporal.2.bias", "router.2.bias"),
}
MOLE_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_BASE_DIR = str(MOLE_DIR / "mole_checkpoints")
CHECKPOINT_DOWNLOAD_BASE_URL = "https://huggingface.co/hybelj/mole/resolve/main/mole_checkpoints"
CHECKPOINT_INDEX_FILENAME = "index.json"
CHECKPOINT_FILE = "checkpoint.pth"
DATASET_DOWNLOAD_URLS = {
    "ETTh1.csv": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh1.csv?download=true",
    "ETTh2.csv": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh2.csv?download=true",
    "ETTm1.csv": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTm1.csv?download=true",
    "ETTm2.csv": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTm2.csv?download=true",
    "ECL.csv": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv?download=true",
    "traffic.csv": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv?download=true",
    "weather.csv": "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv?download=true",
}


@dataclass(frozen=True)
class CheckpointEndpointOptions:
    checkpoint_path: str = ""
    dataset_csv_path: str = ""
    assets_root: str = CHECKPOINT_BASE_DIR
    dataset: str = "ETTh1"


@dataclass(frozen=True)
class MoleCheckpointEntry:
    dataset: str
    data_path: str
    base_model: str
    experts: int
    seq_len: int
    pred_len: int
    enc_in: int


@dataclass(frozen=True)
class MoleCheckpointResolution:
    entry: MoleCheckpointEntry
    checkpoint_path: str
    dataset_csv_path: str
    freq: str


def _safe_resolve_under(base_directory: Path, relative_path: str) -> Path:
    resolved = (base_directory / relative_path).resolve(strict=False)
    try:
        resolved.relative_to(base_directory)
    except ValueError as error:
        raise ValueError(f"path escapes base directory: {relative_path}") from error
    return resolved


def _base_model_to_cli_name(base_model: str) -> str:
    normalized = base_model.strip().lower()
    if normalized not in BASE_MODEL_TYPES:
        raise ValueError(f"unsupported base model from metadata: {base_model!r}")
    return normalized


def _base_model_to_folder_name(base_model_type: str) -> str:
    folder_map = {
        "dlinear": "DLinear",
        "rlinear": "RLinear",
        "rmlp": "RMLP",
    }
    try:
        return folder_map[base_model_type.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported base_model_type: {base_model_type}") from error


def _download_file_if_needed(path: Path, url: str) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}", file=sys.stderr)
    urllib.request.urlretrieve(url, str(path))


def load_checkpoint_index(*, assets_root: str) -> list[MoleCheckpointEntry]:
    base_directory = Path(assets_root).resolve(strict=False)
    index_path = _safe_resolve_under(base_directory, CHECKPOINT_INDEX_FILENAME)
    _download_file_if_needed(
        index_path,
        f"{CHECKPOINT_DOWNLOAD_BASE_URL}/{CHECKPOINT_INDEX_FILENAME}?download=true",
    )

    with index_path.open() as index_file:
        raw_entries = json.load(index_file)
    if not isinstance(raw_entries, list):
        raise ValueError(f"invalid checkpoint index format: expected list, got {type(raw_entries)!r}")

    entries: list[MoleCheckpointEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        entries.append(
            MoleCheckpointEntry(
                dataset=str(raw_entry["dataset"]),
                data_path=str(raw_entry["data_path"]),
                base_model=_base_model_to_cli_name(str(raw_entry["base_model"])),
                experts=int(raw_entry["experts"]),
                seq_len=int(raw_entry["seq_len"]),
                pred_len=int(raw_entry["pred_len"]),
                enc_in=int(raw_entry["enc_in"]),
            )
        )
    return entries


def _download_dataset_if_needed(dataset_csv_path: Path) -> None:
    dataset_name = dataset_csv_path.name
    dataset_url = DATASET_DOWNLOAD_URLS.get(dataset_name)
    if dataset_url is None:
        raise FileNotFoundError(
            f"dataset CSV not found: {dataset_csv_path}. Unknown dataset file for auto-download: {dataset_name}"
        )
    _download_file_if_needed(dataset_csv_path, dataset_url)


def _download_checkpoint_if_needed(checkpoint_path: Path, relative_path: str) -> None:
    quoted_path = quote(relative_path, safe="/")
    _download_file_if_needed(
        checkpoint_path,
        f"{CHECKPOINT_DOWNLOAD_BASE_URL}/{quoted_path}?download=true",
    )


def _resolve_dataset_csv_path(entry: MoleCheckpointEntry, *, assets_root: str) -> Path:
    base_directory = Path(assets_root).resolve(strict=False)
    filename = Path(entry.data_path).name
    candidate_paths = [
        _safe_resolve_under(base_directory, f"datasets/{filename}"),
        _safe_resolve_under(base_directory, filename),
    ]
    for candidate in candidate_paths:
        if candidate.is_file():
            return candidate
    canonical = candidate_paths[0]
    _download_dataset_if_needed(canonical)
    if canonical.is_file():
        return canonical
    raise FileNotFoundError(f"dataset CSV not found and could not be downloaded: {canonical}")


def _load_checkpoint_payload(checkpoint_path: Path) -> object:
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(checkpoint_path, map_location="cpu")


def _infer_checkpoint_freq(checkpoint_path: Path, *, fallback_dataset: str) -> str:
    try:
        checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
    except Exception:
        return "h" if fallback_dataset.lower().startswith("etth") else "t"
    source_state_dict = _extract_state_dict_payload(checkpoint_payload)
    router_weight = source_state_dict.get("Linear_Temporal.0.weight")
    if isinstance(router_weight, torch.Tensor) and router_weight.ndim == 2:
        return "h" if router_weight.shape[1] == 4 else "t"
    return "h" if fallback_dataset.lower().startswith("etth") else "t"


def resolve_mole_checkpoint(
    *,
    dataset: str,
    base_model_type: str,
    num_experts: int,
    assets_root: str,
) -> MoleCheckpointResolution:
    entries = load_checkpoint_index(assets_root=assets_root)
    matches = [
        entry
        for entry in entries
        if entry.dataset.lower() == dataset.lower()
        and entry.base_model == base_model_type.lower()
        and entry.experts == num_experts
    ]
    if not matches:
        raise ValueError(
            f"no checkpoint entry found for dataset={dataset}, base_model_type={base_model_type}, num_experts={num_experts}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"ambiguous checkpoint selection for dataset={dataset}, base_model_type={base_model_type}, num_experts={num_experts}"
        )
    entry = matches[0]
    model_folder = _base_model_to_folder_name(entry.base_model)
    checkpoint_relative = f"{entry.dataset}/{model_folder}/experts_{entry.experts}/{CHECKPOINT_FILE}"
    base_directory = Path(assets_root).resolve(strict=False)
    checkpoint_path = _safe_resolve_under(base_directory, checkpoint_relative)
    _download_checkpoint_if_needed(checkpoint_path, checkpoint_relative)
    dataset_csv_path = _resolve_dataset_csv_path(entry, assets_root=assets_root)
    freq = _infer_checkpoint_freq(checkpoint_path, fallback_dataset=entry.dataset)
    return MoleCheckpointResolution(
        entry=entry,
        checkpoint_path=str(checkpoint_path),
        dataset_csv_path=str(dataset_csv_path),
        freq=freq,
    )


class TimeSeriesWindowDataset(Dataset):
    """Sliding-window dataset that yields (x, y, x_mark, y_mark) tuples."""

    def __init__(self, values: torch.Tensor, time_marks: torch.Tensor, seq_len: int, pred_len: int):
        if values.ndim != 2 or time_marks.ndim != 2:
            raise ValueError("values and time_marks must be rank-2 tensors")
        if values.shape[0] != time_marks.shape[0]:
            raise ValueError("values and time_marks must have matching time dimension")
        if values.shape[0] < seq_len + pred_len:
            raise ValueError(f"dataset too small for seq_len={seq_len}, pred_len={pred_len}")
        self.values = values
        self.time_marks = time_marks
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return self.values.shape[0] - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_end = index + self.seq_len
        target_end = input_end + self.pred_len
        return (
            self.values[index:input_end],
            self.values[input_end:target_end],
            self.time_marks[index:input_end],
            self.time_marks[input_end:target_end],
        )


def add_dataset_arguments(
    parser: ArgumentParser,
) -> None:
    """Register dataset-related command-line arguments."""
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=CHECKPOINT_BASE_DIR,
        help="MoLE assets root directory containing index.json and checkpoint folders",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ETTh1",
        help="Dataset name from checkpoint index (e.g. ETTh1, ETTh2, ECL, traffic, weather)",
    )


def add_model_arguments(
    parser: ArgumentParser,
    *,
    include_input_dim: bool = False,
) -> None:
    """Register model-related command-line arguments."""
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


def model_config_from_args(args: Namespace) -> MoLEConfig:
    """Build a MoLEConfig from parsed CLI arguments."""
    return MoLEConfig(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        input_dim=args.input_dim if hasattr(args, "input_dim") else 7,
        base_model_type=args.base_model_type,
        num_experts=args.num_experts,
        freq=args.freq,
    )


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_ttnn_memory_config(config: MoLEConfig) -> Any:
    """Pick TTNN memory config based on model dimensions."""
    if config.input_dim >= 256 or config.pred_len >= 336 or config.seq_len >= 336:
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.L1_MEMORY_CONFIG


def select_ttnn_activation_memory_config(config: MoLEConfig) -> Any:
    if config.input_dim >= (512 if config.base_model_type == "dlinear" else 256):
        return ttnn.DRAM_MEMORY_CONFIG
    return ttnn.L1_MEMORY_CONFIG


def _parse_timestamp(raw_value: str) -> datetime:
    """Parse a timestamp used by common forecasting CSV datasets."""
    for parser in (
        datetime.fromisoformat,
        lambda value: datetime.strptime(value, "%Y-%m-%d %H:%M:%S"),
    ):
        try:
            return parser(raw_value)
        except ValueError:
            continue
    raise ValueError(f"unsupported timestamp format: {raw_value!r}")


def _build_time_marks(date_values: list[str], freq: str) -> torch.Tensor:
    timestamps = [_parse_timestamp(value) for value in date_values]
    if freq.lower().endswith("h"):
        rows = [[ts.month, ts.day, ts.weekday(), ts.hour] for ts in timestamps]
    else:
        rows = [[ts.month, ts.day, ts.weekday(), ts.hour, ts.minute // 15] for ts in timestamps]
    return torch.tensor(rows, dtype=torch.float32)


def _load_local_csv(dataset_csv_path: Path, freq: str) -> tuple[torch.Tensor, torch.Tensor]:
    with dataset_csv_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        if not header or header[0].lower() != "date":
            raise ValueError(f"expected first CSV column to be 'date': {dataset_csv_path}")

        date_values: list[str] = []
        values_list: list[list[float]] = []
        for row_index, row in enumerate(reader, start=2):
            if len(row) < 2:
                raise ValueError(f"row {row_index} in {dataset_csv_path} has fewer than 2 columns")
            date_values.append(row[0])
            try:
                values_list.append([float(value) for value in row[1:]])
            except ValueError as error:
                raise ValueError(f"row {row_index} in {dataset_csv_path} has non-numeric feature values") from error

    if not values_list:
        raise ValueError(f"dataset is empty: {dataset_csv_path}")

    try:
        marks = _build_time_marks(date_values, freq)
    except ValueError as error:
        raise ValueError(f"failed to parse timestamp column in {dataset_csv_path}") from error

    values = torch.tensor(values_list, dtype=torch.float32)
    return values, marks


def _slice_with_context(values: torch.Tensor, start: int, end: int, seq_len: int) -> torch.Tensor:
    """Slice [start, end) with a left context window of up to seq_len rows."""
    return values[max(0, start - seq_len) : end]


def _compute_split_indices(row_count: int) -> tuple[int, int, int]:
    """Return normalization end, test-context start, and evaluation end indices."""
    normalization_end = int(row_count * TRAIN_SPLIT_RATIO)
    context_end = int(row_count * TEST_CONTEXT_SPLIT_RATIO)
    evaluation_end = row_count
    return normalization_end, context_end, evaluation_end


def _normalize_values(values: torch.Tensor, normalization_end: int) -> torch.Tensor:
    """Standardize feature columns using the train/normalization prefix only."""
    normalization_values = values[:normalization_end]
    mean = normalization_values.mean(dim=0)
    std = normalization_values.std(dim=0, unbiased=False)
    std = torch.where(std == 0, torch.ones_like(std), std)
    return (values - mean) / std


def create_local_dataset_loaders(
    dataset_csv_path: str,
    *,
    seq_len: int,
    pred_len: int,
    eval_batch_size: int,
    freq: str,
) -> tuple[dict[str, DataLoader], int, str]:
    """Load a local CSV dataset and create evaluation dataloaders."""
    values, marks = _load_local_csv(Path(dataset_csv_path), freq)

    row_count = values.shape[0]
    normalization_end, context_end, evaluation_end = _compute_split_indices(row_count)

    values = values[:evaluation_end]
    marks = marks[:evaluation_end]
    normalized_values = _normalize_values(values, normalization_end)

    test_values = _slice_with_context(normalized_values, context_end, evaluation_end, seq_len)
    test_marks = _slice_with_context(marks, context_end, evaluation_end, seq_len)

    batch_size = eval_batch_size if eval_batch_size > 0 else 1
    loaders = {
        "test": DataLoader(
            TimeSeriesWindowDataset(test_values, test_marks, seq_len, pred_len),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
    }
    return loaders, int(values.shape[1]), freq


def _extract_state_dict_payload(raw_checkpoint: object) -> dict[str, torch.Tensor]:
    """Extract a tensor-only state dict from common checkpoint payload formats."""
    if isinstance(raw_checkpoint, dict):
        for key in CHECKPOINT_STATE_DICT_KEYS:
            candidate = raw_checkpoint.get(key)
            if isinstance(candidate, dict):
                return {str(name): value for name, value in candidate.items() if isinstance(value, torch.Tensor)}
        return {str(name): value for name, value in raw_checkpoint.items() if isinstance(value, torch.Tensor)}
    raise TypeError(f"unsupported checkpoint payload type: {type(raw_checkpoint)!r}")


def _build_source_key_lookup(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Build key lookup that includes progressively de-prefixed aliases."""
    lookup: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        lookup.setdefault(key, value)
        trimmed_key = key
        while True:
            matched_prefix = next(
                (prefix for prefix in CHECKPOINT_KEY_PREFIXES if trimmed_key.startswith(prefix)), None
            )
            if matched_prefix is None:
                break
            trimmed_key = trimmed_key[len(matched_prefix) :]
            lookup.setdefault(trimmed_key, value)
    return lookup


def _first_tensor_with_shape(
    source_lookup: dict[str, torch.Tensor],
    keys: tuple[str, ...],
    target_shape: torch.Size,
    *,
    target_key: str | None = None,
) -> torch.Tensor | None:
    matches: list[tuple[str, torch.Tensor]] = []
    for key in keys:
        candidate = source_lookup.get(key)
        if isinstance(candidate, torch.Tensor) and candidate.shape == target_shape:
            matches.append((key, candidate))
    if len(matches) > 1:
        matched_keys = [m[0] for m in matches]
        warnings.warn(
            f"Ambiguous checkpoint key resolution for {target_key!r}: "
            f"{len(matches)} candidates match shape {target_shape}: {matched_keys}. "
            f"Using first match: {matches[0][0]!r}",
            stacklevel=3,
        )
    if matches:
        return matches[0][1]
    return None


def _slice_tensor_for_expert(
    source: torch.Tensor, *, expert_index: int, target_shape: torch.Size
) -> torch.Tensor | None:
    if source.ndim != len(target_shape):
        return None
    if source.ndim == 0:
        return None
    if source.shape[1:] != target_shape[1:]:
        return None
    # Exact shape match → replicate single-expert checkpoint across all experts
    if source.shape == target_shape:
        return source
    chunk_size = target_shape[0]
    if chunk_size <= 0 or source.shape[0] < chunk_size:
        return None
    if source.shape[0] % chunk_size != 0:
        return None
    num_chunks = source.shape[0] // chunk_size
    if expert_index >= num_chunks:
        return None
    start = expert_index * chunk_size
    end = start + chunk_size
    return source[start:end]


def _resolve_candidate_tensor(
    *,
    target_key: str,
    target_shape: torch.Size,
    source_lookup: dict[str, torch.Tensor],
    base_model_type: str,
) -> torch.Tensor | None:
    direct = _first_tensor_with_shape(source_lookup, (target_key,), target_shape, target_key=target_key)
    if direct is not None:
        return direct

    if base_model_type == "dlinear":
        baseline_packed_match = BASELINE_DLINEAR_PATTERN.fullmatch(target_key)
        if baseline_packed_match is not None:
            baseline = _first_tensor_with_shape(
                source_lookup,
                (
                    target_key,
                    f"experts.0.{target_key}",
                    f"models.0.{target_key}",
                ),
                target_shape,
                target_key=target_key,
            )
            if baseline is not None:
                return baseline

            for candidate_key in (target_key, f"model.{target_key}", f"net.{target_key}"):
                packed_tensor = source_lookup.get(candidate_key)
                if not isinstance(packed_tensor, torch.Tensor):
                    continue
                sliced = _slice_tensor_for_expert(
                    packed_tensor,
                    expert_index=0,
                    target_shape=target_shape,
                )
                if sliced is not None and sliced.shape == target_shape:
                    return sliced

    if target_key.startswith("router."):
        return _first_tensor_with_shape(
            source_lookup,
            ROUTER_KEY_ALIASES.get(target_key, ()),
            target_shape,
            target_key=target_key,
        )

    expert_match = EXPERT_STATE_KEY_PATTERN.fullmatch(target_key)
    if expert_match is None:
        return None

    expert_index = int(expert_match.group(1))
    sub_key = expert_match.group(2)
    direct = _first_tensor_with_shape(
        source_lookup,
        (
            f"experts.{expert_index}.{sub_key}",
            f"models.{expert_index}.{sub_key}",
            f"model.models.{expert_index}.{sub_key}",
        ),
        target_shape,
        target_key=target_key,
    )
    if direct is not None:
        return direct

    if base_model_type != "dlinear":
        # For rlinear/rmlp: try the sub_key directly as a packed/shared tensor
        for candidate_key in (sub_key, f"model.{sub_key}", f"net.{sub_key}"):
            packed_tensor = source_lookup.get(candidate_key)
            if not isinstance(packed_tensor, torch.Tensor):
                continue
            if packed_tensor.shape == target_shape:
                return packed_tensor
            sliced = _slice_tensor_for_expert(
                packed_tensor,
                expert_index=expert_index,
                target_shape=target_shape,
            )
            if sliced is not None and sliced.shape == target_shape:
                return sliced
        return None

    packed_match = BASELINE_DLINEAR_PATTERN.fullmatch(sub_key)
    if packed_match is None:
        # Not a DLinear seasonal/trend key — try sub_key directly (covers
        # rev.affine_*, temporal.*, Linear.*, etc.)
        for candidate_key in (sub_key, f"model.{sub_key}", f"net.{sub_key}"):
            packed_tensor = source_lookup.get(candidate_key)
            if not isinstance(packed_tensor, torch.Tensor):
                continue
            if packed_tensor.shape == target_shape:
                return packed_tensor
            sliced = _slice_tensor_for_expert(
                packed_tensor,
                expert_index=expert_index,
                target_shape=target_shape,
            )
            if sliced is not None and sliced.shape == target_shape:
                return sliced
        return None

    packed_key = f"{packed_match.group(1)}.{packed_match.group(2)}"
    for candidate_key in (packed_key, f"model.{packed_key}", f"net.{packed_key}"):
        packed_tensor = source_lookup.get(candidate_key)
        if not isinstance(packed_tensor, torch.Tensor):
            continue
        sliced = _slice_tensor_for_expert(
            packed_tensor,
            expert_index=expert_index,
            target_shape=target_shape,
        )
        if sliced is not None and sliced.shape == target_shape:
            return sliced

    return None


def load_reference_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> dict[str, object]:
    """Load a source checkpoint into a reference model with key-shape remapping."""
    resolved_path = Path(checkpoint_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {resolved_path}")

    source_state_dict = _extract_state_dict_payload(_load_checkpoint_payload(resolved_path))
    source_lookup = _build_source_key_lookup(source_state_dict)

    target_state_dict = model.state_dict()
    converted: dict[str, torch.Tensor] = {}
    for target_key, target_tensor in target_state_dict.items():
        resolved_tensor = _resolve_candidate_tensor(
            target_key=target_key,
            target_shape=target_tensor.shape,
            source_lookup=source_lookup,
            base_model_type=getattr(getattr(model, "config", None), "base_model_type", ""),
        )
        if resolved_tensor is not None:
            converted[target_key] = resolved_tensor

    load_result = model.load_state_dict(converted, strict=False)
    missing_keys = list(load_result.missing_keys)

    model_num_experts = getattr(getattr(model, "config", None), "num_experts", None)
    if model_num_experts == 1:
        missing_keys = [key for key in missing_keys if not key.startswith("router.")]

    if missing_keys:
        missing_keys_sorted = sorted(missing_keys)
        source_keys_sorted = sorted(source_state_dict.keys())
        source_sample = source_keys_sorted[:40]
        missing_sample = missing_keys_sorted[:30]
        raise ValueError(
            "checkpoint could not fully initialize model. "
            f"missing_key_count={len(missing_keys_sorted)} "
            f"loaded_key_count={len(converted)} "
            f"source_key_count={len(source_state_dict)}\n"
            f"missing_keys_sample={missing_sample}\n"
            f"source_keys_sample={source_sample}"
        )

    return {
        "checkpoint_path": str(resolved_path),
        "loaded_keys": sorted(converted.keys()),
        "source_key_count": len(source_state_dict),
    }


def _runtime_options(config: MoLEConfig) -> TtRuntimeOptions:
    memory_config = select_ttnn_memory_config(config)
    activation_memory_config = select_ttnn_activation_memory_config(config)
    return TtRuntimeOptions(
        memory_config=memory_config,
        activation_memory_config=activation_memory_config,
        dtype=ttnn.bfloat16,
    )


def config_from_checkpoint_resolution(
    resolution: MoleCheckpointResolution,
    *,
    base_model_type: str,
    num_experts: int,
) -> MoLEConfig:
    return MoLEConfig(
        seq_len=resolution.entry.seq_len,
        pred_len=resolution.entry.pred_len,
        input_dim=resolution.entry.enc_in,
        base_model_type=base_model_type,
        num_experts=num_experts,
        freq=resolution.freq,
    )


def build_ttnn_mole_from_checkpoint(
    device: Any,
    config: MoLEConfig,
    checkpoint_path: str | Path,
) -> TtMoLE:
    reference_model = MixtureOfLinearExperts(config).eval()
    load_reference_checkpoint(reference_model, checkpoint_path)
    checkpoint_state_dict = {key: value.detach().clone() for key, value in reference_model.state_dict().items()}
    return TtMoLE(
        config,
        checkpoint_state_dict=checkpoint_state_dict,
        device=device,
        runtime_options=_runtime_options(config),
    )


def open_ttnn_device() -> Any:
    """Open a TTNN device with MoLE demo defaults."""
    return ttnn.open_device(
        device_id=0,
        l1_small_size=DEFAULT_TTNN_L1_SMALL_SIZE,
        trace_region_size=DEFAULT_TTNN_TRACE_REGION_SIZE,
    )


def close_ttnn_device(device: Any) -> None:
    """Close TTNN device, releasing cached MoLE traces first."""
    if device is None:
        return
    try:
        release_active_traces_for_device(device=device)
    except Exception:
        pass
    try:
        ttnn.close_device(device)
    except Exception as error:  # noqa: BLE001
        error_message = str(error)
        if (
            "SubDeviceManagerTracker is not initialized on MeshDevice" in error_message
            or "remote-only MeshDevices" in error_message
        ):
            return
        raise


def upload_mole_inputs(
    *, model: TtMoLE, device: Any, torch_input: torch.Tensor, torch_input_mark: torch.Tensor
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Upload host-side MoLE inputs and time marks to TT device tensors."""
    return upload_timeseries_and_marks_to_device(
        model=model,
        device=device,
        torch_input=torch_input,
        torch_input_mark=torch_input_mark,
        memory_config=model.memory_config,
    )


def predict_mole_from_torch(
    *,
    model: TtMoLE,
    device: Any,
    torch_input: torch.Tensor,
    torch_input_mark: torch.Tensor | None,
    return_router_output: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Run TT MoLE prediction from torch inputs, optionally returning router output.

    This endpoint intentionally uses non-trace forwards. The evaluation flow uploads
    fresh batch tensors every iteration, and enabling trace replay here can trigger
    device trace-capture write/assert logs depending on allocator state.
    """
    if torch_input_mark is None:
        raise ValueError("TT MoLE evaluation requires time marks (x_mark); the dataloader must return input marks.")
    tt_input, tt_marks = upload_mole_inputs(
        model=model,
        device=device,
        torch_input=torch_input,
        torch_input_mark=torch_input_mark,
    )
    if return_router_output:
        prediction, router = model.forward_no_trace(tt_input, tt_marks)
        return (
            to_torch_with_cached_host(model=model, device_tensor=prediction, cache_name="mole_prediction").squeeze(0),
            to_torch_with_cached_host(model=model, device_tensor=router, cache_name="mole_router").squeeze(0),
        )
    prediction = model.forward_prediction_no_trace(tt_input, tt_marks)
    return to_torch_with_cached_host(model=model, device_tensor=prediction, cache_name="mole_prediction").squeeze(0)


def unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Normalize dataset batch tuples into (x, y, x_mark, y_mark)."""
    if len(batch) == 2:
        inputs, targets = batch
        return inputs, targets, None, None
    if len(batch) == 4:
        inputs, targets, input_marks, target_marks = batch
        return inputs, targets, input_marks, target_marks
    raise ValueError(f"unsupported batch format with {len(batch)} elements")


def evaluate_mole_predictions(
    *,
    endpoint: "CheckpointInferenceEndpoint",
    model: TtMoLE,
    loader: DataLoader,
    return_router_output: bool,
) -> dict[str, object]:
    """Run full-loader inference and return aggregate error metrics."""
    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_count = 0
    prediction_shape: tuple[int, ...] | None = None
    target_shape: tuple[int, ...] | None = None
    router_shape: tuple[int, ...] | None = None

    for batch in loader:
        torch_input, torch_target, torch_input_mark, _ = unpack_batch(batch)
        if torch_input_mark is None:
            raise ValueError("run inference requires x_mark time features")

        output = endpoint.predict_from_torch(
            model=model,
            torch_input=torch_input,
            torch_input_mark=torch_input_mark,
            return_router_output=return_router_output,
        )

        if isinstance(output, tuple):
            prediction, router = output
            if router_shape is None:
                router_shape = tuple(router.shape)
        else:
            prediction = output

        if prediction_shape is None:
            prediction_shape = tuple(prediction.shape)
        if target_shape is None:
            target_shape = tuple(torch_target.shape)

        prediction = prediction.to(dtype=torch.float32)
        target = torch_target.to(dtype=torch.float32)
        diff = prediction - target
        total_squared_error += float(torch.sum(diff * diff).item())
        total_absolute_error += float(torch.sum(torch.abs(diff)).item())
        total_count += int(diff.numel())

    if total_count <= 0:
        raise ValueError("evaluation loader produced zero elements")

    return {
        "prediction_shape": prediction_shape,
        "target_shape": target_shape,
        "router_shape": router_shape,
        "mse": total_squared_error / total_count,
        "mae": total_absolute_error / total_count,
        "num_points": total_count,
    }


class CheckpointInferenceEndpoint:
    """Owns TT device and checkpoint options for inference helper methods."""

    def __init__(self, *, device: Any, options: CheckpointEndpointOptions):
        self.device = device
        self.options = options

    def resolve_dataset(
        self,
        model_config: MoLEConfig,
        *,
        eval_batch_size: int,
    ) -> tuple[dict[str, object], MoLEConfig]:
        resolved = resolve_mole_checkpoint(
            dataset=self.options.dataset,
            base_model_type=model_config.base_model_type,
            num_experts=model_config.num_experts or 0,
            assets_root=self.options.assets_root,
        )
        loaders, input_dim, resolved_freq = create_local_dataset_loaders(
            resolved.dataset_csv_path,
            seq_len=resolved.entry.seq_len,
            pred_len=resolved.entry.pred_len,
            eval_batch_size=eval_batch_size,
            freq=resolved.freq,
        )
        next_config = replace(
            model_config,
            input_dim=input_dim,
            freq=resolved_freq,
            seq_len=resolved.entry.seq_len,
            pred_len=resolved.entry.pred_len,
        )
        self.options = replace(
            self.options,
            checkpoint_path=resolved.checkpoint_path,
            dataset_csv_path=resolved.dataset_csv_path,
        )
        if resolved.entry.enc_in != input_dim:
            raise ValueError(
                f"dataset feature count mismatch for {resolved.entry.dataset}: metadata enc_in={resolved.entry.enc_in}, csv_features={input_dim}"
            )
        return loaders, next_config

    def build_mole_ttnn(self, model_config: MoLEConfig) -> TtMoLE:
        return build_ttnn_mole_from_checkpoint(self.device, model_config, self.options.checkpoint_path)

    def predict_from_torch(
        self,
        *,
        model: TtMoLE,
        torch_input: torch.Tensor,
        torch_input_mark: torch.Tensor,
        return_router_output: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return predict_mole_from_torch(
            model=model,
            device=self.device,
            torch_input=torch_input,
            torch_input_mark=torch_input_mark,
            return_router_output=return_router_output,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Checkpoint inference runner for MoLE TTNN models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_dataset_arguments(parser)
    add_model_arguments(parser)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--return-router-output",
        action="store_true",
        help="Return and print router output shape in addition to prediction shape",
    )
    args = parser.parse_args()

    set_random_seed(args.seed)
    resolution = resolve_mole_checkpoint(
        dataset=args.dataset,
        base_model_type=args.base_model_type,
        num_experts=args.num_experts,
        assets_root=args.dataset_dir,
    )
    config = config_from_checkpoint_resolution(
        resolution,
        base_model_type=args.base_model_type,
        num_experts=args.num_experts,
    )

    device = open_ttnn_device()
    try:
        endpoint = CheckpointInferenceEndpoint(
            device=device,
            options=CheckpointEndpointOptions(
                checkpoint_path=resolution.checkpoint_path,
                dataset_csv_path=resolution.dataset_csv_path,
                assets_root=args.dataset_dir,
                dataset=args.dataset,
            ),
        )
        loaders, config = endpoint.resolve_dataset(
            config,
            eval_batch_size=args.batch_size,
        )
        model = endpoint.build_mole_ttnn(config)

        metrics = evaluate_mole_predictions(
            endpoint=endpoint,
            model=model,
            loader=loaders["test"],
            return_router_output=args.return_router_output,
        )

        prediction_shape = metrics["prediction_shape"]
        router_shape = metrics["router_shape"]
        target_shape = metrics["target_shape"]
        if prediction_shape is not None:
            if router_shape is not None:
                print(f"prediction_shape={prediction_shape} router_shape={router_shape}")
            else:
                print(f"prediction_shape={prediction_shape}")
        if target_shape is not None:
            print(f"target_shape={target_shape}")
        print(f"checkpoint_path={endpoint.options.checkpoint_path}")
        print(f"dataset_csv_path={endpoint.options.dataset_csv_path}")
        print(f"mse={metrics['mse']:.6f} mae={metrics['mae']:.6f} num_points={metrics['num_points']}")
    finally:
        close_ttnn_device(device)


if __name__ == "__main__":
    main()
