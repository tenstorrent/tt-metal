# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import csv
import http.client
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import torch
from torch.utils.data import DataLoader, Dataset


ETT_HOURLY_SPLITS = (12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24)
ETT_MINUTE_SPLITS = tuple(split * 4 for split in ETT_HOURLY_SPLITS)
DEFAULT_DATASET_CACHE_DIR = Path.home() / ".cache" / "tt-metal" / "mole" / "datasets"
TSLIB_DATASET_BASE_URL = "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main"
ALLOWED_DATASET_DOWNLOAD_HOSTS = {"huggingface.co", "cdn-lfs.hf.co"}
TSLIB_DATASET_SOURCES = {
    "etth1": {
        "relative_path": Path("ETT-small") / "ETTh1.csv",
    },
    "etth2": {
        "relative_path": Path("ETT-small") / "ETTh2.csv",
    },
    "ettm1": {
        "relative_path": Path("ETT-small") / "ETTm1.csv",
    },
    "ettm2": {
        "relative_path": Path("ETT-small") / "ETTm2.csv",
    },
    "weather": {
        "relative_path": Path("weather") / "weather.csv",
    },
    "electricity": {
        "relative_path": Path("electricity") / "electricity.csv",
    },
    "traffic": {
        "relative_path": Path("traffic") / "traffic.csv",
    },
}


@dataclass(frozen=True)
class DatasetSplitConfig:
    train_end: int
    val_end: int
    test_end: int


@dataclass
class RegressionMetricTotals:
    squared_error_sum: float = 0.0
    numel: int = 0


def update_regression_metric_totals(
    totals: RegressionMetricTotals,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> RegressionMetricTotals:
    difference = predictions - targets
    totals.squared_error_sum += difference.pow(2).sum().item()
    totals.numel += difference.numel()
    return totals


def finalize_regression_metric_totals(totals: RegressionMetricTotals) -> dict[str, float]:
    if totals.numel == 0:
        raise ValueError("regression metrics require at least one prediction element")
    return {
        "mse": totals.squared_error_sum / totals.numel,
    }


def _validate_dataset_url(dataset_url: str) -> str:
    parsed = urlsplit(dataset_url)
    if parsed.scheme.lower() != "https":
        raise ValueError(f"dataset download requires https URL, got: {dataset_url}")
    hostname = (parsed.hostname or "").lower()
    if hostname not in ALLOWED_DATASET_DOWNLOAD_HOSTS:
        raise ValueError(f"dataset download host is not allowed: {hostname}")
    return dataset_url


def _download_https_file(dataset_url: str, destination: Path, *, max_redirects: int = 3) -> None:
    current_url = _validate_dataset_url(dataset_url)
    for _ in range(max_redirects + 1):
        parsed = urlsplit(current_url)
        path_and_query = parsed.path or "/"
        if parsed.query:
            path_and_query = f"{path_and_query}?{parsed.query}"

        connection = http.client.HTTPSConnection(parsed.netloc, timeout=60)
        try:
            connection.request("GET", path_and_query)
            response = connection.getresponse()

            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location:
                    raise RuntimeError(f"redirect response missing Location header for {current_url}")
                current_url = _validate_dataset_url(urljoin(current_url, location))
                continue

            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"download failed with HTTP {response.status} for {current_url}")

            with destination.open("wb") as output_file:
                output_file.write(response.read())
            return
        finally:
            connection.close()

    raise RuntimeError(f"too many redirects while downloading dataset from {dataset_url}")


class TimeSeriesWindowDataset(Dataset):
    def __init__(self, values: torch.Tensor, time_marks: torch.Tensor, seq_len: int, pred_len: int):
        if values.ndim != 2:
            raise ValueError(f"expected [time, features], got {tuple(values.shape)}")
        if time_marks.ndim != 2:
            raise ValueError(f"expected [time, mark_features], got {tuple(time_marks.shape)}")
        if values.shape[0] != time_marks.shape[0]:
            raise ValueError(
                f"values and time_marks must have matching time dimension, got {values.shape[0]} vs {time_marks.shape[0]}"
            )
        if values.shape[0] < seq_len + pred_len:
            raise ValueError(
                f"dataset length {values.shape[0]} is too small for seq_len={seq_len}, pred_len={pred_len}"
            )
        self.values = values
        self.time_marks = time_marks
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return self.values.shape[0] - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input_start = index
        input_end = input_start + self.seq_len
        target_end = input_end + self.pred_len
        return (
            self.values[input_start:input_end],
            self.values[input_end:target_end],
            self.time_marks[input_start:input_end],
            self.time_marks[input_end:target_end],
        )


def resolve_tslib_dataset_path(
    dataset_name: str,
    dataset_path: str | Path | None = None,
    *,
    cache_dir: str | Path | None = None,
    auto_download: bool = True,
) -> Path:
    if dataset_path is not None:
        candidate = Path(dataset_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"dataset not found: {candidate}")

    normalized_name = dataset_name.lower()
    dataset_source = TSLIB_DATASET_SOURCES.get(normalized_name)
    if dataset_source is None:
        raise ValueError(
            f"dataset_path is required for unsupported dataset '{dataset_name}'. Supported auto-download datasets: {sorted(TSLIB_DATASET_SOURCES)}"
        )

    base_cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_DATASET_CACHE_DIR
    resolved_path = base_cache_dir / dataset_source["relative_path"]
    if resolved_path.exists():
        return resolved_path
    if not auto_download:
        raise FileNotFoundError(f"dataset not found in cache and auto_download is disabled: {resolved_path}")

    relative_path = dataset_source.get("relative_path")
    if not isinstance(relative_path, Path):
        raise FileNotFoundError(
            f"dataset '{dataset_name}' is registered but no relative path is configured; place the file at {resolved_path} or pass --dataset-path"
        )
    dataset_url = f"{TSLIB_DATASET_BASE_URL}/{relative_path.as_posix()}?download=true"

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved_path.with_suffix(resolved_path.suffix + ".tmp")
    try:
        _download_https_file(dataset_url, temporary_path)
        os.replace(temporary_path, resolved_path)
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download dataset '{dataset_name}' from {dataset_url}") from error
    return resolved_path


def infer_split_config(dataset_name: str, row_count: int) -> DatasetSplitConfig:
    normalized_name = dataset_name.lower()
    if normalized_name in {"etth1", "etth2"}:
        train_size, val_size, test_size = ETT_HOURLY_SPLITS
    elif normalized_name in {"ettm1", "ettm2"}:
        train_size, val_size, test_size = ETT_MINUTE_SPLITS
    else:
        train_size = int(row_count * 0.7)
        test_size = int(row_count * 0.2)
        val_size = row_count - train_size - test_size

    total_required = train_size + val_size + test_size
    if total_required > row_count:
        train_size = int(row_count * 0.7)
        test_size = int(row_count * 0.2)
        val_size = row_count - train_size - test_size

    return DatasetSplitConfig(
        train_end=train_size,
        val_end=train_size + val_size,
        test_end=train_size + val_size + test_size,
    )


def _slice_split_with_context(values: torch.Tensor, start_index: int, end_index: int, seq_len: int) -> torch.Tensor:
    context_start = max(0, start_index - seq_len)
    return values[context_start:end_index]


def _parse_timestamp(raw_value: str) -> datetime:
    try:
        return datetime.fromisoformat(raw_value)
    except ValueError:
        return datetime.strptime(raw_value, "%Y-%m-%d %H:%M:%S")


def _build_time_marks(date_values: list[str], freq: str) -> torch.Tensor:
    timestamps = [_parse_timestamp(value) for value in date_values]
    if freq.lower().endswith("h"):
        mark_rows = [[ts.month, ts.day, ts.weekday(), ts.hour] for ts in timestamps]
    else:
        mark_rows = [[ts.month, ts.day, ts.weekday(), ts.hour, ts.minute // 15] for ts in timestamps]
    return torch.tensor(mark_rows, dtype=torch.float32)


def load_tslib_csv(
    dataset_path: str | Path,
    *,
    freq: str = "h",
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    with dataset_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)
        rows = list(reader)

    if not rows:
        raise ValueError(f"dataset is empty: {dataset_path}")

    has_date_column = bool(header) and header[0].lower() == "date"
    if not has_date_column:
        raise ValueError(f"expected first CSV column to be 'date' for time-feature extraction: {dataset_path}")

    date_values = [row[0] for row in rows]
    values = torch.tensor([[float(value) for value in row[1:]] for row in rows], dtype=torch.float32)
    time_marks = _build_time_marks(date_values, freq)
    return values, time_marks


def create_real_dataset_loaders(
    dataset_name: str,
    dataset_path: str | Path | None,
    *,
    seq_len: int,
    pred_len: int,
    batch_size: int,
    eval_batch_size: int,
    freq: str = "h",
) -> tuple[dict[str, DataLoader], int]:
    resolved_path = resolve_tslib_dataset_path(dataset_name, dataset_path)
    values, time_marks = load_tslib_csv(resolved_path, freq=freq)

    split = infer_split_config(dataset_name, values.shape[0])
    values = values[: split.test_end]
    time_marks = time_marks[: split.test_end]

    train_values = values[: split.train_end]

    mean = train_values.mean(dim=0)
    train_std = train_values.std(dim=0, unbiased=False)
    std = torch.where(train_std == 0, torch.ones_like(mean), train_std)
    normalized_values = (values - mean) / std

    train_values = normalized_values[: split.train_end]
    val_values = _slice_split_with_context(normalized_values, split.train_end, split.val_end, seq_len)
    test_values = _slice_split_with_context(normalized_values, split.val_end, split.test_end, seq_len)
    train_marks = time_marks[: split.train_end]
    val_marks = _slice_split_with_context(time_marks, split.train_end, split.val_end, seq_len)
    test_marks = _slice_split_with_context(time_marks, split.val_end, split.test_end, seq_len)

    loaders = {
        "train": DataLoader(
            TimeSeriesWindowDataset(train_values, train_marks, seq_len, pred_len),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        ),
        "val": DataLoader(
            TimeSeriesWindowDataset(val_values, val_marks, seq_len, pred_len),
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
        ),
        "test": DataLoader(
            TimeSeriesWindowDataset(test_values, test_marks, seq_len, pred_len),
            batch_size=eval_batch_size,
            shuffle=False,
            drop_last=False,
        ),
    }
    return loaders, values.shape[1]
