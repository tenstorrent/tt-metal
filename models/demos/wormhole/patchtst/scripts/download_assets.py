# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import time
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

from models.demos.wormhole.patchtst.config import PatchTSTDemoConfig
from models.demos.wormhole.patchtst.demo.data_utils import ARCHIVE_DATASET_FILES, FORECAST_DATASET_FILES

DATASET_URLS = {
    "etth1": (
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/ETT-small/ETTh1.csv",
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    ),
    "weather": ("https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/weather/weather.csv", None),
    "traffic": ("https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/traffic/traffic.csv", None),
    "electricity": (
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/electricity/electricity.csv",
        None,
    ),
    "exchange_rate": (
        "https://huggingface.co/datasets/thuml/Time-Series-Library/resolve/main/exchange_rate/exchange_rate.csv",
        None,
    ),
    "heartbeat_cls": ("https://www.timeseriesclassification.com/aeon-toolkit/Heartbeat.zip", None),
    "flood_modeling1_reg": ("https://www.timeseriesclassification.com/aeon-toolkit/FloodModeling1.zip", None),
}


def _download(url: str, destination: Path, timeout_seconds: float, retries: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(max(retries, 1)):
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                destination.write_bytes(response.read())
            return
        except Exception as error:  # pragma: no cover
            last_error = error
            if attempt + 1 < max(retries, 1):
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"Failed to download {url} after {max(retries, 1)} attempts: {last_error}") from last_error


def _download_with_fallback(
    primary_url: str,
    fallback_url: str,
    destination: Path,
    timeout_seconds: float,
    retries: int,
) -> None:
    try:
        _download(primary_url, destination, timeout_seconds=timeout_seconds, retries=retries)
    except Exception as primary_error:
        print(f"Primary download failed for {destination.name}: {primary_error}; trying fallback URL.")
        _download(fallback_url, destination, timeout_seconds=timeout_seconds, retries=retries)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_archive_dataset(dataset_name: str, archive_path: Path, dataset_root: Path) -> list[Path]:
    spec = ARCHIVE_DATASET_FILES[dataset_name]
    extracted_paths: list[Path] = []
    with zipfile.ZipFile(archive_path) as archive:
        for relative_path in (spec.train_file, spec.test_file):
            try:
                payload = archive.read(relative_path.name)
            except KeyError as error:
                raise RuntimeError(
                    f"Archive {archive_path} did not contain expected member {relative_path.name} for {dataset_name}"
                ) from error
            destination = dataset_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            extracted_paths.append(destination)
    return extracted_paths


def _parse_checksums(raw_checksums: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in raw_checksums:
        if "=" not in raw:
            raise ValueError(f"Invalid --checksum value {raw!r}; expected format dataset_name=sha256hex")
        dataset_name, expected = raw.split("=", 1)
        dataset_name = dataset_name.strip().lower()
        expected = expected.strip().lower()
        if dataset_name not in DATASET_URLS:
            raise ValueError(f"Invalid checksum dataset key {dataset_name!r}")
        if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
            raise ValueError(f"Invalid sha256 checksum for {dataset_name!r}: {expected!r}")
        result[dataset_name] = expected
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PatchTST demo datasets and checkpoints")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/patchtst"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("models/demos/wormhole/patchtst/.hf_checkpoints"))
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=sorted(DATASET_URLS.keys()),
        choices=sorted(DATASET_URLS.keys()),
        help="Datasets to fetch into dataset-root.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--checksum",
        action="append",
        default=[],
        help="Optional dataset checksum in dataset_name=sha256hex format. Can be repeated.",
    )
    args = parser.parse_args()

    cfg = PatchTSTDemoConfig(dataset_root=args.dataset_root)
    checksums = _parse_checksums(args.checksum)

    downloaded_paths: list[Path] = []
    for dataset_name in args.datasets:
        primary_url, fallback_url = DATASET_URLS[dataset_name]
        if dataset_name in ARCHIVE_DATASET_FILES:
            destination = args.dataset_root / f"{dataset_name}.zip"
            if fallback_url is None:
                _download(primary_url, destination, timeout_seconds=args.timeout_seconds, retries=args.retries)
            else:
                _download_with_fallback(
                    primary_url,
                    fallback_url,
                    destination,
                    timeout_seconds=args.timeout_seconds,
                    retries=args.retries,
                )
            extracted = _extract_archive_dataset(dataset_name, destination, args.dataset_root)
            destination.unlink(missing_ok=True)
            downloaded_paths.extend(extracted)
        else:
            destination = args.dataset_root / FORECAST_DATASET_FILES[dataset_name]
            if fallback_url is None:
                _download(primary_url, destination, timeout_seconds=args.timeout_seconds, retries=args.retries)
            else:
                _download_with_fallback(
                    primary_url,
                    fallback_url,
                    destination,
                    timeout_seconds=args.timeout_seconds,
                    retries=args.retries,
                )
            if dataset_name in checksums:
                observed = _sha256(destination)
                expected = checksums[dataset_name]
                if observed != expected:
                    raise RuntimeError(
                        f"Checksum mismatch for {dataset_name}: expected={expected}, observed={observed}"
                    )
            downloaded_paths.append(destination)

    for task in ("forecast", "pretraining", "channel_attention"):
        snapshot_download(
            repo_id=cfg.checkpoint_ids[task],
            repo_type="model",
            revision=cfg.checkpoint_revisions[task],
            local_dir=args.checkpoint_root / task,
            local_dir_use_symlinks=False,
        )

    for path in downloaded_paths:
        print(f"Downloaded dataset -> {path}")
    print(f"Downloaded reference checkpoints -> {args.checkpoint_root}")
    print(
        "Generate real classification/regression checkpoints with `python -m models.demos.wormhole.patchtst.demo.demo finetune`."
    )


if __name__ == "__main__":
    main()
