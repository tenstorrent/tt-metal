# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Resolve the MiniMax-H3 diffusers snapshot, optionally fetching it from HuggingFace.

Resolution order matches the lightx2v loader (`experimental/utils/lightx2v_loader.py`): an explicit
path first, then the HuggingFace cache, then a download -- and the download only happens when it has
been opted into with `TT_DIT_ALLOW_HF_DOWNLOAD=1`, so no test run pulls 144 GB by surprise.

Only the diffusers partitions the caller asks for are fetched. The repo's top-level `FL2VA/` and
`Ref2VA/` trees are self-contained original-format bundles that duplicate the root partitions and
are never read here, so skipping them takes the full 498 GB repo down to ~144 GB for `t2va`.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

MINIMAX_H3_REPO_ID = "MiniMaxAI/MiniMax-H3"

MODEL_PATH_ENV = "MINIMAX_H3_MODEL_PATH"
ALLOW_DOWNLOAD_ENV = "TT_DIT_ALLOW_HF_DOWNLOAD"

# Cheap partitions (~20 MB together) that the pipeline reads for any task: the scheduler configs,
# the tokenizer, and the processor. Always fetched so a partition-scoped download still runs.
_ALWAYS_PATTERNS = ("model_index.json", "scheduler/*", "audio_scheduler/*", "tokenizer/*", "processor/*")

# Approximate on-disk cost per partition, for the pre-download log line.
_PARTITION_GB = {
    "transformer": 66.3,
    "transformer_ref": 66.3,
    "text_encoder": 66.7,
    "vae": 10.4,
    "audio_vae": 0.6,
}


class WeightsNotFoundError(FileNotFoundError):
    """Raised when the MiniMax-H3 snapshot is not on disk and downloads are disabled."""


def _hf_cache_root() -> Path:
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")


def _has_partitions(directory: Path, required: tuple[str, ...]) -> bool:
    return directory.is_dir() and all((directory / name).is_dir() for name in required)


def _cached_snapshot(required: tuple[str, ...]) -> Path | None:
    """The newest cached snapshot that already holds every required partition, if any."""
    snapshots = _hf_cache_root() / "hub" / f"models--{MINIMAX_H3_REPO_ID.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return None
    for snapshot in sorted(snapshots.iterdir(), reverse=True):
        if _has_partitions(snapshot, required):
            return snapshot
    return None


def _download(required: tuple[str, ...]) -> Path:
    from huggingface_hub import snapshot_download

    allow_patterns = [*_ALWAYS_PATTERNS, *(f"{name}/*" for name in required)]
    estimate = sum(_PARTITION_GB.get(name, 0.0) for name in required)
    logger.info(
        f"Downloading MiniMax-H3 partitions {list(required)} (~{estimate:.0f} GB) from "
        f"HuggingFace: {MINIMAX_H3_REPO_ID} -> {_hf_cache_root()}"
    )
    return Path(snapshot_download(repo_id=MINIMAX_H3_REPO_ID, allow_patterns=allow_patterns))


def resolve_weights_dir(
    *required_subdirs: str,
    weights_dir: str | os.PathLike | None = None,
    allow_download: bool | None = None,
) -> Path:
    """The snapshot directory holding every partition in `required_subdirs`.

    `weights_dir` (or `$MINIMAX_H3_MODEL_PATH`) wins when it is set; otherwise the HuggingFace cache
    is searched and then, with `$TT_DIT_ALLOW_HF_DOWNLOAD=1`, fetched. Raises `WeightsNotFoundError`
    when nothing resolves and downloads are off.
    """
    if allow_download is None:
        allow_download = os.environ.get(ALLOW_DOWNLOAD_ENV) == "1"

    required = tuple(required_subdirs)
    explicit = weights_dir or os.environ.get(MODEL_PATH_ENV)

    if explicit:
        directory = Path(explicit)
        if not directory.is_dir():
            raise WeightsNotFoundError(f"{MODEL_PATH_ENV} points at {directory}, which is not a directory")
        missing = [name for name in required if not (directory / name).is_dir()]
        if missing:
            raise WeightsNotFoundError(f"MiniMax-H3 snapshot at {directory} is missing {missing}")
        return directory

    cached = _cached_snapshot(required)
    if cached is not None:
        return cached

    if not allow_download:
        raise WeightsNotFoundError(
            f"MiniMax-H3 weights not found (needed partitions: {list(required) or ['<any>']}).\n"
            f"Point {MODEL_PATH_ENV} at a diffusers snapshot, or re-run with {ALLOW_DOWNLOAD_ENV}=1 "
            f"to fetch {MINIMAX_H3_REPO_ID} into {_hf_cache_root()}."
        )

    snapshot = _download(required)
    missing = [name for name in required if not (snapshot / name).is_dir()]
    if missing:
        raise WeightsNotFoundError(f"downloaded MiniMax-H3 snapshot at {snapshot} is missing {missing}")
    return snapshot
