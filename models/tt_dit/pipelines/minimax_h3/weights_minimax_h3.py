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

# Cheap assets (~20 MB together) that the pipeline reads for any task: the model index, the two
# scheduler configs, the tokenizer, and the processor. Always fetched so a partition-scoped download
# still runs, and required of a cached snapshot for the same reason: a snapshot that another tool
# fetched with its own `allow_patterns` can hold every weight partition and still lack them.
_ALWAYS_FILES = ("model_index.json",)
_ALWAYS_DIRS = ("scheduler", "audio_scheduler", "tokenizer", "processor")
_ALWAYS_PATTERNS = (*_ALWAYS_FILES, *(f"{name}/*" for name in _ALWAYS_DIRS))

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


def _missing_partitions(directory: Path, required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not (directory / name).is_dir()]


def _missing_always_assets(directory: Path) -> list[str]:
    missing = [name for name in _ALWAYS_FILES if not (directory / name).is_file()]
    missing += [name for name in _ALWAYS_DIRS if not (directory / name).is_dir()]
    return missing


def _is_complete(directory: Path, required: tuple[str, ...]) -> bool:
    """Whether `directory` holds every required partition and every always-fetched asset."""
    return directory.is_dir() and not _missing_partitions(directory, required) and not _missing_always_assets(directory)


def _cached_snapshot(required: tuple[str, ...]) -> Path | None:
    """A cached snapshot that already holds every required partition and the always-fetched assets.

    The revision `refs/main` points at is preferred, then the most recently modified snapshot. A
    partial snapshot (weights present, tokenizer absent) is passed over rather than returned, so the
    caller falls through to a download that completes it instead of failing later at the tokenizer.
    """
    repo_dir = _hf_cache_root() / "hub" / f"models--{MINIMAX_H3_REPO_ID.replace('/', '--')}"
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted((s for s in snapshots.iterdir() if s.is_dir()), key=lambda s: s.stat().st_mtime, reverse=True)
    main_ref = repo_dir / "refs" / "main"
    if main_ref.is_file():
        revision = main_ref.read_text().strip()
        candidates.sort(key=lambda s: s.name != revision)  # stable: refs/main first, then by mtime
    for snapshot in candidates:
        if _is_complete(snapshot, required):
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

    An explicit directory is checked for the required partitions only: it is the caller's own layout,
    and a partition-only directory is a legitimate way to run one component's tests. A cached or
    downloaded snapshot must also hold the always-fetched assets, since a snapshot missing them can
    be completed by the download this resolver gates.
    """
    if allow_download is None:
        allow_download = os.environ.get(ALLOW_DOWNLOAD_ENV) == "1"

    required = tuple(required_subdirs)
    explicit = weights_dir or os.environ.get(MODEL_PATH_ENV)

    if explicit:
        directory = Path(explicit)
        if not directory.is_dir():
            raise WeightsNotFoundError(f"{MODEL_PATH_ENV} points at {directory}, which is not a directory")
        missing = _missing_partitions(directory, required)
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
    missing = _missing_partitions(snapshot, required) + _missing_always_assets(snapshot)
    if missing:
        raise WeightsNotFoundError(f"downloaded MiniMax-H3 snapshot at {snapshot} is missing {missing}")
    return snapshot
