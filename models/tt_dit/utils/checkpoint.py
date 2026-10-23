# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Resolve a HuggingFace repo id to a shared local mirror of its weights.

The mirror holds full repo snapshots at `<root>/<org>/<repo>` (the layout `hf download --local-dir`
produces), so a repo found there loads with no hub lookup, and the directory keeps the repo id's
basename that the tt_dit cache namespace is keyed on. A repo absent from the mirror resolves to the
repo id itself, so callers fall back to the HuggingFace cache unchanged.

The mirror is opt-in: it is only consulted when `$TT_DIT_WEIGHTS_ROOT` names its root. CI jobs whose
runners mount a shared mirror set it in their job env.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

WEIGHTS_ROOT_ENV = "TT_DIT_WEIGHTS_ROOT"


def weights_root() -> Path | None:
    """The mirror root from `$TT_DIT_WEIGHTS_ROOT`, or None (mirror disabled) when it is unset or empty."""
    root = os.environ.get(WEIGHTS_ROOT_ENV)
    return Path(root) if root else None


def _is_repo_id(name: str) -> bool:
    org, _, repo = name.partition("/")
    return bool(org) and bool(repo) and "/" not in repo and not org.startswith(".")


def local_checkpoint_dir(repo_id: str, *required: str) -> Path | None:
    """The mirror directory for `repo_id` if it holds every entry in `required` (files or subfolders), else None."""
    root = weights_root()
    if root is None or not _is_repo_id(repo_id):
        return None
    directory = root / repo_id
    if not directory.is_dir():
        return None
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        logger.warning(f"Local mirror {directory} is missing {missing}; not using it for {repo_id}")
        return None
    return directory


def local_revision(name: str) -> str | None:
    """The commit `hf download --local-dir` recorded for directory `name`, or None.

    `from_pretrained(..., return_commit_hash=True)` returns None for a local directory, so callers that key
    data on the commit (goldens) read it from the download metadata instead.
    """
    download = Path(name) / ".cache" / "huggingface" / "download"
    if not download.is_dir():
        return None
    preferred = download / "model_index.json.metadata"
    for metadata in [preferred] if preferred.is_file() else sorted(download.rglob("*.metadata")):
        lines = metadata.read_text().split()
        if lines:
            return lines[0]
    return None


def resolve_checkpoint(name: str, *required: str) -> str:
    """`name` as a local mirror path when the mirror holds it, else `name` unchanged.

    `name` that is already a directory, or is not an `org/repo` id, is returned as is.
    """
    if os.path.isdir(name):
        return name
    local = local_checkpoint_dir(name, *required)
    if local is None:
        return name
    logger.info(f"Resolving {name} from local mirror {local} (no hub lookup)")
    return str(local)
