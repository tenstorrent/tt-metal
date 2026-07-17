# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import torch
from loguru import logger

# Environment variable to specify the golden data directory.
# If not set, golden data will be computed inline and not cached.
TT_DIT_TEST_GOLDEN_CACHE_DIR = "TT_DIT_TEST_GOLDEN_CACHE_DIR"


def _golden_dir() -> Path | None:
    """Return the golden data directory from the environment, or None if not set."""
    val = os.environ.get(TT_DIT_TEST_GOLDEN_CACHE_DIR)
    if val is None:
        return None
    return Path(val)


def golden_path(commit_hash: str, test_name: str, param_id: str) -> Path | None:
    """Return the full path for a golden data file, or None if the directory is not configured.

    Golden data is namespaced by the model ``commit_hash`` so that golden values produced
    against different model revisions do not collide.
    """
    root = _golden_dir()
    if root is None:
        return None
    return root / test_name / param_id / f"{commit_hash}.pt"


def load_golden(commit_hash: str, test_name: str, param_id: str) -> dict[str, torch.Tensor] | None:
    """Load golden data from disk if it exists.

    Returns a dict of tensors if the file exists, None otherwise.
    """
    path = golden_path(commit_hash, test_name, param_id)
    if path is None or not path.exists():
        logger.warning(
            f"No golden data found for commit={commit_hash} test={test_name} param={param_id} (looked in {path})"
        )
        return None
    logger.info(f"Loading golden data from {path}")
    return torch.load(path, weights_only=True)


def save_golden(commit_hash: str, test_name: str, param_id: str, data: dict[str, torch.Tensor]) -> None:
    """Save golden data to disk. Only saves if the directory is configured."""
    path = golden_path(commit_hash, test_name, param_id)
    if path is None:
        logger.info(f"No golden path for {_golden_dir()} {commit_hash} {test_name} {param_id}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving golden data to {path}")
    torch.save(data, path)
