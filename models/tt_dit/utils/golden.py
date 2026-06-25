# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import torch
from loguru import logger

# Environment variable to specify the golden data directory.
# If not set, golden data will be computed inline and not cached.
WAN_GROUND_TRUTH_DIR_ENV = "WAN_GROUND_TRUTH_DIR"


def _golden_dir() -> Path | None:
    """Return the golden data directory from the environment, or None if not set."""
    val = os.environ.get(WAN_GROUND_TRUTH_DIR_ENV)
    if val is None:
        return None
    return Path(val)


def golden_path(test_name: str, param_id: str) -> Path | None:
    """Return the full path for a golden data file, or None if the directory is not configured."""
    root = _golden_dir()
    if root is None:
        return None
    return root / test_name / f"{param_id}.pt"


def load_golden(test_name: str, param_id: str) -> dict[str, torch.Tensor] | None:
    """Load golden data from disk if it exists.

    Returns a dict of tensors if the file exists, None otherwise.
    """
    path = golden_path(test_name, param_id)
    if path is None or not path.exists():
        return None
    logger.info(f"Loading golden data from {path}")
    return torch.load(path, weights_only=True)


def save_golden(test_name: str, param_id: str, data: dict[str, torch.Tensor]) -> None:
    """Save golden data to disk. Only saves if the directory is configured."""
    path = golden_path(test_name, param_id)
    if path is None:
        logger.info(f"No golden path for {_golden_dir()} {test_name} {param_id}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving golden data to {path}")
    torch.save(data, path)
