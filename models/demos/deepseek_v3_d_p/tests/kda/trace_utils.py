# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for replaying captured KDA tensors."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open


def load_trace_rows(path: Path, start: int, count: int) -> torch.Tensor:
    """Read a bounded row interval from a named single-stream safetensor file."""
    if start < 0 or count <= 0:
        raise ValueError("trace row interval must be nonnegative and nonempty")
    with safe_open(path, framework="pt", device="cpu") as shard:
        value = shard.get_slice(path.stem)[start : start + count]
    if value.shape[0] != count:
        raise ValueError(f"{path}: requested {count} rows at {start}, got {value.shape[0]}")
    return value
