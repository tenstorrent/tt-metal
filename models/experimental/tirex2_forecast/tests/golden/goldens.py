# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Where the fp32 CPU goldens live (outside git) and a cached loader."""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import torch

GOLDEN_DIR = Path(os.environ.get("TIREX2_GOLDENS", "/home/andy/bringup/goldens/tirex2"))
MANIFEST = Path(__file__).resolve().parent / "MANIFEST.json"


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / name


def have_goldens() -> bool:
    return (GOLDEN_DIR / "single.pt").is_file()


@functools.lru_cache(maxsize=4)
def golden(name: str) -> dict:
    """``torch.load`` of ``<GOLDEN_DIR>/<name>.pt`` (cached; treat the result as read-only)."""
    return torch.load(GOLDEN_DIR / f"{name}.pt", map_location="cpu", weights_only=True)


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())
