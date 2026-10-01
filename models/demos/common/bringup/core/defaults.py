# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The framework's switches (defaults.yaml next to the framework root): Spec.get falls back to them, and the module
constants that used to hold defaults (thresholds, retry policy, review switches) are read from here."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PATH = Path(__file__).resolve().parents[1] / "defaults.yaml"
_MISSING = object()


@lru_cache(maxsize=1)
def data() -> dict:
    return yaml.safe_load(PATH.read_text()) or {}


def get(key: str, default=_MISSING):
    cur = data()
    for k in key.split("."):
        if not isinstance(cur, dict) or k not in cur:
            if default is _MISSING:
                raise KeyError(f"{key} is not in {PATH.name}")
            return default
        cur = cur[k]
    return cur


def has(key: str) -> bool:
    return get(key, _MISSING_SENTINEL) is not _MISSING_SENTINEL


_MISSING_SENTINEL = object()
