# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
General utilities for the GPT-OSS demo.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Iterator, List, Optional

from loguru import logger

import ttnn


def get_cache_file_name(tensor_cache_path, name):
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


_BFOPT_ENV = "GPT_OSS_ENABLE_BFLOAT_OPT"


def should_enable_bfloat_opt(dtype) -> bool:
    if dtype not in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        return False
    return os.getenv(_BFOPT_ENV, "").lower() in ("1", "true", "yes", "y")


_PROFILE_ENV = "GPT_OSS_PROFILE"


def profile_enabled() -> bool:
    return os.getenv(_PROFILE_ENV, "").lower() in ("1", "true", "yes", "y")


@dataclass
class _ProfileLedger:
    _lock: Lock = field(default_factory=Lock)
    _totals: Dict[str, float] = field(default_factory=dict)
    _counts: Dict[str, int] = field(default_factory=dict)
    _maxes: Dict[str, float] = field(default_factory=dict)

    def record(self, key: str, seconds: float) -> None:
        with self._lock:
            self._totals[key] = self._totals.get(key, 0.0) + seconds
            self._counts[key] = self._counts.get(key, 0) + 1
            self._maxes[key] = max(self._maxes.get(key, 0.0), seconds)

    def items(self) -> List[tuple]:
        with self._lock:
            return [
                (key, self._totals[key], self._counts[key], self._maxes[key])
                for key in sorted(self._totals.keys(), key=lambda k: -self._totals[k])
            ]

    def clear(self) -> None:
        with self._lock:
            self._totals.clear()
            self._counts.clear()
            self._maxes.clear()


_PROFILE_LEDGER = _ProfileLedger()


def reset_ledger() -> None:
    _PROFILE_LEDGER.clear()


@contextmanager
def time_span(name: str, *, enabled: Optional[bool] = None) -> Iterator[None]:
    active = profile_enabled() if enabled is None else enabled
    if not active:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        _PROFILE_LEDGER.record(name, elapsed)
        logger.info(f"[gpt_oss.profile] {name}: {elapsed:.3f}s")


def log_summary(header: str = "GPT-OSS CPU-side timing summary") -> None:
    if not profile_enabled():
        return

    rows = _PROFILE_LEDGER.items()
    if not rows:
        return

    total_all = sum(total for _, total, _, _ in rows) or 1.0

    logger.info(f"=== {header} ===")
    logger.info(f"{'stage':<60} {'total_s':>10} {'calls':>8} {'max_s':>10} {'%total':>8}")
    for key, total, count, max_s in rows:
        pct = 100.0 * total / total_all
        logger.info(f"{key:<60} {total:>10.3f} {count:>8d} {max_s:>10.3f} {pct:>7.1f}%")
    logger.info("=" * 80)
