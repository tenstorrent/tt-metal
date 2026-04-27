# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Fast cache completeness checker using single directory scan + in-memory pattern matching.

Replaces sequential Path.glob() calls which each scan the entire directory (~24ms per call).
For 2303 pattern checks: 2303 × 24ms = 56s with glob, vs 240ms with this approach (230x speedup).
"""

import time
from collections import defaultdict
from pathlib import Path

from loguru import logger


class FastCacheChecker:
    """
    Optimized cache file existence checker.

    Instead of N sequential glob() calls (each ~24ms due to directory scan),
    this does:
    - 1 iterdir() to build file set (~40ms)
    - N in-memory pattern matches (~0.03ms each)

    Usage:
        checker = FastCacheChecker(cache_path)
        if checker.pattern_exists("layer_0.mla.q_a_proj*.tensorbin", "MLA"):
            ...
        checker.report()  # prints timing summary

    Or as context manager:
        with FastCacheChecker(cache_path) as checker:
            checker.pattern_exists(...)
        # auto-reports on exit
    """

    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._files: set[str] | None = None
        self._timings: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self._iterdir_ms: float = 0.0

    def _ensure_files_loaded(self) -> set[str]:
        if self._files is None:
            start = time.perf_counter()
            self._files = set(f.name for f in self.cache_path.iterdir())
            self._iterdir_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"FastCacheChecker: loaded {len(self._files)} files in {self._iterdir_ms:.1f}ms")
        return self._files

    def pattern_exists(self, pattern: str, component: str = "unknown") -> bool:
        """Check if any file matches glob pattern (e.g., 'layer_0.mla.q_a_proj*.tensorbin')."""
        files = self._ensure_files_loaded()

        start = time.perf_counter()
        if "*" in pattern:
            prefix, suffix = pattern.split("*", 1)
        else:
            prefix, suffix = pattern, ""

        result = any(f.startswith(prefix) and f.endswith(suffix) for f in files)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._timings[component].append((pattern, elapsed_ms))
        return result

    def report(self):
        """Print timing summary."""
        total_ms = self._iterdir_ms
        total_calls = 0

        logger.info(f"  iterdir: {self._iterdir_ms:.1f} ms ({len(self._files or [])} files)")
        for component, timings in sorted(self._timings.items()):
            comp_total = sum(t[1] for t in timings)
            comp_count = len(timings)
            total_ms += comp_total
            total_calls += comp_count
            logger.info(f"  {component}: {comp_count} checks, {comp_total:.2f} ms")
        logger.info(f"  TOTAL: {total_calls} checks, {total_ms:.1f} ms ({total_ms / 1000:.3f} s)")

    def clear(self):
        """Clear cached state."""
        self._timings.clear()
        self._files = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.report()
        self.clear()


# Module-level singleton for backward compatibility with existing component code
_checker: FastCacheChecker | None = None


def init_checker(cache_path: Path):
    """Initialize the global checker for a cache directory."""
    global _checker
    _checker = FastCacheChecker(cache_path)


def pattern_exists(pattern: str, component: str = "unknown") -> bool:
    """Check pattern using global checker. Must call init_checker() first."""
    if _checker is None:
        raise RuntimeError("Call init_checker(cache_path) before pattern_exists()")
    return _checker.pattern_exists(pattern, component)


def report_and_clear():
    """Report timings and clear global checker."""
    global _checker
    if _checker:
        _checker.report()
        _checker.clear()
        _checker = None
