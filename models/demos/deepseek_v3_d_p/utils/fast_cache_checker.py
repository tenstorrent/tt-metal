# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Fast cache completeness checker using single directory scan + in-memory pattern matching.

Replaces sequential Path.glob() calls which each scan the entire directory (~24ms per call).
For 2303 pattern checks: 2303 × 24ms = 56s with glob, vs 240ms with this approach (230x speedup).
"""

import bisect
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
        self._files: list[str] | None = None  # sorted, so a literal prefix is one bisect
        self._timings: dict[str, list[tuple[str, float]]] = defaultdict(list)
        self._iterdir_ms: float = 0.0

    def _ensure_files_loaded(self) -> list[str]:
        if self._files is None:
            start = time.perf_counter()
            self._files = sorted(f.name for f in self.cache_path.iterdir())
            self._iterdir_ms = (time.perf_counter() - start) * 1000
            logger.debug(f"FastCacheChecker: loaded {len(self._files)} files in {self._iterdir_ms:.1f}ms")
        return self._files

    def pattern_exists(self, pattern: str, component: str = "unknown") -> bool:
        """Whether a directory entry's name matches ``pattern``: a name with at most one ``*``
        (e.g. 'layer_0.mla.q_a_proj*.tensorbin'), matched as ``any(cache_path.glob(pattern))`` matches it. Other
        glob syntax is rejected rather than matched literally; a pattern without ``*`` must equal a name.
        """
        if pattern.count("*") > 1 or any(c in pattern for c in "?[]/"):
            raise ValueError(f"FastCacheChecker supports one '*' and no other glob syntax, got {pattern!r}")
        files = self._ensure_files_loaded()

        start = time.perf_counter()
        prefix, star, suffix = pattern.partition("*")
        if not star:
            index = bisect.bisect_left(files, pattern)
            result = index < len(files) and files[index] == pattern
        else:
            # Candidates share the literal prefix, so they are contiguous in sorted order. The length bound keeps
            # prefix and suffix from overlapping, which '*' (zero or more characters) never allows.
            min_len = len(prefix) + len(suffix)
            result = False
            for index in range(bisect.bisect_left(files, prefix), len(files)):
                name = files[index]
                if not name.startswith(prefix):
                    break
                if len(name) >= min_len and name.endswith(suffix):
                    result = True
                    break
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


def mesh_cache_key(mesh_shape: tuple, sp_axis: int = 0) -> str:
    """Weight-cache directory key for a mesh: ``"{rows}x{cols}"``, plus ``"_sp1"`` when SP is on axis 1.

    Tensorbin names carry only dtype and layout, and each file bakes in its per-device placement. The
    SP-axis choice changes that placement without changing the mesh shape (a square 4x4 looks the
    same either way), so it has to be part of the key. SP on axis 0 keeps the bare ``rows x cols``
    key so existing caches stay valid.
    """
    assert sp_axis in (0, 1), f"sp_axis must be 0 or 1, got {sp_axis}"
    rows, cols = tuple(mesh_shape)
    return f"{rows}x{cols}" + ("_sp1" if sp_axis == 1 else "")


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
