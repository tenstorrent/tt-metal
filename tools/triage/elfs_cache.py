#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    elfs_cache

Description:
    Thread-safe data provider for caching ParsedElfFile objects.
    Provides an API for grabbing or caching ParsedElfFile objects by given elf path.

Owner:
    adjordjevic-TT
"""

import errno
import logging
from pathlib import Path
import shutil
import tempfile
import threading
import time
from triage import triage_singleton, ScriptConfig, run_script, TTTriageError
from ttexalens.context import Context
from ttexalens.hardware.risc_debug import ParsedElfFile
from ttexalens.tt_exalens_lib import parse_elf

script_config = ScriptConfig(
    data_provider=True,
)

logger = logging.getLogger(__name__)


class ElfsCache:
    """
    Thread-safe cache for ParsedElfFile objects.

    This class provides a thread-safe API for caching and retrieving ParsedElfFile
    objects parsed from ELF files. It automatically parses and caches ELF files
    on first access and returns cached instances on subsequent requests.
    """

    def __init__(self, context: Context):
        """
        Initialize the ELF cache.

        Args:
            context: ttexalens Context object for parsing ELF files
        """
        self.context = context
        self._cache: dict[str, ParsedElfFile] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _is_estale_error(exc: Exception) -> bool:
        if isinstance(exc, OSError) and exc.errno == errno.ESTALE:
            return True
        return "stale file handle" in str(exc).lower()

    def _parse_elf_with_estale_retry(self, elf_path: str) -> ParsedElfFile | None:
        """Parse ELF, retrying with backoff (up to 10s) when ESTALE is encountered."""
        last_exc = None
        max_retries = 4
        backoff_delays = [1.0, 2.0, 3.0, 4.0]

        try:
            return parse_elf(elf_path, self.context)
        except Exception as exc:
            if not self._is_estale_error(exc):
                raise
            last_exc = exc

        for attempt in range(max_retries):
            delay = backoff_delays[attempt]
            logger.info(
                "Retrying ELF parse due to stale file handle (refreshing file handle), "
                "attempt %d/%d, waiting %.1fs...",
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)
            try:
                with tempfile.TemporaryDirectory(prefix="tt-triage-elf-") as tmp_dir:
                    local_elf_path = Path(tmp_dir) / Path(elf_path).name
                    shutil.copy2(elf_path, local_elf_path)
                    return parse_elf(str(local_elf_path), self.context)
            except Exception as exc:
                if not self._is_estale_error(exc):
                    raise
                last_exc = exc

        raise last_exc

    def __getitem__(self, elf_path: str) -> ParsedElfFile:
        """
        Get a ParsedElfFile from cache or parse and cache it if not present.

        This method is thread-safe. Cache writes are synchronized, but parsing
        may occur concurrently for the same path (only the first successful
        result is cached). ESTALE retries happen outside the lock to avoid
        blocking other threads for up to 10s.

        Args:
            elf_path: Path to the ELF file

        Returns:
            ParsedElfFile object for the given path

        Raises:
            TTTriageError: If the ELF file does not exist or cannot be parsed.
        """
        with self._lock:
            if elf_path in self._cache:
                return self._cache[elf_path]

        # Parse outside the lock so ESTALE retries (up to ~10s) don't block other threads.
        try:
            parsed_elf = self._parse_elf_with_estale_retry(elf_path)
        except FileNotFoundError as exc:
            raise TTTriageError(f"ELF file {elf_path} does not exist.") from exc
        except Exception as exc:
            if self._is_estale_error(exc):
                raise TTTriageError(
                    f"Failed to parse ELF file {elf_path} due to stale file handle on filesystem: {exc}"
                ) from exc
            raise
        if not parsed_elf:
            raise TTTriageError(
                f"Failed to extract DWARF info from ELF file {elf_path}.\nRun workload with TT_METAL_RISCV_DEBUG_INFO=1 to enable debug info."
            )

        with self._lock:
            if elf_path not in self._cache:
                self._cache[elf_path] = parsed_elf
            return self._cache[elf_path]

    def has_elf(self, elf_path: str) -> bool:
        """
        Check if an ELF file is already cached.

        Args:
            elf_path: Path to the ELF file

        Returns:
            True if the ELF file is cached, False otherwise
        """
        with self._lock:
            return elf_path in self._cache

    def clear_cache(self) -> None:
        """
        Clear all cached ELF files.

        This method removes all cached ParsedElfFile objects from the cache.
        """
        with self._lock:
            self._cache.clear()

    def get_cached_paths(self) -> list[str]:
        """
        Get list of all cached ELF file paths.

        Returns:
            List of ELF file paths currently in the cache
        """
        with self._lock:
            return list(self._cache.keys())


@triage_singleton
def run(args, context: Context) -> ElfsCache:
    """
    Create and return a thread-safe ELF cache instance.

    Args:
        args: Script arguments (unused)
        context: ttexalens Context object

    Returns:
        ElfsCache instance for caching ParsedElfFile objects
    """
    return ElfsCache(context)


if __name__ == "__main__":
    run_script()
