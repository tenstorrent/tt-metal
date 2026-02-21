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

import os
import errno
import shutil
import tempfile
import threading
from triage import triage_singleton, ScriptConfig, run_script, TTTriageError
from ttexalens.context import Context
from ttexalens.hardware.risc_debug import ParsedElfFile
from ttexalens.tt_exalens_lib import parse_elf

script_config = ScriptConfig(
    data_provider=True,
)


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
        """Parse ELF, retrying only once when ESTALE is encountered."""
        try:
            return parse_elf(elf_path, self.context)
        except Exception as exc:
            if not self._is_estale_error(exc):
                raise

        # Retry once by reopening through a fresh local copy.
        with tempfile.TemporaryDirectory(prefix="tt-triage-elf-") as tmp_dir:
            local_elf_path = os.path.join(tmp_dir, os.path.basename(elf_path))
            shutil.copy2(elf_path, local_elf_path)
            return parse_elf(local_elf_path, self.context)

    def __getitem__(self, elf_path: str) -> ParsedElfFile:
        """
        Get a ParsedElfFile from cache or parse and cache it if not present.

        This method is thread-safe and will only parse each ELF file once,
        returning the cached instance on subsequent calls.

        Args:
            elf_path: Path to the ELF file

        Returns:
            ParsedElfFile object for the given path
        """
        if not os.path.exists(elf_path):
            raise TTTriageError(f"ELF file {elf_path} does not exist.")
        with self._lock:
            if elf_path not in self._cache:
                try:
                    parsed_elf = self._parse_elf_with_estale_retry(elf_path)
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
