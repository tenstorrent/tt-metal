#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    elfs_cache

Description:
    Thread-safe data provider for caching ParsedElfFile objects.
    Provides an API for grabbing or caching ParsedElfFile objects by given elf path.
"""

import threading
from triage import triage_singleton, ScriptConfig, run_script
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
        with self._lock:
            if elf_path not in self._cache:
                self._cache[elf_path] = parse_elf(elf_path, self.context)
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
