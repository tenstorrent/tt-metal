#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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

import io
import os
import threading
from elftools.elf.elffile import ELFFile
from triage import triage_singleton, ScriptConfig, run_script, TTTriageError
from ttexalens.context import Context
from ttexalens.hardware.risc_debug import ParsedElfFile
from ttexalens.tt_exalens_lib import parse_elf

script_config = ScriptConfig(
    data_provider=True,
)


def _release_elf_file_descriptor(parsed_elf: ParsedElfFile) -> None:
    """
    Replace the ParsedElfFile's ELFFile object with a fresh one backed by an
    in-memory BytesIO buffer, then close the original OS file descriptor.

    pyelftools' ELFFile reads sections and DWARF data lazily from its stream,
    and also caches internal objects (e.g. _section_header_stringtable) that
    hold direct references to that stream.  Simply swapping elf_obj.stream
    leaves those stale references pointing at the now-closed fd, causing
    "ValueError: seek of closed file" on the next section or DWARF access.

    The safe approach is to slurp the file into BytesIO, construct a brand-new
    ELFFile from that buffer (so all internal stream references are fresh), and
    replace parsed_elf.elf atomically before any @cached_property has been
    evaluated.  parse_elf() opens each ELF with open(path, "rb"), and the
    resulting fd stays alive as long as the ParsedElfFile is in the cache.
    With ~500 unique ELF files in a typical triage run this exhausts the OS fd
    limit (ulimit -n), causing [Errno 24] Too many open files.
    """
    stream = parsed_elf.elf.stream
    if stream is None or isinstance(stream, io.BytesIO):
        return  # Already in-memory or no stream to close
    try:
        stream.seek(0)
        content = stream.read()
        parsed_elf.elf = ELFFile(io.BytesIO(content))
    finally:
        stream.close()


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
        if not os.path.exists(elf_path):
            raise TTTriageError(f"ELF file {elf_path} does not exist.")
        with self._lock:
            if elf_path not in self._cache:
                parsed_elf = parse_elf(elf_path, self.context)
                if not parsed_elf:
                    raise TTTriageError(
                        f"Failed to extract DWARF info from ELF file {elf_path}.\nRun workload with TT_METAL_RISCV_DEBUG_INFO=1 to enable debug info."
                    )
                _release_elf_file_descriptor(parsed_elf)
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
