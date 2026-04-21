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
    ELF files are accessed via mmap rather than held-open file descriptors.

Owner:
    adjordjevic-TT
"""

import mmap
import os
import threading

from elftools.elf.elffile import ELFFile

from triage import triage_singleton, ScriptConfig, run_script, TTTriageError
from ttexalens.context import Context
from ttexalens.hardware.risc_debug import ParsedElfFile
from utils import INFO

script_config = ScriptConfig(
    data_provider=True,
)


class ElfsCache:
    """
    Thread-safe cache for ParsedElfFile objects, backed by mmap.
    When `enabled=False` the cache acts as a pass-through: every lookup
    re-parses the ELF. Intended as a mitigation toggle via --disable-elf-cache
    if the cache misbehaves.
    """

    def __init__(self, context: Context, enabled: bool = True):
        self.context = context
        self._enabled = enabled
        self._cache: dict[str, ParsedElfFile] = {}
        self._lock = threading.Lock()

        self._distinct_paths: set[str] = set()
        self._total_bytes = 0

    def __getitem__(self, elf_path: str) -> ParsedElfFile:
        if not os.path.exists(elf_path):
            raise TTTriageError(f"ELF file {elf_path} does not exist.")

        with self._lock:
            if self._enabled and elf_path in self._cache:
                return self._cache[elf_path]

            parsed_elf = self._parse(elf_path)

            if elf_path not in self._distinct_paths:
                self._distinct_paths.add(elf_path)
                try:
                    self._total_bytes += os.path.getsize(elf_path)
                except OSError:
                    pass

            if self._enabled:
                self._cache[elf_path] = parsed_elf
            return parsed_elf

    def has_elf(self, elf_path: str) -> bool:
        with self._lock:
            return elf_path in self._cache

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def get_cached_paths(self) -> list[str]:
        with self._lock:
            return list(self._cache.keys())

    def log_stats(self) -> None:
        with self._lock:
            INFO(
                f"elfs cache enabled={self._enabled}\n"
                f"distinct elf files={len(self._distinct_paths)}\n"
                f"total size={self._total_bytes / 1e6:.1f}MB\n"
            )

    @staticmethod
    def _parse(elf_path: str) -> ParsedElfFile:
        fd = os.open(elf_path, os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
        finally:
            os.close(fd)

        try:
            elf = ELFFile(mm)
            if not elf.has_dwarf_info():
                mm.close()
                raise TTTriageError(
                    f"Failed to extract DWARF info from ELF file {elf_path}.\n"
                    f"Run workload with TT_METAL_RISCV_DEBUG_INFO=1 to enable debug info."
                )
        except Exception:
            mm.close()
            raise

        return ParsedElfFile(elf, elf_path)


@triage_singleton
def run(args, context: Context) -> ElfsCache:
    enabled = not bool(args["--disable-elf-cache"])
    return ElfsCache(context, enabled=enabled)


if __name__ == "__main__":
    run_script()
