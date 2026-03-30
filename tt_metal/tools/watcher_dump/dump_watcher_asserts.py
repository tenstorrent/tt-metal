#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Resolve a watcher ASSERT() failure to a callstack using ELF DWARF.

Called automatically by the C++ watcher when a kernel trips an ASSERT():

    python3 dump_watcher_asserts.py \\
        --file-id   0x002B \\
        --line-num  142    \\
        --kernel-elf /path/to/built/.../trisc1/trisc1.elf \\
        --kernel-offset 0xFFC00000

The C++ watcher already reads file_id + line_num from the device mailbox and
knows the kernel ELF path from the build environment.  This script does the
DWARF lookup to resolve file_id back to a source path and find the function
at that file:line.
"""

import argparse
import os
import subprocess
import sys

try:
    from elftools.elf.elffile import ELFFile
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyelftools", "-q"])
    from elftools.elf.elffile import ELFFile


# ── FNV-1a hash (mirrors device-side constexpr debug_file_hash) ──────────────


def _debug_file_hash(s: str) -> int:
    h = 2166136261
    for c in s:
        h = ((h ^ ord(c)) * 16777619) & 0xFFFFFFFF
    return ((h >> 16) ^ (h & 0xFFFF)) & 0xFFFF


# ── DWARF helpers (pyelftools only) ──────────────────────────────────────────


def _build_file_line_ranges(elf_file: ELFFile) -> dict:
    """
    Build a map of (start_addr, end_addr) -> (filename, line, col)
    from the DWARF line programs.  Same logic as ElfDwarf.file_lines_ranges
    in ttexalens.
    """
    if not elf_file.has_dwarf_info():
        return {}
    dwarf = elf_file.get_dwarf_info(relocate_dwarf_sections=False)
    result = {}
    for CU in dwarf.iter_CUs():
        lp = dwarf.line_program_for_CU(CU)
        if not lp:
            continue
        delta = 1 if lp.header.version < 5 else 0
        prev = None
        for entry in lp.get_entries():
            if entry.state is None:
                continue
            fe = lp["file_entry"][entry.state.file - delta]
            d = lp["include_directory"][fe.dir_index].decode("utf-8")
            fname = os.path.join(d, fe.name.decode("utf-8"))
            cur = (entry.state.address, fname, entry.state.line, entry.state.column)
            if prev is not None:
                result[(prev[0], cur[0])] = (prev[1], prev[2], prev[3])
            prev = cur
        if prev is not None:
            result[(prev[0], prev[0] + 4)] = (prev[1], prev[2], prev[3])
    return result


def resolve_file_from_hash(file_id: int, ranges: dict) -> str | None:
    """
    Scan the DWARF file paths and return the one matching file_id.

    The hash was computed from __FILE__ at compile time.  The DWARF stores
    absolute paths; __FILE__ is typically a relative path from the repo root.
    We recover it by trying progressively shorter prefixes (at directory
    boundaries) of each DWARF path until the hash matches — no hardcoded
    path needed.
    """
    seen: set[str] = set()
    for (_start, _end), (dwarf_fname, _line, _col) in ranges.items():
        if dwarf_fname in seen:
            continue
        seen.add(dwarf_fname)

        # Try the full path as stored in DWARF.
        if _debug_file_hash(dwarf_fname) == file_id:
            return dwarf_fname

        # Try stripping one directory component at a time from the left.
        # This recovers the relative path __FILE__ expanded to at compile time.
        parts = dwarf_fname.split("/")
        for i in range(1, len(parts)):
            candidate = "/".join(parts[i:])
            if candidate and _debug_file_hash(candidate) == file_id:
                return candidate

    return None


def find_assert_address(ranges: dict, filename: str, line_num: int) -> int | None:
    """Return the first address in the DWARF line table at (filename, line_num)."""
    for (start, _end), (dwarf_fname, line, _col) in ranges.items():
        if line == line_num and (dwarf_fname == filename or dwarf_fname.endswith(filename)):
            return start
    return None


def find_function_at_address(elf_file: ELFFile, address: int) -> str | None:
    """
    Walk the DWARF DIE tree to find the narrowest subprogram whose address
    range contains `address`.  Same algorithm as ElfDwarf.find_function_by_address
    in ttexalens.
    """
    if not elf_file.has_dwarf_info():
        return None
    dwarf = elf_file.get_dwarf_info(relocate_dwarf_sections=False)

    def _get_ranges(die):
        if "DW_AT_low_pc" not in die.attributes:
            return []
        lo = die.attributes["DW_AT_low_pc"].value
        hi_a = die.attributes.get("DW_AT_high_pc")
        if not hi_a:
            return []
        hi = hi_a.value if hi_a.form == "DW_FORM_addr" else lo + hi_a.value
        return [(lo, hi)]

    best_name, best_span = None, None

    def _walk(die):
        nonlocal best_name, best_span
        for lo, hi in _get_ranges(die):
            if lo <= address < hi:
                span = hi - lo
                if die.tag in ("DW_TAG_subprogram", "DW_TAG_inlined_subroutine"):
                    n = die.attributes.get("DW_AT_name")
                    if n:
                        name = n.value.decode() if isinstance(n.value, bytes) else n.value
                        if best_span is None or span < best_span:
                            best_span = span
                            best_name = name
                for child in die.iter_children():
                    _walk(child)

    for CU in dwarf.iter_CUs():
        _walk(CU.get_top_DIE())
    return best_name


# ── Core resolution ───────────────────────────────────────────────────────────


def resolve_assert(file_id: int, line_num: int, elf_paths: list[tuple[str, int | None]]) -> dict:
    """
    Resolve file_id + line_num to a source file and function using DWARF.
    elf_paths: list of (elf_file_path, load_offset_or_None).
    """
    resolved_file: str | None = None
    func_name: str | None = None
    assert_line: int = line_num

    for elf_path, _offset in elf_paths:
        if not elf_path or not os.path.exists(elf_path):
            continue
        try:
            with open(elf_path, "rb") as f:
                elf = ELFFile(f)
                ranges = _build_file_line_ranges(elf)

                if resolved_file is None:
                    resolved_file = resolve_file_from_hash(file_id, ranges)

                if resolved_file and func_name is None:
                    addr = find_assert_address(ranges, resolved_file, line_num)
                    if addr is not None:
                        func_name = find_function_at_address(elf, addr)
        except Exception:
            continue

    return {
        "file": resolved_file or f"unknown file (hash=0x{file_id:04X})",
        "line": assert_line,
        "function": func_name,
    }


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file-id", required=True, type=lambda x: int(x, 0), help="file_id hash from mailbox")
    parser.add_argument("--line-num", required=True, type=int, help="line_num from mailbox")
    parser.add_argument("--kernel-elf", required=False, default=None, help="kernel ELF file path")
    parser.add_argument(
        "--kernel-offset", required=False, type=lambda x: int(x, 0), default=None, help="kernel load offset (hex)"
    )
    parser.add_argument("--firmware-elf", required=False, default=None, help="firmware ELF file path")
    args = parser.parse_args()

    elf_paths: list[tuple[str, int | None]] = []
    if args.kernel_elf:
        elf_paths.append((args.kernel_elf, args.kernel_offset))
    if args.firmware_elf:
        elf_paths.append((args.firmware_elf, None))

    if not elf_paths:
        parser.error("at least one of --kernel-elf or --firmware-elf is required")

    result = resolve_assert(args.file_id, args.line_num, elf_paths)

    print(f"File     : {result['file']}")
    print(f"Line     : {result['line']}")
    print(f"Function : {result['function'] or '(rebuild with TT_METAL_RISCV_DEBUG_INFO=1 for function name)'}")


if __name__ == "__main__":
    main()
