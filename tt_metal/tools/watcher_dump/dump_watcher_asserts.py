#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Resolve a watcher ASSERT() failure to a callstack using ELF DWARF.

Called automatically by the C++ watcher when a kernel trips an ASSERT():

    python3 dump_watcher_asserts.py \\
        --msg-ptr   0x06600010 \\
        --kernel-elf /path/to/built/.../trisc1/trisc1.elf

The C++ watcher reads msg_ptr from the device mailbox.  When bit 31 is clear,
msg_ptr is the VMA of a debug_assert_info_t struct in .debug_assert_msgs:
  [uint32_t filename_ptr][uint16_t line_num][char msg...\0]
The filename string is also in .debug_assert_msgs — no hash needed.
When bit 31 is set, msg_ptr encodes only a line number (plain/type asserts in
constexpr functions where static locals are not allowed).
"""

import argparse
import os
import struct
import subprocess
import sys

try:
    from elftools.elf.elffile import ELFFile
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyelftools", "-q"])
    from elftools.elf.elffile import ELFFile


# ── Read debug_assert_info_t from ELF by VMA ─────────────────────────────────


def read_assert_info_from_ptr(msg_ptr: int, elf_paths: list) -> dict | None:
    """
    Read assert location info from msg_ptr.
    Bit 31 = 1: packed line number only — bits[15:0]=line_num, no filename/message.
    Bit 31 = 0: VMA pointer to a debug_assert_info_t in .debug_assert_msgs.
    Returns {'line_num': int, 'msg': str, 'filename': str|None} or None if not found.
    """
    if msg_ptr == 0:
        return None
    if msg_ptr & 0x80000000:
        return {"line_num": msg_ptr & 0xFFFF, "msg": "", "filename": None}

    for elf_path, _offset in elf_paths:
        if not elf_path or not os.path.exists(elf_path):
            continue
        try:
            with open(elf_path, "rb") as f:
                elf = ELFFile(f)
                section = elf.get_section_by_name(".debug_assert_msgs")
                if section is None:
                    continue
                sh_addr = section["sh_addr"]
                sh_size = section["sh_size"]
                if msg_ptr < sh_addr or msg_ptr >= sh_addr + sh_size:
                    continue
                offset = msg_ptr - sh_addr
                data = section.data()
                # Struct layout (packed): [uint32_t filename_ptr][uint16_t line_num][char msg...\0]
                if offset + 6 > len(data):
                    continue
                filename_ptr, line_num = struct.unpack_from("<IH", data, offset)

                def read_str_at(ptr):
                    if ptr == 0 or ptr < sh_addr or ptr >= sh_addr + sh_size:
                        return ""
                    off = ptr - sh_addr
                    end = data.index(b"\x00", off)
                    return data[off:end].decode("utf-8", errors="replace")

                msg_bytes = data[offset + 6 :]
                msg_end = msg_bytes.find(b"\x00")
                msg = msg_bytes[:msg_end].decode("utf-8", errors="replace") if msg_end > 0 else ""

                return {
                    "line_num": line_num,
                    "msg": msg,
                    "filename": read_str_at(filename_ptr),
                }
        except Exception:
            continue
    return None


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


def resolve_assert(msg_ptr: int, elf_paths: list) -> dict:
    """
    Resolve msg_ptr to a source file and function using the .debug_assert_msgs
    section (for filename + line_num + message) and DWARF (for function name).
    elf_paths: list of (elf_file_path, load_offset_or_None).
    """
    # Step 1: read filename + line_num + message from the ELF section.
    info = read_assert_info_from_ptr(msg_ptr, elf_paths)
    if info is None:
        return {
            "file": f"unknown file (msg_ptr=0x{msg_ptr:08x})",
            "line": 0,
            "message": "",
            "function": None,
        }

    line_num = info["line_num"]
    message = info["msg"]
    resolved_file: str | None = info.get("filename") or None
    func_name: str | None = None

    for elf_path, _offset in elf_paths:
        if not elf_path or not os.path.exists(elf_path):
            continue
        try:
            with open(elf_path, "rb") as f:
                elf = ELFFile(f)
                ranges = _build_file_line_ranges(elf)
                if resolved_file and func_name is None:
                    addr = find_assert_address(ranges, resolved_file, line_num)
                    if addr is not None:
                        func_name = find_function_at_address(elf, addr)
        except Exception:
            continue

    return {
        "file": resolved_file or "unknown file",
        "line": line_num,
        "message": message,
        "function": func_name,
    }


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--msg-ptr", required=True, type=lambda x: int(x, 0), help="msg_ptr VMA from mailbox (hex)")
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

    result = resolve_assert(args.msg_ptr, elf_paths)

    print(f"File     : {result['file']}")
    print(f"Line     : {result['line']}")
    if result["message"]:
        print(f"Message  : {result['message']}")
    print(f"Function : {result['function'] or '(rebuild with TT_METAL_RISCV_DEBUG_INFO=1 for function name)'}")


if __name__ == "__main__":
    main()
