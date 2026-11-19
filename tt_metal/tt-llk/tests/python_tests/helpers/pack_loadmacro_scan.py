# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Finds SFPLOADMACRO instructions in a Blackhole PACK-thread kernel ELF.

On Blackhole a store scheduled by SFPLOADMACRO that lands late (a store delay of 7, or an
instruction-dependent delay) is not serialised against the other writers of the same half of Dst (FPU
results, MOVA2D/MOVB2D). When the two land in one cycle the macro store wins and the other write is lost,
with nothing stalled or reported (tt-metal#64475). A PACK kernel that issues
SFPLOADMACRO while MATH can write that half is therefore a silent-corruption hazard unless its macros are
known to be in the safe range, and a kernel that does it safely has to say why in PACK_LOADMACRO_ALLOWLIST.
"""

import re
import struct
import subprocess
from pathlib import Path

# Test source -> why its PACK-thread SFPLOADMACRO use is safe.
PACK_LOADMACRO_ALLOWLIST = {
    "sources/topk_xl_split_test.cpp": (
        "K 1024 and 2048 keep the fused sort macros; every MATH step and PACK part holds the mutex of "
        "each Dst half it writes and releases it behind a stall on its own unit (ckernel_sfpu_topk_xl_split.h)"
    ),
}

_SHF_EXECINSTR = 0x4

# A Tensix instruction sits in the RISC-V stream rotated left by two bits. SFPLOADMACRO is opcode 0x93,
# so the low two bits of its first byte are 0b10 and the top six of its last byte are 0b010011.
_LAST_BYTE = re.compile(rb"[\x4c-\x4f]")


def _executable_sections(elf: bytes):
    """Yields the bytes of each executable section of a little-endian ELF32."""
    if elf[:4] != b"\x7fELF" or elf[4] != 1 or elf[5] != 1:
        raise ValueError("not a little-endian ELF32 file")
    (shoff,) = struct.unpack_from("<I", elf, 0x20)
    shentsize, shnum = struct.unpack_from("<HH", elf, 0x2E)
    for index in range(shnum):
        _, _, flags, _, offset, size = struct.unpack_from(
            "<IIIIII", elf, shoff + index * shentsize
        )
        if flags & _SHF_EXECINSTR:
            yield elf[offset : offset + size]


def may_contain_loadmacro(elf_path: Path) -> bool:
    """Cheap filter with no false negatives: a word that could be SFPLOADMACRO at any 2-byte offset."""
    elf = Path(elf_path).read_bytes()
    for section in _executable_sections(elf):
        for match in _LAST_BYTE.finditer(section, 3):
            first = match.start() - 3
            if first % 2 == 0 and section[first] & 0x3 == 0x2:
                return True
    return False


def count_loadmacros(elf_path: Path, objdump: str) -> int:
    """SFPLOADMACRO instructions in the ELF, as the toolchain's disassembler decodes them."""
    if not may_contain_loadmacro(elf_path):
        return 0
    listing = subprocess.run(
        [objdump, "-d", str(elf_path)], capture_output=True, text=True, check=True
    ).stdout
    return sum(1 for line in listing.splitlines() if "sfploadmacro" in line.lower())


def check_pack_elf(test_name: str, elf_path: Path, objdump: str) -> None:
    """Raises if a PACK ELF issues SFPLOADMACRO and its test source is not on the allowlist."""
    if test_name in PACK_LOADMACRO_ALLOWLIST:
        return
    count = count_loadmacros(elf_path, objdump)
    if count:
        raise RuntimeError(
            f"{test_name}: the PACK kernel issues {count} SFPLOADMACRO instruction(s). On Blackhole a "
            "macro-scheduled Dst store silently overrides a same-cycle write from MATH to the same half "
            "of Dst (tt-metal#64475). Use a macro-free SFPU routine on PACK, or, if MATH cannot write that half "
            "meanwhile, add the test to PACK_LOADMACRO_ALLOWLIST in helpers/pack_loadmacro_scan.py with "
            f"the reason. ELF: {elf_path}"
        )
