# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The PACK-thread SFPLOADMACRO check: the byte filter has no false negatives, and the allowlist is what lets a kernel through."""

import struct

import pytest
from helpers import pack_loadmacro_scan as scan


def _tensix_word(opcode: int, params: int = 0) -> int:
    """A Tensix instruction as it sits in the RISC-V stream: rotated left by two bits."""
    word = (opcode << 24) | params
    return ((word << 2) | (word >> 30)) & 0xFFFFFFFF


SFPLOADMACRO = _tensix_word(0x93, (3 << 20) | (2 << 16) | (7 << 13) | 4)
SFPLOAD = _tensix_word(0x70, 4)
SFPSTORE = _tensix_word(0x72, 4)
RISCV_NOP = 0x00000013


def _elf32(sections) -> bytes:
    """A little-endian ELF32 with one section per (bytes, executable) pair after the null section."""
    header_size = 0x34
    section_header_size = 0x28
    body = b""
    headers = [struct.pack("<IIIIIIIIII", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)]
    for data, executable in sections:
        offset = header_size + len(body)
        body += data
        headers.append(
            struct.pack(
                "<IIIIIIIIII",
                0,  # name
                1,  # SHT_PROGBITS
                0x6 if executable else 0x2,  # SHF_ALLOC | SHF_EXECINSTR, or SHF_ALLOC
                0,
                offset,
                len(data),
                0,
                0,
                4,
                0,
            )
        )
    section_header_offset = header_size + len(body)
    ident = b"\x7fELF" + bytes([1, 1, 1]) + bytes(9)
    header = ident + struct.pack(
        "<HHIIIIIHHHHHH",
        2,
        0xF3,
        1,
        0,
        0,
        section_header_offset,
        0,
        header_size,
        0,
        0,
        section_header_size,
        len(headers),
        0,
    )
    return header + body + b"".join(headers)


def _words(*words) -> bytes:
    return b"".join(struct.pack("<I", word) for word in words)


def _write(tmp_path, sections):
    path = tmp_path / "pack.elf"
    path.write_bytes(_elf32(sections))
    return path


def test_filter_finds_a_loadmacro_in_code(tmp_path):
    path = _write(
        tmp_path, [(_words(RISCV_NOP, SFPLOAD, SFPLOADMACRO, SFPSTORE), True)]
    )
    assert scan.may_contain_loadmacro(path)


def test_filter_finds_a_loadmacro_at_a_two_byte_offset(tmp_path):
    # After one compressed instruction the next word starts at offset 2.
    path = _write(tmp_path, [(b"\x01\x00" + _words(SFPLOADMACRO) + b"\x01\x00", True)])
    assert scan.may_contain_loadmacro(path)


def test_filter_passes_code_without_a_loadmacro(tmp_path):
    path = _write(tmp_path, [(_words(RISCV_NOP, SFPLOAD, SFPSTORE, RISCV_NOP), True)])
    assert not scan.may_contain_loadmacro(path)


def test_filter_ignores_data_sections(tmp_path):
    path = _write(
        tmp_path, [(_words(RISCV_NOP), True), (_words(SFPLOADMACRO) * 4, False)]
    )
    assert not scan.may_contain_loadmacro(path)


@pytest.mark.parametrize("params", [0, 0xFFFFFF, 0x123456])
def test_filter_matches_every_operand_encoding(tmp_path, params):
    path = _write(tmp_path, [(_words(_tensix_word(0x93, params)), True)])
    assert scan.may_contain_loadmacro(path)


def test_no_disassembly_when_the_filter_is_clean(tmp_path):
    path = _write(tmp_path, [(_words(RISCV_NOP, SFPSTORE), True)])
    # A missing objdump would raise if it were run.
    assert scan.count_loadmacros(path, "/nonexistent/objdump") == 0


def test_check_raises_for_an_unlisted_kernel(tmp_path, monkeypatch):
    path = _write(tmp_path, [(_words(SFPLOADMACRO), True)])
    monkeypatch.setattr(scan, "count_loadmacros", lambda elf, objdump: 3)
    with pytest.raises(  # allow-pytest.raises: no expect_error in LLK suite
        RuntimeError, match="3 SFPLOADMACRO"
    ):
        scan.check_pack_elf("sources/some_new_test.cpp", path, "objdump")


def test_check_passes_an_allowlisted_kernel(tmp_path, monkeypatch):
    path = _write(tmp_path, [(_words(SFPLOADMACRO), True)])
    monkeypatch.setattr(scan, "count_loadmacros", lambda elf, objdump: 3)
    listed = next(iter(scan.PACK_LOADMACRO_ALLOWLIST))
    scan.check_pack_elf(listed, path, "objdump")


def test_every_allowlist_entry_gives_a_reason():
    for source, reason in scan.PACK_LOADMACRO_ALLOWLIST.items():
        assert source.startswith("sources/") and source.endswith(".cpp")
        assert len(reason.split()) >= 8, f"{source}: say why the macros are safe"
