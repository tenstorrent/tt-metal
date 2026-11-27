# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TILE_LOOP window search (helpers/perf/window_hash.py).

Run: pytest test_perf_window_hash.py
"""

import types

import pytest
from helpers.perf.window_hash import _code, _compute, _hash16, _window, enabled

_MARKER = b"LLK_PROFILER:sources/x.cpp:42:TILE_LOOP"
_ID = _hash16(_MARKER)
_ELF = b"\x00" + _MARKER + b"\x00"


def _insns(lines):
    return [(0x100 + 4 * k, m, o) for k, (m, o) in enumerate(lines)]


def _loop(start, end):
    return _insns(
        [
            ("addi", "sp,sp,-16"),
            *start,
            ("addi", "a4,a4,1"),
            ("bne", "a4,a3,10c <run_kernel+0xc>"),
            *end,
            ("ret", ""),
        ]
    )


def test_inline_zone_entries_bound_the_window():
    insns = _loop(
        [("lui", f"a5,{hex(0xA0000 | _ID)}")], [("lui", f"a5,{hex(0xB0000 | _ID)}")]
    )

    assert _window(insns, _ELF) == [1, 2, 3, 4]


def test_other_zone_ids_are_not_a_window():
    insns = _loop(
        [("lui", f"a5,{hex(0xA0000 | (_ID ^ 1))}")],
        [("lui", f"a5,{hex(0xB0000 | (_ID ^ 1))}")],
    )

    assert _window(insns, _ELF) is None


def test_paths_that_leave_without_the_zone_end_are_not_in_the_window():
    insns = _insns(
        [
            ("lui", f"a5,{hex(0xA0000 | _ID)}"),
            ("beqz", "a0,114 <run_kernel+0x14>"),
            ("addi", "a4,a4,1"),
            ("lui", f"a5,{hex(0xB0000 | _ID)}"),
            ("ret", ""),
            ("lw", "s10,52(sp)"),
            ("ret", ""),
        ]
    )

    assert _window(insns, _ELF) == [0, 1, 2, 3]


_DUMP = """
/build/a/unpack.elf:     file format elf32-littleriscv


Disassembly of section .text:

00010000 <_start>:
   10000:\taddi\tsp,sp,-16
   10004:\tret
"""


def _objdump(monkeypatch, mode, returncode=0):
    monkeypatch.setenv("LLK_WINDOW_HASH", mode)
    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode=returncode, stdout=_DUMP),
    )
    return _compute("objdump", "unpack.elf")


def test_code_hash_ignores_where_the_elf_was_built():
    assert _code(_DUMP).startswith("code-")
    assert _code(_DUMP) == _code(_DUMP.replace("/build/a/", "/another/place/"))


@pytest.mark.parametrize(
    "old, new",
    [
        pytest.param("sp,sp,-16", "sp,sp,-32", id="operand"),
        pytest.param("ret", "ret\n   10008:\tnop", id="added instruction"),
        pytest.param("10004:", "10008:", id="address"),
        pytest.param("<_start>", "<_begin>", id="symbol"),
        pytest.param(".text", ".init", id="section"),
    ],
)
def test_code_hash_changes_with_the_code(old, new):
    assert old in _DUMP
    assert _code(_DUMP.replace(old, new)) != _code(_DUMP)


@pytest.mark.parametrize(
    "text", ["", "x.elf: file format elf32-littleriscv\n", "Disassembly"]
)
def test_no_disassembly_gives_no_code_hash(text):
    assert _code(text) == "no-code"


def test_code_mode_hashes_the_whole_disassembly(monkeypatch):
    assert _objdump(monkeypatch, "code") == _code(_DUMP)


def test_code_mode_reports_a_failed_objdump(monkeypatch):
    assert _objdump(monkeypatch, "code", returncode=1) == "no-objdump"


def test_window_mode_gives_no_code_hash(monkeypatch):
    assert not _objdump(monkeypatch, "1").startswith("code-")


@pytest.mark.parametrize(
    "value, on", [("1", True), ("code", True), ("", False), ("0", False), (None, False)]
)
def test_fingerprints_are_on_in_window_and_code_mode(monkeypatch, value, on):
    monkeypatch.delenv("LLK_WINDOW_HASH", raising=False)
    if value is not None:
        monkeypatch.setenv("LLK_WINDOW_HASH", value)

    assert enabled() is on
