# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the TILE_LOOP window search (helpers/perf/window_hash.py).

Run: pytest test_perf_window_hash.py
"""

from helpers.perf.window_hash import _hash16, _window

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


def test_zone_calls_bound_the_window():
    insns = _loop(
        [("jal", "200 <zone_reserve>")],
        [
            ("lui", f"a0,{hex(_ID >> 12)}"),
            ("addi", f"a0,a0,{_ID & 0xFFF}"),
            ("jal", "300 <zone_record>"),
        ],
    )

    assert _window(insns, _ELF) == [1, 2, 3, 4, 5, 6]


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
