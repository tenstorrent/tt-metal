# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/ulp_affected.py``: which sweep cells a change can
move."""

import struct

import pytest
from helpers import ulp_affected
from helpers.format_config import DataFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, MathOperation
from helpers.ulp_affected import Built, loaded_digest, main, read_built, select
from helpers.ulp_table import TABLE_PATHS, BudgetKey, Row, measured_arch

F16B = DataFormat.Float16_b
ABS = MathOperation.Abs
TABLE = {ABS: (Row(input=F16B, output=F16B, measured=1),)}
CELL = Built(
    "test_unary_sfpu_ulp.py::sweep[Abs]",
    ABS,
    BudgetKey(ApproximationMode.No, F16B, F16B, DestAccumulation.No),
    "sources/t.cpp/aaaa",
)


def _elf(text=b"\x13\x00\x00\x00", debug=b"/build/one", bss=16):
    """A 32-bit little-endian ELF with a loaded .text, a .bss and an unloaded debug
    section."""
    sections = [
        (0, 0, 0, 0, 0),
        (1, 0x6, 0x1000, 52, len(text)),  # PROGBITS, ALLOC|EXEC
        (8, 0x3, 0x2000, 52 + len(text), bss),  # NOBITS, ALLOC|WRITE
        (1, 0x30, 0, 52 + len(text), len(debug)),  # not ALLOC
    ]
    header = b"\x7fELF\x01\x01\x01" + bytes(9)
    header += struct.pack(
        "<HHIIIIIHHHHHH",
        2,
        0xF3,
        1,
        0,
        0,
        52 + len(text + debug),
        0,
        52,
        0,
        0,
        40,
        4,
        0,
    )
    table = b"".join(struct.pack("<10I", 0, *s, 0, 0, 4, 0) for s in sections)
    return header + text + debug + table


def _build(root, **elf):
    for path in (root / CELL.build / "elf" / "math.elf", root / "shared/elf/brisc.elf"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_elf(**elf) if path.name == "math.elf" else _elf())


def test_debug_info_does_not_count_and_loaded_bytes_do(tmp_path):
    paths = [tmp_path / f"{n}.elf" for n in "abcd"]
    for path, elf in zip(
        paths,
        [
            _elf(),
            _elf(debug=b"/build/two"),
            _elf(text=b"\x13\x01\x00\x00"),
            _elf(bss=32),
        ],
    ):
        path.write_bytes(elf)
    a, debug, text, bss = map(loaded_digest, paths)
    assert a == debug and len({a, text, bss}) == 3


@pytest.mark.parametrize(
    "head_elf, base_elf, base_table, why",
    [
        ({}, {}, TABLE, None),
        (dict(text=b"\x13\x01\x00\x00"), {}, TABLE, "rebuilt"),
        ({}, None, TABLE, "new build"),
        ({}, {}, {}, "row changed"),
        (dict(bss=64), {}, {}, "rebuilt and row changed"),
    ],
    ids=["nothing", "rebuilt", "new-build", "row-changed", "both"],
)
def test_select(tmp_path, head_elf, base_elf, base_table, why):
    head, base = tmp_path / "head", tmp_path / "base"
    _build(head, debug=b"/head", **head_elf)
    if base_elf is not None:
        _build(base, debug=b"/base", **base_elf)
    picked = select([CELL], head, base, base_table, TABLE)
    assert picked == ([] if why is None else [(CELL, why)])


def test_a_recorded_cell_with_no_build_is_an_error(tmp_path):
    with pytest.raises(ValueError):  # allow-pytest.raises: host-only test
        select([CELL], tmp_path, tmp_path, TABLE, TABLE)


def test_records_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(ulp_affected, "VARIANTS_PATH", tmp_path / "built.jsonl")
    ulp_affected.VARIANTS_PATH.write_text("")
    for _ in range(2):
        ulp_affected.record_built(CELL.node, CELL.op, CELL.key, CELL.build)
    assert read_built(tmp_path / "built.jsonl") == [CELL, CELL]


def test_main_writes_the_picked_nodes(tmp_path, monkeypatch, capsys):
    head, base = tmp_path / "head", tmp_path / "base"
    _build(head, text=b"\x13\x01\x00\x00")
    _build(base)
    monkeypatch.setattr(ulp_affected, "VARIANTS_PATH", tmp_path / "built.jsonl")
    ulp_affected.record_built(CELL.node, CELL.op, CELL.key, CELL.build)
    argv = [
        *("--records", str(tmp_path / "built.jsonl")),
        *("--base", str(base), "--head", str(head)),
        # The head table is this session's; the same file as the base changes no row.
        *("--base-table", str(TABLE_PATHS[measured_arch()])),
        *("--nodes", str(tmp_path / "nodes.txt")),
    ]
    assert main(argv) == 0
    assert (tmp_path / "nodes.txt").read_text() == f"{CELL.node}\n"
    assert "1 of 1 sweep cells to run" in capsys.readouterr().out
