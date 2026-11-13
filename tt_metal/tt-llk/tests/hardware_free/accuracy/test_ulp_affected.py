# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/ulp_affected.py``: which sweep cells a change can
move."""

import struct

import pytest
from helpers import ulp_affected
from helpers.format_config import DataFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, MathOperation
from helpers.ulp_affected import (
    ELF32_LE_MAGIC,
    SHF_ALLOC,
    SHT_NOBITS,
    Built,
    loaded_digest,
    main,
    read_built,
    select,
)
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

ELF32_EHDR_SIZE, ELF32_SHDR_SIZE = 52, 40
EM_RISCV = 0xF3
SHT_PROGBITS = 1
SHF_WRITE, SHF_EXECINSTR = 0x1, 0x4
SHF_MERGE_STRINGS = 0x30  # .debug_str: merged strings, not loaded


def _elf(text=b"\x13\x00\x00\x00", debug=b"/build/one", bss=16):
    """A 32-bit little-endian ELF with a loaded .text, a .bss and an unloaded debug
    section."""
    data = ELF32_EHDR_SIZE + len(text)
    sections = [
        (0, 0, 0, 0, 0),
        (SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 0x1000, ELF32_EHDR_SIZE, len(text)),
        (SHT_NOBITS, SHF_ALLOC | SHF_WRITE, 0x2000, data, bss),
        (SHT_PROGBITS, SHF_MERGE_STRINGS, 0, data, len(debug)),
    ]
    # EV_CURRENT, then e_ident's padding.
    header = ELF32_LE_MAGIC + b"\x01" + bytes(9)
    header += struct.pack(
        "<HHIIIIIHHHHHH",
        2,  # e_type: ET_EXEC
        EM_RISCV,
        1,  # e_version
        0,  # e_entry
        0,  # e_phoff
        data + len(debug),  # e_shoff
        0,  # e_flags
        ELF32_EHDR_SIZE,
        0,  # e_phentsize
        0,  # e_phnum
        ELF32_SHDR_SIZE,
        len(sections),
        0,  # e_shstrndx
    )
    table = b"".join(struct.pack("<10I", 0, *s, 0, 0, 4, 0) for s in sections)
    return header + text + debug + table


def _build(root, shared=None, **elf):
    """*root* as one build of CELL, and the shared ELF every cell loads."""
    for path, fields in (
        (root / CELL.build / "elf" / "math.elf", elf),
        (root / "shared/elf/brisc.elf", shared or {}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_elf(**fields))


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


ARCH_ROW = Row(input=F16B, output=F16B, arch=measured_arch(), measured=0)
NOTED = {ABS: (Row(input=F16B, output=F16B, measured=1, note="a note"),)}


@pytest.mark.parametrize(
    "head_elf, base_elf, base_table, head_table, why",
    [
        ({}, {}, TABLE, TABLE, None),
        (dict(text=b"\x13\x01\x00\x00"), {}, TABLE, TABLE, "rebuilt"),
        (dict(shared=dict(bss=32)), {}, TABLE, TABLE, "rebuilt"),
        ({}, None, TABLE, TABLE, "new build"),
        ({}, {}, {}, TABLE, "row changed"),
        ({}, {}, TABLE, {ABS: (*TABLE[ABS], ARCH_ROW)}, "row changed"),
        ({}, {}, TABLE, NOTED, None),
        (dict(bss=64), {}, {}, TABLE, "rebuilt and row changed"),
    ],
    ids=[
        "nothing",
        "rebuilt",
        "shared-rebuilt",
        "new-build",
        "row-changed",
        "arch-row-added",
        "note-only",
        "both",
    ],
)
def test_select(tmp_path, head_elf, base_elf, base_table, head_table, why):
    """A row keyed on the architecture judges the cell, as the sweep resolves it under
    the card's; a note judges nothing."""
    head, base = tmp_path / "head", tmp_path / "base"
    _build(head, debug=b"/head", **head_elf)
    if base_elf is not None:
        _build(base, debug=b"/base", **base_elf)
    picked = select([CELL], head, base, base_table, head_table)
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


@pytest.mark.parametrize(
    "head_elf, flags, nodes, summary",
    [
        (dict(text=b"\x13\x01\x00\x00"), [], f"{CELL.node}\n", "1 of 1"),
        ({}, [], "", "0 of 1"),
        ({}, ["--all"], f"{CELL.node}\n", "1 of 1"),
    ],
    ids=["one-picked", "none-picked", "all"],
)
def test_main_writes_the_picked_nodes(
    tmp_path, monkeypatch, capsys, head_elf, flags, nodes, summary
):
    """An empty node list is what skips the card job (`cells=0`); `--all`, for a change
    to the check itself, runs it whatever the builds say."""
    head, base = tmp_path / "head", tmp_path / "base"
    _build(head, **head_elf)
    _build(base)
    monkeypatch.setattr(ulp_affected, "VARIANTS_PATH", tmp_path / "built.jsonl")
    ulp_affected.record_built(CELL.node, CELL.op, CELL.key, CELL.build)
    argv = [
        *("--records", str(tmp_path / "built.jsonl")),
        *("--base", str(base), "--head", str(head)),
        # The head table is this session's; the same file as the base changes no row.
        *("--base-table", str(TABLE_PATHS[measured_arch()])),
        *("--nodes", str(tmp_path / "nodes.txt")),
        *flags,
    ]
    assert main(argv) == 0
    assert (tmp_path / "nodes.txt").read_text() == nodes
    assert f"{summary} sweep cells to run" in capsys.readouterr().out
