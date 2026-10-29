# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for ``helpers/ulp_affected.py``: which sweep cells a change can
move, judged by what the card loads and by the row each cell resolves to."""

import struct

import pytest
from helpers import ulp_affected
from helpers.format_config import DataFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, MathOperation
from helpers.ulp_affected import (
    SHARED_ELF_DIR,
    Built,
    Reason,
    build_digest,
    loaded_digest,
    main,
    read_built,
    record_built,
    select,
)
from helpers.ulp_table import BudgetKey, Row, write_table_file

F16B, F32 = DataFormat.Float16_b, DataFormat.Float32
ABS, ELU = MathOperation.Abs, MathOperation.Elu

_SHF_ALLOC = 0x2
_PROGBITS, _NOBITS = 1, 8


def _elf(text=b"\x13\x00\x00\x00", debug=b"/build/one", bss=16):
    """A minimal 32-bit little-endian ELF: a loaded ``.text``, a ``.bss`` of *bss*
    bytes, and a ``.debug_str`` the card never loads."""
    body = text + debug
    shoff = 52 + len(body)
    sections = [
        (0, 0, 0, 0, 0, 0),
        (_PROGBITS, _SHF_ALLOC | 0x4, 0x1000, 52, len(text), 0),
        (_NOBITS, _SHF_ALLOC | 0x1, 0x2000, 52 + len(text), bss, 0),
        (_PROGBITS, 0x30, 0, 52 + len(text), len(debug), 0),
    ]
    header = b"\x7fELF" + bytes([1, 1, 1]) + bytes(9)
    header += struct.pack(
        "<HHIIIIIHHHHHH", 2, 0xF3, 1, 0, 0, shoff, 0, 52, 0, 0, 40, len(sections), 0
    )
    table = b"".join(
        struct.pack("<IIIIIIIIII", 0, kind, flags, addr, off, size, 0, 0, 4, 0)
        for kind, flags, addr, off, size, _ in sections
    )
    return header + body + table


def _write_build(root, build, **elf):
    elf_dir = root / build / "elf"
    elf_dir.mkdir(parents=True, exist_ok=True)
    for name in ("unpack", "math", "pack"):
        (elf_dir / f"{name}.elf").write_bytes(_elf(**elf) if name == "math" else _elf())
    shared = root / SHARED_ELF_DIR
    shared.mkdir(parents=True, exist_ok=True)
    if not (shared / "brisc.elf").exists():
        (shared / "brisc.elf").write_bytes(_elf())


def _cell(op=ABS, build="sources/t.cpp/aaaa", input_format=F16B):
    key = BudgetKey(
        approx_mode=ApproximationMode.No,
        input_format=input_format,
        output_format=input_format,
        dest_acc=DestAccumulation.No,
    )
    return Built(
        f"test_unary_sfpu_ulp.py::sweep[{op.name}-{input_format.name}]", op, key, build
    )


_TABLE = {ABS: (Row(input=F16B, output=F16B, measured=1),)}


# ── What the card loads ───────────────────────────────────────────────────────


def test_debug_info_is_not_what_the_card_loads(tmp_path):
    a, b = tmp_path / "a.elf", tmp_path / "b.elf"
    a.write_bytes(_elf(debug=b"/runner/one/tt-llk-build"))
    b.write_bytes(_elf(debug=b"/runner/two/tt-llk-build"))
    assert a.read_bytes() != b.read_bytes()
    assert loaded_digest(a) == loaded_digest(b)


@pytest.mark.parametrize(
    "changed", [dict(text=b"\x13\x01\x00\x00"), dict(bss=32)], ids=["text", "bss-size"]
)
def test_a_loaded_section_that_changes_changes_the_digest(tmp_path, changed):
    a, b = tmp_path / "a.elf", tmp_path / "b.elf"
    a.write_bytes(_elf())
    b.write_bytes(_elf(**changed))
    assert loaded_digest(a) != loaded_digest(b)


def test_a_file_that_is_not_an_elf32_is_refused(tmp_path):
    path = tmp_path / "x.elf"
    path.write_bytes(b"\x7fELF" + bytes([2, 1]) + bytes(64))
    with pytest.raises(  # allow-pytest.raises: host-only test
        ValueError, match="32-bit little-endian"
    ):
        loaded_digest(path)


def test_a_shared_elf_belongs_to_every_build(tmp_path):
    for root in (tmp_path / "base", tmp_path / "head"):
        _write_build(root, "sources/t.cpp/aaaa")
    (tmp_path / "head" / SHARED_ELF_DIR / "brisc.elf").write_bytes(
        _elf(text=b"\x13\x02\x00\x00")
    )
    assert build_digest(tmp_path / "base", "sources/t.cpp/aaaa") != build_digest(
        tmp_path / "head", "sources/t.cpp/aaaa"
    )


# ── Which cells a change can move ─────────────────────────────────────────────


def _roots(tmp_path):
    return tmp_path / "head", tmp_path / "base"


def test_nothing_rebuilt_and_no_row_changed_selects_nothing(tmp_path):
    head, base = _roots(tmp_path)
    for root in (head, base):
        _write_build(root, "sources/t.cpp/aaaa", debug=str(root).encode())
    assert select([_cell()], head, base, _TABLE, _TABLE) == []


def test_a_rebuilt_kernel_selects_every_cell_of_its_build_only(tmp_path):
    head, base = _roots(tmp_path)
    _write_build(base, "sources/t.cpp/aaaa")
    _write_build(head, "sources/t.cpp/aaaa", text=b"\x13\x01\x00\x00")
    for root in (head, base):
        _write_build(root, "sources/t.cpp/bbbb")
    moved = [_cell(build="sources/t.cpp/aaaa", input_format=f) for f in (F16B, F32)]
    still = _cell(op=ELU, build="sources/t.cpp/bbbb")
    picked = select([*moved, still], head, base, _TABLE, _TABLE)
    assert [s.built for s in picked] == moved
    assert all(s.reasons == (Reason.REBUILT,) for s in picked)


def test_a_build_the_base_lacks_is_new(tmp_path):
    head, base = _roots(tmp_path)
    _write_build(head, "sources/t.cpp/aaaa")
    (picked,) = select([_cell()], head, base, _TABLE, _TABLE)
    assert picked.reasons == (Reason.NEW_BUILD,)


@pytest.mark.parametrize(
    "head_table",
    [
        {ABS: (Row(input=F16B, output=F16B, measured=0),)},
        {},
    ],
    ids=["tightened", "removed"],
)
def test_a_changed_row_selects_its_cells_though_nothing_rebuilt(tmp_path, head_table):
    head, base = _roots(tmp_path)
    for root in (head, base):
        _write_build(root, "sources/t.cpp/aaaa")
    (picked,) = select([_cell()], head, base, _TABLE, head_table)
    assert picked.reasons == (Reason.ROW,)


def test_a_row_that_newly_covers_a_cell_selects_it(tmp_path):
    head, base = _roots(tmp_path)
    for root in (head, base):
        _write_build(root, "sources/t.cpp/aaaa")
    (picked,) = select([_cell()], head, base, {}, _TABLE)
    assert picked.reasons == (Reason.ROW,)


def test_both_reasons_are_named(tmp_path):
    head, base = _roots(tmp_path)
    _write_build(base, "sources/t.cpp/aaaa")
    _write_build(head, "sources/t.cpp/aaaa", bss=64)
    (picked,) = select([_cell()], head, base, {}, _TABLE)
    assert picked.reasons == (Reason.REBUILT, Reason.ROW)


def test_a_recorded_cell_the_head_did_not_build_is_an_error(tmp_path):
    head, base = _roots(tmp_path)
    with pytest.raises(  # allow-pytest.raises: host-only test
        ValueError, match="though the sweep recorded building it"
    ):
        select([_cell()], head, base, _TABLE, _TABLE)


# ── The records and the command ───────────────────────────────────────────────


def test_records_round_trip(tmp_path, monkeypatch):
    path = tmp_path / "built.jsonl"
    monkeypatch.setattr(ulp_affected, "VARIANTS_PATH", path)
    ulp_affected.prepare_variants_file()
    cells = [_cell(), _cell(op=ELU, input_format=F32)]
    for cell in cells:
        record_built(cell.node, cell.op, cell.key, cell.build)
    assert read_built(path) == cells


def test_nothing_is_recorded_without_the_option(tmp_path, monkeypatch):
    monkeypatch.setattr(ulp_affected, "VARIANTS_PATH", None)
    cell = _cell()
    record_built(cell.node, cell.op, cell.key, cell.build)
    assert not list(tmp_path.iterdir())


def _argv(tmp_path, **extra):
    head, base = _roots(tmp_path)
    table = tmp_path / "table.yaml"
    write_table_file(_TABLE, table)
    argv = [
        "select",
        "--variants",
        str(tmp_path / "built.jsonl"),
        "--head-build",
        str(head),
        "--base-build",
        str(base),
        "--base-table",
        str(table),
        "--head-table",
        str(table),
        "--nodes",
        str(tmp_path / "nodes.txt"),
    ]
    for flag, value in extra.items():
        argv += [f"--{flag}", str(value)]
    return argv


def test_select_writes_the_nodes_and_ships_their_builds(tmp_path, capsys):
    head, base = _roots(tmp_path)
    moved, still = _cell(), _cell(op=ELU, build="sources/t.cpp/bbbb")
    _write_build(base, moved.build)
    _write_build(head, moved.build, text=b"\x13\x01\x00\x00")
    for root in (head, base):
        _write_build(root, still.build)
    (head / moved.build / "obj").mkdir()
    (tmp_path / "built.jsonl").write_text(
        "".join(c.to_json() + "\n" for c in (moved, still))
    )
    ship = tmp_path / "ship"
    assert main(_argv(tmp_path, ship=ship)) == 0
    assert (tmp_path / "nodes.txt").read_text() == f"{moved.node}\n"
    assert (ship / moved.build / "elf" / "math.elf").exists()
    assert not (ship / moved.build / "obj").exists()
    assert not (ship / still.build).exists()
    assert (ship / SHARED_ELF_DIR / "brisc.elf").exists()
    assert "1 of 2 cells, over 1 op(s): Abs." in capsys.readouterr().out


def test_select_with_nothing_to_run_writes_an_empty_list(tmp_path, capsys):
    head, base = _roots(tmp_path)
    cell = _cell()
    for root in (head, base):
        _write_build(root, cell.build)
    (tmp_path / "built.jsonl").write_text(cell.to_json() + "\n")
    assert main(_argv(tmp_path, ship=tmp_path / "ship")) == 0
    assert (tmp_path / "nodes.txt").read_text() == ""
    assert not (tmp_path / "ship").exists()
    assert "the sweep has nothing to run" in capsys.readouterr().out


def test_a_build_that_recorded_nothing_cannot_select(tmp_path, capsys):
    (tmp_path / "built.jsonl").write_text("")
    assert main(_argv(tmp_path)) == 2
    assert "records no cell" in capsys.readouterr().out
