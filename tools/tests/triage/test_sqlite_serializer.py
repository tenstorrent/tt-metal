# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Unit tests for the SQLite output back end. These need no hardware - they feed
# synthetic dataclasses through the serializer and query the result back.

from dataclasses import dataclass
import os
import sqlite3
import sys

import pytest

metal_home = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
triage_home = os.path.join(metal_home, "tools", "triage")
sys.path.insert(0, triage_home)


from sqlite_serializer import (
    DIAGNOSTICS_TABLE,
    SqliteSerializer,
    _check_for_duplicates,
    _check_row_widths,
    _column_type,
    _sql_value,
    quote_identifier,
)
from triage import CheckEntry, CheckType, hex_serializer, recurse_field, triage_field


class FakeNocBlock:
    def __init__(self, block_type: str):
        self.block_type = block_type


class FakeDevice:
    def __init__(self, id: int):
        self.id = id


class FakeLocation:
    """Stand-in for an `OnChipCoordinate`, which needs a real device to build."""

    def __init__(self, device: FakeDevice, block_type: str, user_str: str):
        self.device = device
        self.noc_block = FakeNocBlock(block_type)
        self._user_str = user_str

    def to_user_str(self) -> str:
        return self._user_str


def error(message, device=None, location=None, risc_name=None):
    return CheckEntry(message, CheckType.ERROR, device=device, location=location, risc_name=risc_name)


def warning(message, device=None, location=None, risc_name=None):
    return CheckEntry(message, CheckType.WARNING, device=device, location=location, risc_name=risc_name)


@dataclass
class Row:
    name: str = triage_field("Kernel Name")
    pc: "int | None" = triage_field("PC", hex_serializer)
    rate: float = triage_field("Heartbeats/s")
    enabled: bool = triage_field("Preload")


@dataclass
class Inner:
    loc: str = triage_field("Loc")


@dataclass
class DuplicateLoc:
    loc: str = triage_field("Loc")
    inner: Inner = recurse_field()


@dataclass
class Narrow:
    a: str = triage_field("A")


@dataclass
class Wide:
    a: str = triage_field("A")
    b: str = triage_field("B")


def write(path, script_name, result, **kwargs):
    """Serialize one result and return a connection to the finished database."""
    serializer = SqliteSerializer(str(path), lambda: 0)
    serializer.emit(
        script_name=script_name,
        execution_time="",
        result=result,
        checks=kwargs.get("checks", []),
        script_failed=kwargs.get("script_failed", False),
        failure_message=kwargs.get("failure_message"),
        documentation=None,
    )
    serializer.close()
    return sqlite3.connect(str(path))


def test_quote_identifier_escapes_embedded_quote():
    assert quote_identifier("Kernel Name") == '"Kernel Name"'
    assert quote_identifier('we"ird') == '"we""ird"'


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 3], "INTEGER"),
        ([1, None, 3], "INTEGER"),
        ([1, 2.5], "REAL"),
        ([1.5, 2.5], "REAL"),
        (["a", "b"], "TEXT"),
        ([1, "a"], "TEXT"),
        ([None, None], "TEXT"),
        ([True, False], "INTEGER"),
    ],
)
def test_column_type(values, expected):
    assert _column_type(values) == expected


def test_sql_value_keeps_numbers_and_maps_none_to_null():
    # The serialized form is what the field's serializer produced; the
    # unserialized form is the attribute behind it.
    assert _sql_value("0x1000", 4096) == 4096
    assert _sql_value("2.5", 2.5) == 2.5
    assert _sql_value("N/A", None) is None
    assert _sql_value("matmul", "matmul") == "matmul"
    # bool is an int subclass, so it stores as 0/1
    assert _sql_value("True", True) == 1
    assert _sql_value("False", False) == 0


def test_sql_value_falls_back_to_string_past_int64():
    too_big = 2**64 - 1
    assert _sql_value(str(too_big), too_big) == str(too_big)
    assert _sql_value(str(2**63 - 1), 2**63 - 1) == 2**63 - 1


def test_check_for_duplicates():
    _check_for_duplicates("demo.py", ["Dev", "Loc", "RiscV"])
    with pytest.raises(ValueError, match="duplicate column 'Loc'"):  # allow-pytest.raises: no expect_error fixture
        _check_for_duplicates("demo.py", ["Dev", "Loc", "Loc"])
    # SQL column names are case insensitive, so these collide too
    with pytest.raises(ValueError, match="differ only in case"):  # allow-pytest.raises: no expect_error fixture
        _check_for_duplicates("demo.py", ["Dev", "Loc", "loc"])


def test_check_row_widths():
    _check_row_widths("demo.py", ["A", "B"], [["1", "2"], ["3", "4"]])
    with pytest.raises(ValueError, match="row 1 has 1 values"):  # allow-pytest.raises: no expect_error fixture
        _check_row_widths("demo.py", ["A", "B"], [["1", "2"], ["3"]])


def test_round_trip_column_types_and_values(tmp_path):
    con = write(
        tmp_path / "t.db",
        "dump_callstacks.py",
        [Row("matmul", 0x1000, 9.5, True), Row("add", None, 0.5, False)],
    )
    schema = con.execute("SELECT sql FROM sqlite_master WHERE name='dump_callstacks'").fetchone()[0]
    assert '"Kernel Name" TEXT' in schema
    assert '"PC" INTEGER' in schema
    assert '"Heartbeats/s" REAL' in schema
    assert '"Preload" INTEGER' in schema

    # A hex-serialized field stays numerically comparable, and NULL is NULL.
    assert con.execute('SELECT count(*) FROM dump_callstacks WHERE "PC" > 4000').fetchone()[0] == 1
    assert con.execute('SELECT count(*) FROM dump_callstacks WHERE "PC" IS NULL').fetchone()[0] == 1
    assert con.execute('SELECT count(*) FROM dump_callstacks WHERE "Preload"').fetchone()[0] == 1
    assert con.execute('SELECT printf(\'0x%X\', "PC") FROM dump_callstacks WHERE "PC" IS NOT NULL').fetchone() == (
        "0x1000",
    )


def test_column_names_are_the_display_headers(tmp_path):
    con = write(tmp_path / "t.db", "demo.py", [Row("matmul", 1, 1.0, True)])
    names = [row[1] for row in con.execute("PRAGMA table_info(demo)")]
    assert names == ["Kernel Name", "PC", "Heartbeats/s", "Preload"]


def test_diagnostics_table_exists_even_when_empty(tmp_path):
    con = write(tmp_path / "t.db", "demo.py", [Row("matmul", 1, 1.0, True)])
    assert con.execute(f"SELECT count(*) FROM {quote_identifier(DIAGNOSTICS_TABLE)}").fetchone()[0] == 0


def test_check_only_script_records_diagnostics_but_no_table(tmp_path):
    device0 = FakeDevice(0)
    con = write(
        tmp_path / "t.db",
        "check_noc_status.py",
        None,
        checks=[
            error(
                "[error]NOC0 mismatch[/]",
                device=device0,
                location=FakeLocation(device0, "tensix", "1-1 (0,0)"),
                risc_name="brisc",
            ),
            warning("skipping", device=FakeDevice(1)),
        ],
    )
    tables = [row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert tables == [DIAGNOSTICS_TABLE]
    rows = con.execute(f"SELECT * FROM {quote_identifier(DIAGNOSTICS_TABLE)}").fetchall()
    assert rows == [
        ("check_noc_status.py", "error", 0, "1-1 (0,0)", "brisc", "NOC0 mismatch"),
        ("check_noc_status.py", "warning", 1, None, None, "skipping"),
    ]


def test_diagnostics_columns_are_the_reported_context(tmp_path):
    con = write(tmp_path / "t.db", "check_l1_status.py", None, checks=[error("no context here")])
    names = [row[1] for row in con.execute(f"PRAGMA table_info({quote_identifier(DIAGNOSTICS_TABLE)})")]
    assert names == ["Script", "Type", "Device", "Location", "RISC", "Message"]
    # Only `message` is required, so the context columns are NULL when unreported.
    assert con.execute(f"SELECT * FROM {quote_identifier(DIAGNOSTICS_TABLE)}").fetchall() == [
        ("check_l1_status.py", "error", None, None, None, "no context here")
    ]


def test_same_message_from_many_cores_groups_into_one_row(tmp_path):
    # The point of splitting the context out of the message: a report repeated
    # across cores collapses to a single row.
    device = FakeDevice(0)
    con = write(
        tmp_path / "t.db",
        "check_core_magic.py",
        None,
        checks=[
            error("Magic value mismatch", device=device, location=FakeLocation(device, "tensix", f"1-{x} (0,{x})"))
            for x in range(5)
        ],
    )
    grouped = con.execute(
        f'SELECT "Message", count(*) FROM {quote_identifier(DIAGNOSTICS_TABLE)} GROUP BY "Message"'
    ).fetchall()
    assert grouped == [("Magic value mismatch", 5)]


def test_failed_script_is_recorded_as_error(tmp_path):
    con = write(
        tmp_path / "t.db",
        "dump_callstacks.py",
        None,
        script_failed=True,
        failure_message="Skipped: dependency inspector_data.py failed.",
    )
    rows = con.execute(f'SELECT "Type", "Message" FROM {quote_identifier(DIAGNOSTICS_TABLE)}').fetchall()
    assert rows == [("error", "Skipped: dependency inspector_data.py failed.")]


def test_duplicate_column_raises_through_emit(tmp_path):
    with pytest.raises(ValueError, match="duplicate column 'Loc'"):  # allow-pytest.raises: no expect_error fixture
        write(tmp_path / "t.db", "dump_fast_dispatch.py", [DuplicateLoc("1-1", Inner("1-1"))])


def test_ragged_rows_raise_through_emit(tmp_path):
    with pytest.raises(ValueError, match="values but the header has"):  # allow-pytest.raises: no expect_error fixture
        write(tmp_path / "t.db", "demo.py", [Narrow("1"), Wide("2", "3")])


def test_existing_database_is_not_reused(tmp_path):
    path = tmp_path / "t.db"
    write(path, "demo.py", [Row("matmul", 1, 1.0, True)])
    with pytest.raises(FileExistsError):  # allow-pytest.raises: no expect_error fixture
        SqliteSerializer(str(path), lambda: 0)
    # the original database is left untouched
    con = sqlite3.connect(str(path))
    assert con.execute("SELECT count(*) FROM demo").fetchone()[0] == 1


def test_record_diagnostics_without_a_result(tmp_path):
    # main() prints skipped scripts and failed providers itself, so they arrive
    # through record_diagnostics rather than emit.
    path = tmp_path / "t.db"
    serializer = SqliteSerializer(str(path), lambda: 0)
    serializer.record_diagnostics("inspector_data.py", [], True, "Inspector unavailable")
    serializer.record_diagnostics("dump_configuration.py", [], True, "Skipped: dependency failed.")
    serializer.record_diagnostics("dispatcher_data.py", [warning("no rank", device=FakeDevice(0))], False, None)
    serializer.close()

    con = sqlite3.connect(str(path))
    rows = con.execute(
        f'SELECT "Script", "Type", "Device", "Message" FROM {quote_identifier(DIAGNOSTICS_TABLE)}'
    ).fetchall()
    assert rows == [
        ("inspector_data.py", "error", None, "Inspector unavailable"),
        ("dump_configuration.py", "error", None, "Skipped: dependency failed."),
        ("dispatcher_data.py", "warning", 0, "no rank"),
    ]
    # no script produced rows, so the only table is diagnostics
    tables = [row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    assert tables == [DIAGNOSTICS_TABLE]


def test_record_diagnostics_with_nothing_to_record(tmp_path):
    path = tmp_path / "t.db"
    serializer = SqliteSerializer(str(path), lambda: 0)
    serializer.record_diagnostics("elfs_cache.py", [], False, None)
    serializer.close()
    con = sqlite3.connect(str(path))
    assert con.execute(f"SELECT count(*) FROM {quote_identifier(DIAGNOSTICS_TABLE)}").fetchone()[0] == 0


def test_multiple_scripts_share_one_database(tmp_path):
    path = tmp_path / "t.db"
    serializer = SqliteSerializer(str(path), lambda: 0)
    for script in ("check_arc.py", "device_info.py"):
        serializer.emit(script, "", [Row("matmul", 1, 1.0, True)], [], False, None, None)
    serializer.close()
    con = sqlite3.connect(str(path))
    tables = sorted(row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'"))
    assert tables == sorted(["check_arc", "device_info", DIAGNOSTICS_TABLE])
