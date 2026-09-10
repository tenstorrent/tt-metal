#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
SQLite output back end for tt-triage.

Each script's output becomes one table, named after the script, with a row per
result and a column per displayed field. Columns are INTEGER, REAL or TEXT,
chosen from the values the script produced.

A shared `diagnostics` table collects the check failures, warnings and script
errors reported alongside those results.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any, Callable, Iterable

from serializers import OutputSerializer, strip_rich_markup, extract_table_data


def quote_identifier(name: str) -> str:
    """Quote a table or column name, doubling any embedded quote.

    Column names are the script's display headers verbatim, so they can contain
    spaces, symbols or reserved words; quoting makes those legal.
    """
    return '"' + name.replace('"', '""') + '"'


def _check_for_duplicates(script: str, headers: Iterable[str]) -> None:
    """Raise if a script serializes two fields under the same column name."""
    seen: dict[str, str] = {}
    for header in headers:
        previous = seen.get(header.lower())
        if previous is not None:
            detail = (
                f"duplicate column {header!r}"
                if previous == header
                else f"columns {previous!r} and {header!r} differ only in case, which SQLite treats as one"
            )
            raise ValueError(f"{script}: {detail}. Drop the redundant triage_field.")
        seen[header.lower()] = header


def _check_row_widths(script: str, columns: list[str], rows: list[list[str]]) -> None:
    """Raise if a row does not line up with the header."""
    for index, row in enumerate(rows):
        if len(row) != len(columns):
            raise ValueError(
                f"{script}: row {index} has {len(row)} values but the header has {len(columns)} "
                f"columns. Return one dataclass type per result list."
            )


def _sql_value(serialized: Any, unserialized: Any) -> Any:
    """Pick what gets stored for one cell: a number, or the serialized string.

    Numbers stay numbers so they remain comparable - a `hex_serializer` field
    would otherwise arrive as "0x1234" and `WHERE pc > 4096` could never work.
    `bool` is an `int` subclass, so it stores as 0/1.
    """
    if unserialized is None:
        return None
    if isinstance(unserialized, int):
        # SQLite's INTEGER is signed 64-bit and there is no wider type, so a
        # value past that cannot be bound. Falling back to the string makes
        # column declared as TEXT, which keeps it exact.
        _INT64_MIN = -(2**63)
        _INT64_MAX = 2**63 - 1
        return unserialized if _INT64_MIN <= unserialized <= _INT64_MAX else serialized
    if isinstance(unserialized, float):
        return unserialized
    return serialized


def _column_type(values: Iterable[Any]) -> str:
    """Choose the SQL type for a column from the values going into it.

    INTEGER or REAL when every value is numeric, TEXT otherwise. NULLs are
    skipped, since they fit any type.
    """
    seen_int = seen_float = False
    for value in values:
        if value is None:
            continue
        if isinstance(value, int):
            seen_int = True
        elif isinstance(value, float):
            seen_float = True
        else:
            return "TEXT"
    if seen_float:
        return "REAL"
    return "INTEGER" if seen_int else "TEXT"


DIAGNOSTICS_TABLE = "diagnostics"


class SqliteSerializer(OutputSerializer):
    """Writes each script's rows into a table named after the script."""

    def __init__(self, path: str, verbose_level_getter: Callable[[], int]):
        path = os.path.abspath(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        if os.path.exists(path):
            raise FileExistsError(f"{path} already exists - remove it or pass a different path")
        self.path = path
        self._verbose_getter = verbose_level_getter
        self._con = sqlite3.connect(path)
        # Created up front so it can always be queried, even on a clean run.
        self._con.execute(
            f"CREATE TABLE {quote_identifier(DIAGNOSTICS_TABLE)} "
            f'("Script" TEXT NOT NULL, "Severity" TEXT NOT NULL, "Message" TEXT NOT NULL)'
        )

    def emit(
        self,
        script_name: str | None,
        execution_time: str,
        result: Any,
        failures: list[str],
        warnings: list[str],
        script_failed: bool,
        failure_message: str | None,
        documentation: str | None,
    ) -> None:
        assert script_name is not None, "cannot serialize a result without a script name"
        self._insert_diagnostics(script_name, failures, warnings, script_failed, failure_message)

        table_data = extract_table_data(result, self._verbose_getter())
        if table_data is None or not table_data.rows:
            # Check-only scripts return nothing; their diagnostics are the output.
            self._con.commit()
            return
        table = os.path.splitext(script_name)[0]
        columns = table_data.columns
        _check_for_duplicates(script_name, columns)
        _check_row_widths(script_name, columns, table_data.rows)

        rows: list[list[Any]] = []
        for serialized, unserialized in zip(table_data.rows, table_data.unserialized_values):
            # Strip the Rich markup some scripts embed in cell values.
            rows.append(
                [
                    strip_rich_markup(value) if isinstance(value, str) else value
                    for value in (_sql_value(text, original) for text, original in zip(serialized, unserialized))
                ]
            )

        spec = ", ".join(f"{quote_identifier(c)} {_column_type(r[i] for r in rows)}" for i, c in enumerate(columns))
        self._con.execute(f"CREATE TABLE IF NOT EXISTS {quote_identifier(table)} ({spec})")

        names = ", ".join(quote_identifier(c) for c in columns)
        marks = ", ".join("?" for _ in columns)
        self._con.executemany(f"INSERT INTO {quote_identifier(table)} ({names}) VALUES ({marks})", rows)
        self._con.commit()

    def record_diagnostics(
        self,
        script_name: str,
        failures: list[str],
        warnings: list[str],
        script_failed: bool,
        failure_message: str | None,
    ) -> None:
        self._insert_diagnostics(script_name, failures, warnings, script_failed, failure_message)
        self._con.commit()

    def _insert_diagnostics(
        self,
        script_name: str,
        failures: list[str],
        warnings: list[str],
        script_failed: bool,
        failure_message: str | None,
    ) -> None:
        rows = [(script_name, "failure", message) for message in failures]
        rows += [(script_name, "warning", message) for message in warnings]
        if script_failed:
            rows.append((script_name, "error", failure_message or "script failed"))
        if not rows:
            return
        self._con.executemany(
            f"INSERT INTO {quote_identifier(DIAGNOSTICS_TABLE)} VALUES (?, ?, ?)",
            [(script, severity, strip_rich_markup(message)) for script, severity, message in rows],
        )

    def close(self) -> None:
        try:
            self._con.commit()
        finally:
            self._con.close()
