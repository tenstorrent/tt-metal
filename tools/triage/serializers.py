#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Output serializers for tt-triage script results.

Each script's result flows through `serialize_result()` in `triage.py`, which
delegates to an `OutputSerializer`. The default `RichSerializer` renders Rich
tables to the console. The `CsvSerializer` writes a single text report
file whose tables are emitted as CSV — easier and cheaper to consume for
machine readers (LLMs, grep, diff) than the Rich box-drawing format.
"""

from __future__ import annotations

import csv
import os
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Iterable


@dataclass
class TableData:
    columns: list[str]
    rows: list[list[str]]


def extract_table_data(result: Any, verbose_level: int = 0) -> TableData | None:
    """
    Convert a dataclass or list-of-dataclasses to a `TableData`.

    Returns `None` for results that aren't a dataclass or non-empty list of
    dataclasses — those are emitted as plain strings by serializers.
    """
    if not (
        is_dataclass(result)
        or (isinstance(result, list) and len(result) > 0 and all(is_dataclass(item) for item in result))
    ):
        return None

    if not isinstance(result, list):
        result = [result]

    columns: list[str] = []

    def collect_header(obj: Any, flds: Iterable[Any]) -> None:
        for fld in flds:
            metadata = fld.metadata
            if metadata.get("verbose", 0) > verbose_level:
                continue
            if metadata.get("dont_serialize"):
                continue
            if metadata.get("recurse"):
                value = getattr(obj, fld.name)
                assert is_dataclass(value)
                collect_header(value, fields(value))
            elif "serialized_name" in metadata:
                columns.append(metadata.get("serialized_name") or fld.name)

    def collect_row(row: list[str], obj: Any, flds: Iterable[Any]) -> None:
        for fld in flds:
            metadata = fld.metadata
            if metadata.get("verbose", 0) > verbose_level:
                continue
            if metadata.get("dont_serialize"):
                continue
            if metadata.get("recurse"):
                value = getattr(obj, fld.name)
                assert is_dataclass(value)
                collect_row(row, value, fields(value))
            elif "additional_fields" in metadata:
                assert all(hasattr(obj, af) for af in metadata["additional_fields"])
                all_values = [getattr(obj, fld.name)]
                all_values.extend(getattr(obj, af) for af in metadata["additional_fields"])
                assert "serializer" in metadata, "Serializer must be provided for combined field."
                row.append(metadata["serializer"](all_values))
            elif "serializer" in metadata:
                row.append(metadata["serializer"](getattr(obj, fld.name)))

    collect_header(result[0], fields(result[0]))
    rows: list[list[str]] = []
    for item in result:
        row: list[str] = []
        collect_row(row, item, fields(item))
        rows.append(row)
    return TableData(columns=columns, rows=rows)


class OutputSerializer(ABC):
    """Sink for one script's serialized result. Subclasses must implement `emit` and `close`."""

    @abstractmethod
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
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class RichSerializer(OutputSerializer):
    """Renders results as Rich tables to the console."""

    def __init__(self, console: Any, utils_module: Any, verbose_level_getter: Callable[[], int]):
        self._console = console
        self._utils = utils_module
        self._verbose_getter = verbose_level_getter

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
        from rich.table import Table

        utils = self._utils
        if script_name is not None:
            print()
            utils.INFO(f"{script_name}{execution_time}:")

        if result is None:
            if len(failures) > 0 or script_failed:
                utils.ERROR("  fail")
                for failure in failures:
                    utils.ERROR(f"    {failure}")
                if script_failed:
                    utils.ERROR(f"    {failure_message}")
                    if documentation:
                        docstring_indented = textwrap.indent(documentation.strip(), "    ")
                        utils.ERROR(f"  Script help:\n{docstring_indented}")
            else:
                utils.INFO("  pass")
                for warning in warnings:
                    utils.WARN(f"    {warning}")
            return

        for failure in failures:
            utils.ERROR(f"  {failure}")
        for warning in warnings:
            utils.WARN(f"  {warning}")

        if isinstance(result, list) and len(result) == 0:
            utils.ERROR("  No results found.")

        table_data = extract_table_data(result, self._verbose_getter())
        if table_data is None:
            utils.INFO(f"  {result}")
            return

        table = Table()
        for col in table_data.columns:
            table.add_column(col, justify="left")
        for row in table_data.rows:
            table.add_row(*row)
        self._console.print(table)

    def close(self) -> None:
        pass


def _strip_rich_markup(s: str) -> str:
    """Strip Rich markup tags (e.g. `[warning]...[/]`) from a string.

    Log helpers in utils.py wrap messages in Rich tags for console rendering,
    and some scripts embed inline markup in their messages. The Rich console
    renders these away; a plain-text file must do the equivalent stripping.
    """
    from rich.text import Text

    return Text.from_markup(s).plain


class CsvSerializer(OutputSerializer):
    """
    Writes a report with one section per script.

    Each section mirrors the console layout (`<script>:`, then `pass`/`fail`
    and any warnings/failures), then emits tabular data as a CSV block.
    """

    def __init__(self, path: str, verbose_level_getter: Callable[[], int]):
        self._path = path
        self._verbose_getter = verbose_level_getter
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._file = open(path, "w", encoding="utf-8")
        self._writer = csv.writer(self._file, lineterminator="\n")

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
        f = self._file
        if script_name is not None:
            f.write("\n")
            f.write(f"{script_name}{execution_time}:\n")

        if result is None:
            if len(failures) > 0 or script_failed:
                f.write("  fail\n")
                for failure in failures:
                    f.write(f"    {_strip_rich_markup(failure)}\n")
                if script_failed and failure_message:
                    f.write(f"    {_strip_rich_markup(failure_message)}\n")
            else:
                f.write("  pass\n")
                for warning in warnings:
                    f.write(f"    {_strip_rich_markup(warning)}\n")
            f.flush()
            return

        for failure in failures:
            f.write(f"  {_strip_rich_markup(failure)}\n")
        for warning in warnings:
            f.write(f"  {_strip_rich_markup(warning)}\n")

        if isinstance(result, list) and len(result) == 0:
            f.write("  No results found.\n")
            f.flush()
            return

        table_data = extract_table_data(result, self._verbose_getter())
        if table_data is None:
            f.write(f"  {result}\n")
            f.flush()
            return

        self._writer.writerow(table_data.columns)
        for row in table_data.rows:
            self._writer.writerow(row)
        f.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class MultiSerializer(OutputSerializer):
    """Fans a single emit out to several serializers (e.g. Rich + CSV)."""

    def __init__(self, serializers: Iterable[OutputSerializer]):
        self._serializers = list(serializers)

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
        for s in self._serializers:
            s.emit(
                script_name=script_name,
                execution_time=execution_time,
                result=result,
                failures=failures,
                warnings=warnings,
                script_failed=script_failed,
                failure_message=failure_message,
                documentation=documentation,
            )

    def close(self) -> None:
        for s in self._serializers:
            s.close()
