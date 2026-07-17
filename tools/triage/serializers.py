#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Output serializers for tt-triage script results.

Each script's result flows through `serialize_result()` in `triage.py`, which
delegates to an `OutputSerializer`. `RichSerializer` renders Rich tables;
`CsvSerializer` renders the same data as CSV-formatted tables,
which is cheaper for machine readers (LLMs, grep, diff) than the box-drawing format.
"""

from __future__ import annotations

import csv
import io
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
    dataclasses - those are emitted as plain strings by serializers.
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
    """Turns one script's result into output. Subclasses pick the format
    (Rich tables, CSV, future JSON, …). The *destination* is handled by a
    `Sink` instance the serializer holds."""

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
        pass

    def close(self) -> None:
        """Release any owned resources (e.g. file handles). Default is a no-op."""
        pass


class Sink(ABC):
    """Where rendered text goes. Decouples *format* (Serializer) from
    *destination* (Sink) so each can vary independently."""

    @abstractmethod
    def write_line(self, line: str) -> None:
        pass

    def close(self) -> None:
        """Release any owned resources (e.g. file handles). Default is a no-op."""
        pass


class ConsoleSink(Sink):
    """`Sink` backed by a Rich console. Supports plain lines and Rich
    renderables (tables, panels) - the latter is needed by `RichSerializer`."""

    def __init__(self, console: Any):
        self._console = console

    def write_line(self, line: str) -> None:
        self._console.print(line, markup=False, highlight=False)

    def write_rich(self, renderable: Any) -> None:
        self._console.print(renderable)


class FileSink(Sink):
    """`Sink` backed by a file on disk. Closes the file in `close()`."""

    def __init__(self, path: str):
        path = os.path.abspath(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._file = open(path, "w", encoding="utf-8")

    def write_line(self, line: str) -> None:
        self._file.write(line)
        self._file.write("\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


class RichSerializer(OutputSerializer):
    """Renders results as Rich tables."""

    def __init__(self, sink: ConsoleSink, utils_module: Any, verbose_level_getter: Callable[[], int]):
        self._sink = sink
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
            return

        table_data = extract_table_data(result, self._verbose_getter())
        if table_data is None:
            utils.INFO(f"  {result}")
            return

        table = Table()
        for col in table_data.columns:
            table.add_column(col, justify="left")
        for row in table_data.rows:
            table.add_row(*row)
        self._sink.write_rich(table)

    def close(self) -> None:
        self._sink.close()


def _one_line(s: str) -> str:
    return s.replace("\n", "\\n").replace("\r", "\\r")


def _strip_rich_markup(s: str) -> str:
    """Strip Rich markup tags (e.g. `[warning]...[/]`) from a string.

    Some script messages contain bracketed text that looks like markup but
    isn't (e.g. `[!] core`, `[0x...]`, coordinate strings like `[1-1 (0,0)]`).
    Rich raises `MarkupError` on those - fall back to the original string in
    that case so we never crash on otherwise valid triage output.
    """
    from rich.errors import MarkupError
    from rich.text import Text

    try:
        return Text.from_markup(s).plain
    except MarkupError:
        return s


class CsvSerializer(OutputSerializer):
    """Renders results as CSV-formatted tables."""

    def __init__(self, sink: Sink, verbose_level_getter: Callable[[], int]):
        self._sink = sink
        self._verbose_getter = verbose_level_getter

    def _print(self, line: str = "") -> None:
        # Strip Rich markup (colour tags like `[blue]0x...[/]` embedded in cell
        # values), then escape embedded newlines so each record is exactly one
        # physical line.
        self._sink.write_line(_one_line(_strip_rich_markup(line)))

    def _print_csv_row(self, row: list[str]) -> None:
        buf = io.StringIO()
        csv.writer(buf, lineterminator="").writerow(row)
        self._print(buf.getvalue())

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
        if script_name is not None:
            self._print()
            self._print(f"{script_name}{execution_time}:")

        if result is None:
            if len(failures) > 0 or script_failed:
                self._print("  fail")
                for failure in failures:
                    self._print(f"    {failure}")
                if script_failed and failure_message:
                    self._print(f"    {failure_message}")
            else:
                self._print("  pass")
                for warning in warnings:
                    self._print(f"    {warning}")
            return

        for failure in failures:
            self._print(f"  {failure}")
        for warning in warnings:
            self._print(f"  {warning}")

        if isinstance(result, list) and len(result) == 0:
            self._print("  No results found.")
            return

        table_data = extract_table_data(result, self._verbose_getter())
        if table_data is None:
            self._print(f"  {result}")
            return

        self._print_csv_row(table_data.columns)
        for row in table_data.rows:
            self._print_csv_row(row)

    def close(self) -> None:
        self._sink.close()


class MultiSerializer(OutputSerializer):
    """Fans `emit` / `close` out to several serializers (e.g. Rich + CSV file)."""

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
