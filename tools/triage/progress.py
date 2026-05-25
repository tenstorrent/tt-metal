#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Progress reporting abstraction. A runner asks the active `ProgressReporter`
for a `session(total)` context manager and calls `.describe()` / `.advance()`
per item. Concrete reporters decide how (or whether) to render."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Iterator


class ProgressSession(ABC):
    @abstractmethod
    def describe(self, description: str) -> None:
        ...

    @abstractmethod
    def advance(self) -> None:
        ...


class ProgressReporter(ABC):
    @abstractmethod
    @contextmanager
    def session(self, total: int) -> Iterator[ProgressSession]:
        ...


class NullProgressSession(ProgressSession):
    def describe(self, description: str) -> None:
        pass

    def advance(self) -> None:
        pass


class NullProgressReporter(ProgressReporter):
    @contextmanager
    def session(self, total: int) -> Iterator[ProgressSession]:
        yield NullProgressSession()


class RichProgressSession(ProgressSession):
    def __init__(self, progress: Any, task_id: Any):
        self._progress = progress
        self._task_id = task_id

    def describe(self, description: str) -> None:
        self._progress.update(self._task_id, description=description)

    def advance(self) -> None:
        self._progress.advance(self._task_id)


class RichProgressReporter(ProgressReporter):
    """Single-process Rich-backed reporter. Supports nested sessions naturally
    (Rich Progress with multiple tasks renders them stacked)."""

    def __init__(self, console: Any):
        self._console = console

    @contextmanager
    def session(self, total: int) -> Iterator[ProgressSession]:
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TimeElapsedColumn(),
            BarColumn(),
            TimeRemainingColumn(),
            TextColumn("[progress.tasks]{task.completed}/{task.total}[/] [progress.description]{task.description}[/]"),
            console=self._console,
            transient=True,
        )
        with progress:
            task_id = progress.add_task("", total=total)
            yield RichProgressSession(progress, task_id)
