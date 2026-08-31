# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from models.perf.benchmarking_utils import BenchmarkProfiler


@dataclass(frozen=True)
class SectionStart:
    name: str


@dataclass(frozen=True)
class SectionEnd:
    name: str


@dataclass(frozen=True)
class DenoiseStep:
    step: int
    total: int
    sigma: float


PipelineEvent = SectionStart | SectionEnd | DenoiseStep
PipelineEventCallback = Callable[[PipelineEvent], None]


def null_callback(_event: PipelineEvent) -> None:
    pass


@contextmanager
def event_section(on_event: PipelineEventCallback, name: str) -> Iterator[None]:
    """Fire `SectionStart` / `SectionEnd` around a stage. `SectionEnd` always runs, even on raise."""
    on_event(SectionStart(name))
    try:
        yield
    finally:
        on_event(SectionEnd(name))


def profiler_event_callback(profiler: BenchmarkProfiler, iteration: int) -> PipelineEventCallback:
    def on_event(event: PipelineEvent) -> None:
        if isinstance(event, SectionStart):
            profiler.start(event.name, iteration)
        elif isinstance(event, SectionEnd):
            profiler.end(event.name, iteration)

    return on_event


def log_event_section(event: PipelineEvent) -> None:
    if isinstance(event, SectionStart):
        logger.info(f"[>>] {event.name}")
    elif isinstance(event, SectionEnd):
        logger.info(f"[<<] {event.name}")
    elif isinstance(event, DenoiseStep):
        logger.info(f"[~~] Denoise step {event.step}/{event.total} (sigma={event.sigma:.4f})")
