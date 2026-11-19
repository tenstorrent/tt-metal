# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

import ttnn

if TYPE_CHECKING:
    from collections.abc import Mapping

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


class SectionTimer:
    """Wall-clock section times from the pipeline's events. With ``sync_steps``, synchronizes
    ``mesh_device`` at the end of every denoise step so per-step times measure execution, not
    dispatch."""

    def __init__(self, mesh_device, sync_steps: bool = False) -> None:
        self.mesh_device = mesh_device
        self.sync_steps = sync_steps
        self.starts: dict[str, float] = {}
        self.durations: dict[str, float] = {}

    def __call__(self, event: PipelineEvent) -> None:
        if isinstance(event, SectionStart):
            self.starts[event.name] = time.perf_counter()
        elif isinstance(event, SectionEnd):
            if self.sync_steps and event.name.startswith("denoising_step_"):
                ttnn.synchronize_device(self.mesh_device)
            self.durations[event.name] = time.perf_counter() - self.starts[event.name]

    def step_ms(self) -> list[float]:
        """Per-step times in step order, in milliseconds."""
        steps = sorted(
            (int(k.rsplit("_", 1)[1]), v) for k, v in self.durations.items() if k.startswith("denoising_step_")
        )
        return [v * 1000 for _, v in steps]


_INDEXED_SECTION = re.compile(r"_\d+$")


def log_section_durations(
    profiler: BenchmarkProfiler,
    iteration: int,
    *,
    per_step: Mapping[str, int] | None = None,
    detail: bool = False,
) -> None:
    """Log every section the profiler recorded, in the order the sections started.

    Sections named in ``per_step`` additionally report their duration divided by that count.
    Sections whose name ends in an index, such as ``denoising_step_7``, are one repetition of an
    enclosing section that already reports the total, so they are omitted unless ``detail`` is set.
    """
    names = [
        name
        for recorded_iteration, name in profiler.start_times
        if recorded_iteration == iteration
        and profiler.contains_step(name, iteration)
        and (detail or not _INDEXED_SECTION.search(name))
    ]
    width = max((len(name) for name in names), default=0)

    for name in names:
        duration = profiler.get_duration(name, iteration)
        steps = (per_step or {}).get(name)
        average = f"  ({duration / steps:6.2f}s per step)" if steps else ""
        logger.info(f"{name:<{width}}  {duration:7.2f}s{average}")


def log_event_section(event: PipelineEvent) -> None:
    if isinstance(event, SectionStart):
        logger.info(f"[>>] {event.name}")
    elif isinstance(event, SectionEnd):
        logger.info(f"[<<] {event.name}")
    elif isinstance(event, DenoiseStep):
        logger.info(f"[~~] Denoise step {event.step}/{event.total} (sigma={event.sigma:.4f})")
