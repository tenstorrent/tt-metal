# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Scratch: record every ttnn op a pipeline call makes, per pipeline section, with its new
program-cache entries, new kernel binaries and wall time. Needs
TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}' so the op hooks fire, and a fresh
TT_METAL_CACHE so that new kernel binaries can be counted."""

import csv
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import ttnn
from models.tt_dit.pipelines.events import SectionEnd, SectionStart


def kernel_count(root: Path) -> int:
    return sum(1 for _ in root.glob("*/kernels/*/*"))


def _shapes(values: object) -> list[str]:
    out = []
    for v in values if isinstance(values, (list, tuple)) else [values]:
        if isinstance(v, ttnn.Tensor):
            out.append("x".join(str(d) for d in v.shape) + f":{str(v.layout).split('.')[-1]}")
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], ttnn.Tensor):
            out.extend(_shapes(v))
    return out


def _caller() -> str:
    """The innermost tt_dit frame outside this file."""
    for frame in reversed(traceback.extract_stack()):
        if "tt_dit/" in frame.filename and "scratch_op_hook" not in frame.filename and "/tests/" not in frame.filename:
            return f"{frame.filename.split('tt_dit/')[-1]}:{frame.lineno}"
    return "?"


class OpRecorder:
    """``devices`` are the meshes the pipeline runs ops on: its submeshes when it creates any, since
    each has its own command queue and program cache, otherwise the mesh itself."""

    def __init__(self, devices: Sequence[ttnn.MeshDevice], out_csv: Path, kernel_root: Path) -> None:
        self._devices = list(devices)
        self._out_csv = out_csv
        self._kernel_root = kernel_root
        self._stack: list[tuple[int, int, float]] = []
        self.section = ["other"]
        self.rows: list[dict] = []
        self.run = ""

    def on_event(self, event: object) -> None:
        if isinstance(event, SectionStart) and not event.name.startswith("denoising_step"):
            self.section.append(event.name)
        elif isinstance(event, SectionEnd) and not event.name.startswith("denoising_step"):
            self.section.pop()

    def _synchronize(self) -> None:
        for d in self._devices:
            ttnn.synchronize_device(d)

    def _entries(self) -> int:
        return sum(d.num_program_cache_entries() for d in self._devices)

    def _pre(self, operation: object, args: object, kwargs: object) -> None:
        self._synchronize()
        self._stack.append((self._entries(), kernel_count(self._kernel_root), time.perf_counter()))

    def _post(self, operation: Any, args: object, kwargs: object, output: object) -> None:
        self._synchronize()
        entries, kernels, start = self._stack.pop()
        self.rows.append(
            {
                "run": self.run,
                "section": self.section[-1],
                "depth": len(self._stack),
                "op": operation.python_fully_qualified_name,
                "entries": self._entries() - entries,
                "kernels": kernel_count(self._kernel_root) - kernels,
                "seconds": time.perf_counter() - start,
                "shapes": " ".join(_shapes(args) + _shapes(list(kwargs.values()))),
                "where": _caller(),
            }
        )

    @contextmanager
    def recording(self, run: str) -> Iterator[None]:
        """Record the ops of one pipeline call; on an exception, log the op it failed after and go on."""
        self.run = run
        status = "ok"
        try:
            with ttnn.register_pre_operation_hook(self._pre), ttnn.register_post_operation_hook(self._post):
                yield
        except Exception as e:  # noqa: BLE001 - scratch: record which op failed and go on
            last = self.rows[-1]["op"] if self.rows else "?"
            status = f"failed after {last}: {str(e).splitlines()[0][:300] if str(e) else type(e).__name__}"
            import traceback

            traceback.print_exc()
            self._stack.clear()
            self.section[:] = ["other"]
        print(f"done {run}: {status}; {len(self.rows)} op records so far", flush=True)
        with open(self._out_csv.with_suffix(".status.txt"), "a") as f:
            f.write(f"{run}: {status}\n")
        if self.rows:
            with open(self._out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(self.rows[0]))
                w.writeheader()
                w.writerows(self.rows)


def run_all(recorder: OpRecorder, runs: list[tuple[str, Callable[[], object]]]) -> None:
    for label, fn in runs:
        with recorder.recording(label):
            fn()
