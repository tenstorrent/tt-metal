#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Script execution drivers. `LocalScriptRunner` for single-process,
`MPIScriptRunner` for multi-rank under tt-run/mpirun (Shape B: async producer/consumer)."""

from __future__ import annotations

import os
import signal
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields, is_dataclass
from time import time
from typing import Any

import utils
from aggregator import MergedResult, Sentinel, merged_to_renderable
from rich.progress import Progress
from triage_script import ScriptArguments, TriageScript, TTTriageError, default_serializer
from ttexalens.context import Context


DEFAULT_SCRIPT_TIMEOUT_SECONDS = int(os.environ.get("TT_TRIAGE_MPI_SCRIPT_TIMEOUT", "600"))


@dataclass(frozen=True)
class RankFailure(Sentinel):
    rank: int
    script_name: str
    failure_message: str


@dataclass(frozen=True)
class RankTimeout(Sentinel):
    rank: int
    script_name: str
    duration_seconds: float


@dataclass(frozen=True)
class RankSkipped(Sentinel):
    rank: int
    script_name: str
    reason: str


def create_runner(
    args: ScriptArguments,
    context: Context,
    process_group: Any,
    *,
    target_paths: set[str] | None = None,
    progress: Any = None,
    progress_task: Any = None,
) -> "ScriptRunner":
    """Pick the runner that fits the process group. `progress` / `progress_task`
    are only consumed by `LocalScriptRunner`."""
    if process_group.is_multi:
        return MPIScriptRunner(args, context, process_group, target_paths=target_paths)
    return LocalScriptRunner(args, context, target_paths=target_paths, progress=progress, progress_task=progress_task)


class ScriptRunner(ABC):
    def __init__(self, args: ScriptArguments, context: Context):
        self.args = args
        self.context = context

    def run_all(self, script_queue: list[TriageScript]) -> None:
        for script in script_queue:
            self.process_script(script)

    @abstractmethod
    def process_script(self, script: TriageScript) -> None:
        ...

    def render_last_result(self) -> None:
        """Render whatever single-target mode stashed. Default no-op."""

    def finalize(self) -> None:
        """End-of-run hook (e.g., MPI barrier). Default no-op."""

    def print_totals(self, init_seconds: float) -> None:
        """Emit aggregate timing summary at end of run. Default no-op."""


class LocalScriptRunner(ScriptRunner):
    """Single-process runner.

    `target_paths=None`     -> full-triage: render every non-data-provider script inline.
    `target_paths={paths}`  -> single-target: run silently, stash target result for `render_last_result()`.
    """

    def __init__(
        self,
        args: ScriptArguments,
        context: Context,
        *,
        progress: Progress | None = None,
        progress_task: Any = None,
        target_paths: set[str] | None = None,
    ):
        super().__init__(args, context)
        self._progress = progress
        self._progress_task = progress_task
        self._target_paths = target_paths
        self._print_times = bool(args["--print-script-times"])
        self.script_seconds = 0.0
        self.serialization_seconds = 0.0
        self.last_result: Any = None
        self.last_target_script: TriageScript | None = None

    def process_script(self, script: TriageScript) -> None:
        if self._progress is not None and self._progress_task is not None:
            self._progress.update(self._progress_task, description=f"Running {script.name}")
        try:
            if self._target_paths is not None:
                self._process_target(script)
            else:
                self._process_full(script)
        finally:
            if self._progress is not None and self._progress_task is not None:
                self._progress.advance(self._progress_task)

    def _process_target(self, script: TriageScript) -> None:
        # Target mode mirrors run_script's old loop: raise loudly on any failure.
        if any(dep.failed for dep in script.depends):
            raise TTTriageError(f"{script.name}: Cannot run script due to failed dependencies.")
        start = time()
        result = script.run(args=self.args, context=self.context, log_error=False)
        self.script_seconds += time() - start
        if script.config.data_provider and result is None:
            raise TTTriageError(f"{script.name}: Data provider script did not return any data.")
        if script.path in self._target_paths and not script.config.data_provider:
            self.last_target_script = script
            self.last_result = result

    def _process_full(self, script: TriageScript) -> None:
        if any(dep.failed for dep in script.depends):
            # Silent skip — the root-cause failure already printed its own message.
            script.failed = True
            script.failure_message = "Cannot run script due to failed dependencies."
            return
        start = time()
        result = script.run(args=self.args, context=self.context)
        elapsed = time() - start
        self.script_seconds += elapsed
        exec_str = f" [{elapsed:.2f}s]" if self._print_times else ""

        if script.config.data_provider:
            if result is None:
                print()
                utils.INFO(f"{script.name}{exec_str}:")
                if script.failure_message is not None:
                    utils.ERROR(f"  Data provider script failed: {script.failure_message}")
                else:
                    utils.ERROR(f"  Data provider script did not return any data.")
            elif exec_str:
                print()
                utils.INFO(f"{script.name}{exec_str}:")
                utils.INFO("  pass")
            return

        self.last_result = result
        ser_start = time()
        from triage import serialize_result  # lazy: triage imports this module

        serialize_result(script, result, exec_str)
        self.serialization_seconds += time() - ser_start

    def render_last_result(self) -> None:
        if self.last_target_script is None:
            return
        from triage import serialize_result

        serialize_result(self.last_target_script, self.last_result)

    def print_totals(self, init_seconds: float) -> None:
        if not self._print_times:
            return
        total = init_seconds + self.script_seconds + self.serialization_seconds
        print()
        utils.INFO(f"Total serialization time: {self.serialization_seconds:.2f}s")
        utils.INFO(f"Total execution time: {total:.2f}s")


class MPIScriptRunner(ScriptRunner):
    """Every rank runs its queue independently.
    Non-root `isend`s a payload tagged by script index; root `recv`s per-script
    payloads from all ranks, merges, and renders + emits a status line.
    `finalize()` Waitalls outstanding sends (non-root) and Barriers."""

    def __init__(
        self,
        args: ScriptArguments,
        context: Context,
        process_group: Any,
        *,
        timeout_seconds: int = DEFAULT_SCRIPT_TIMEOUT_SECONDS,
        target_paths: set[str] | None = None,
    ):
        super().__init__(args, context)
        self._pg = process_group
        self._comm = process_group.comm
        self._timeout = timeout_seconds
        self._target_paths = target_paths
        self._requests: list[Any] = []  # non-root: outstanding isends
        self._idx = 0
        self._total = 0
        # Root-only stash for single-target mode.
        self.last_target_script: TriageScript | None = None
        self.last_result: Any = None

    def run_all(self, script_queue: list[TriageScript]) -> None:
        self._total = len(script_queue)
        for i, script in enumerate(script_queue):
            self._idx = i
            self.process_script(script)

    def process_script(self, script: TriageScript) -> None:
        # Every rank runs its own copy. Skip decisions below are config-driven
        # (data_provider, target filter), so all ranks take the same path — no
        # stalled root recv waiting on payloads that ranks decided not to send.
        result, elapsed, early_sentinel = self._run_with_timeout(script)

        if script.config.data_provider:
            # Per-rank log only; failures get mpirun --tag-output attribution.
            # Suppress cascade-skip noise (root cause already logged its own failure).
            if not isinstance(early_sentinel, RankSkipped):
                self._report_local_data_provider(script, result, elapsed)
            return
        if self._target_paths is not None and script.path not in self._target_paths:
            return

        payload = self._build_payload(script, result, early_sentinel)

        if self._pg.is_root:
            parts = self._collect_parts(payload)
            self._render_or_stash(script, parts, elapsed)
        else:
            req = self._comm.isend(payload, dest=0, tag=self._idx)
            self._requests.append(req)

    def render_last_result(self) -> None:
        if not self._pg.is_root or self.last_target_script is None:
            return
        self._render_inline(self.last_target_script, self.last_result)

    def finalize(self) -> None:
        if not self._pg.is_root:
            # Block until every isend has been picked up by root before barrier/exit.
            from mpi4py import MPI

            if self._requests:
                MPI.Request.Waitall(self._requests)
                self._requests.clear()
        self._pg.barrier()

    def _run_with_timeout(self, script: TriageScript) -> tuple[Any, float, Sentinel | None]:
        """Run with SIGALRM timeout. Returns `(result, elapsed, early_sentinel)`;
        `early_sentinel` is `RankSkipped` on cascade, `RankTimeout` on timeout,
        or `None` (caller inspects `script.failed` for script-internal failures)."""
        if any(dep.failed for dep in script.depends):
            script.failed = True
            script.failure_message = "Cannot run script due to failed dependencies."
            return None, 0.0, RankSkipped(rank=self._pg.rank, script_name=script.name, reason=script.failure_message)
        if self._timeout <= 0:
            start = time()
            result = script.run(args=self.args, context=self.context)
            return result, time() - start, None

        class _ScriptTimeout(Exception):
            pass

        def _handler(signum, frame):
            raise _ScriptTimeout()

        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(self._timeout)
        start = time()
        try:
            result = script.run(args=self.args, context=self.context)
            return result, time() - start, None
        except _ScriptTimeout:
            elapsed = time() - start
            script.failed = True
            script.failure_message = f"Exceeded {self._timeout}s timeout"
            return None, elapsed, RankTimeout(rank=self._pg.rank, script_name=script.name, duration_seconds=elapsed)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _report_local_data_provider(
        self,
        script: TriageScript,
        result: Any,
        elapsed: float,
    ) -> None:
        """Per-rank logging for data_providers; mpirun --tag-output attributes the lines.
        Caller is responsible for skipping cascade-skip cases."""
        print_times = bool(self.args["--print-script-times"])
        exec_str = f" [{elapsed:.2f}s]" if print_times else ""
        if result is None:
            print()
            utils.INFO(f"{script.name}{exec_str}:")
            if script.failure_message is not None:
                utils.ERROR(f"  Data provider script failed: {script.failure_message}")
            else:
                utils.ERROR(f"  Data provider script did not return any data.")
        elif exec_str:
            print()
            utils.INFO(f"{script.name}{exec_str}:")
            utils.INFO("  pass")

    def _serialize_dataclass(self, obj: Any) -> Any:
        """Replace each dataclass field with its registered serializer's output so the
        result pickles cleanly across ranks. Non-dataclass values pass through."""
        if isinstance(obj, list):
            return [self._serialize_dataclass(x) for x in obj]
        if not is_dataclass(obj) or isinstance(obj, type):
            return obj
        cls = type(obj)
        kwargs = {}
        for f in fields(cls):
            v = getattr(obj, f.name)
            if f.metadata.get("recurse"):
                kwargs[f.name] = self._serialize_dataclass(v)
            else:
                kwargs[f.name] = f.metadata.get("serializer", default_serializer)(v)
        return cls(**kwargs)

    def _build_payload(
        self,
        script: TriageScript,
        result: Any,
        early_sentinel: Sentinel | None,
    ) -> Any:
        if early_sentinel is not None:
            return early_sentinel
        if script.failed:
            return RankFailure(
                rank=self._pg.rank,
                script_name=script.name,
                failure_message=script.failure_message or "unknown error",
            )
        try:
            return self._serialize_dataclass(script.to_raw_data(result))
        except Exception:
            return RankFailure(
                rank=self._pg.rank,
                script_name=script.name,
                failure_message=f"to_raw_data raised:\n{traceback.format_exc()}",
            )

    def _collect_parts(self, own_payload: Any) -> list[Any]:
        """Root: assemble per-rank parts for the current script index."""
        from mpi4py import MPI

        parts: list[Any] = [None] * self._pg.size
        parts[0] = own_payload
        for _ in range(self._pg.size - 1):
            status = MPI.Status()
            received = self._comm.recv(source=MPI.ANY_SOURCE, tag=self._idx, status=status)
            parts[status.Get_source()] = received
        return parts

    def _render_or_stash(self, script: TriageScript, parts: list[Any], own_elapsed: float) -> None:
        self._print_script_summary(script, parts, own_elapsed)
        if not any(p is not None and not isinstance(p, Sentinel) for p in parts):
            sentinels = [p for p in parts if isinstance(p, Sentinel)]
            merged = MergedResult(rows=[], sentinels=sentinels) if sentinels else None
        else:
            try:
                merged = script.merge(parts) if len(parts) > 1 else parts[0]
            except Exception:
                utils.ERROR(f"  {script.name}: merge raised on rank 0:\n{traceback.format_exc()}")
                return
        if self._target_paths is not None:
            self.last_target_script = script
            self.last_result = merged
            return
        self._render_inline(script, merged)

    def _render_inline(self, script: TriageScript, merged: Any) -> None:
        if isinstance(merged, MergedResult):
            for sentinel in merged.sentinels:
                if isinstance(sentinel, RankFailure):
                    first = sentinel.failure_message.splitlines()[0] if sentinel.failure_message else "unknown"
                    utils.ERROR(f"  rank {sentinel.rank}: {first}")
                elif isinstance(sentinel, RankTimeout):
                    utils.ERROR(f"  rank {sentinel.rank}: timeout after {sentinel.duration_seconds:.1f}s")
                elif isinstance(sentinel, RankSkipped):
                    utils.WARN(f"  rank {sentinel.rank}: skipped ({sentinel.reason})")
        render_input = merged_to_renderable(merged, tag_field_name="rank", tag_column_header="Rank")
        from triage import serialize_result

        serialize_result(script, render_input)

    def _print_script_summary(self, script: TriageScript, parts: list[Any], own_elapsed: float) -> None:
        n = len(parts)
        failed = sum(1 for p in parts if isinstance(p, RankFailure))
        timed_out = sum(1 for p in parts if isinstance(p, RankTimeout))
        skipped = sum(1 for p in parts if isinstance(p, RankSkipped))
        ok = n - failed - timed_out - skipped
        bits = [f"{ok}/{n} ok"]
        if failed:
            bits.append(f"{failed} failed")
        if timed_out:
            bits.append(f"{timed_out} timed out")
        if skipped:
            bits.append(f"{skipped} skipped")
        utils.INFO(f"[{self._idx + 1}/{self._total}] {script.name}: " + ", ".join(bits))
