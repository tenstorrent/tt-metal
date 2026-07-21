# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Compilation and trace-resource lifecycle for the configured LLM runtime.

The compiler deliberately does not allocate or bind KV cache tensors.  A
caller-provided context getter exposes the cache already owned by
``PagedKVCacheManager`` and compilation fails until that context is available.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

import ttnn


def _trim_host_allocator() -> None:
    """Return released trace-capture staging arenas to the OS when supported."""
    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return
    malloc_trim.argtypes = (ctypes.c_size_t,)
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


class GraphState(str, Enum):
    COMPILED = "compiled"
    CAPTURED = "captured"
    RELEASED = "released"


@dataclass(frozen=True)
class GraphKey:
    """Canonical identity for one concrete prefill or decode program."""

    mode: str
    batch_size: int
    page_table_width: int
    sampling_path: str
    sequence_length: int | None = None
    chunk_page_table_width: int | None = None

    def __post_init__(self):
        if self.mode not in ("prefill", "decode"):
            raise ValueError(f"Unsupported graph mode: {self.mode!r}")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.page_table_width <= 0:
            raise ValueError("page_table_width must be positive")
        if self.mode == "prefill" and self.sequence_length is None:
            raise ValueError("prefill graph keys require sequence_length")
        if self.mode == "decode" and self.sequence_length is not None:
            raise ValueError("decode graph keys must not contain sequence_length")


@dataclass(frozen=True)
class OutputSpec:
    shape: tuple[int, ...]
    dtype: Any
    layout: Any = None
    memory_config: Any = None

    @classmethod
    def from_value(cls, value: Any) -> "OutputSpec":
        value = _primary_output(value)
        if isinstance(value, torch.Tensor):
            return cls(shape=tuple(value.shape), dtype=value.dtype)
        if isinstance(value, ttnn.Tensor):
            allocated = value.is_allocated() if hasattr(value, "is_allocated") else False
            return cls(
                shape=tuple(value.shape),
                dtype=value.dtype,
                layout=value.layout,
                memory_config=value.spec.memory_config if allocated else None,
            )
        raise TypeError(f"Cannot derive an output specification from {type(value).__name__}")


@dataclass
class PersistentInputs:
    """Trace-owned persistent replay inputs.

    ``values`` intentionally remains model-private.  Public runtime APIs do not
    expose tensor names or layouts, and capture callbacks receive the object as
    an opaque value.
    """

    values: Any


@dataclass(frozen=True)
class InputRefreshPolicy:
    every_replay: tuple[str, ...] = ()
    full_on_batch_reset: bool = True
    full_on_graph_switch: bool = True
    full_without_device_feedback: bool = True
    refresh_page_table_on_change: bool = True


@dataclass(frozen=True)
class RefreshDecision:
    full: bool
    page_table: bool
    fields: tuple[str, ...]


@dataclass
class TraceArtifact:
    trace_id: int
    persistent_inputs: PersistentInputs
    outputs: Any
    refresh_policy: InputRefreshPolicy
    trace_released: bool = False
    deallocated_tensor_ids: set[int] = field(default_factory=set, repr=False)


@dataclass
class CompiledGraph:
    key: GraphKey
    output_spec: OutputSpec
    trace: TraceArtifact | None = None
    state: GraphState = GraphState.COMPILED
    trace_eligible: bool = True


@dataclass(frozen=True)
class TraceCapturePlan:
    """Concrete callbacks needed to materialize and capture one trace."""

    prepare_inputs: Callable[[], PersistentInputs | Any]
    capture: Callable[[PersistentInputs], Any]
    refresh_policy: InputRefreshPolicy = InputRefreshPolicy()


@dataclass
class _RollbackOrphan:
    values: Any
    deallocated_tensor_ids: set[int] = field(default_factory=set)


class LLMGraphCompiler:
    """Own compiled-program metadata and optional TT trace artifacts."""

    def __init__(self, mesh_device, trace_config, bound_cache_context: Callable[[], Any]):
        self.mesh_device = mesh_device
        self.trace_config = trace_config
        self._bound_cache_context = bound_cache_context
        self._graphs: dict[GraphKey, CompiledGraph] = {}
        self._compile_orphans: list[_RollbackOrphan] = []
        self._rollback_orphans: list[_RollbackOrphan] = []
        self._activated = False
        self._capture_in_progress = False
        self._released = False
        self._previous_replay_key: GraphKey | None = None

    @property
    def graphs(self) -> Mapping[GraphKey, CompiledGraph]:
        return self._graphs.copy()

    @property
    def trace_active(self) -> bool:
        return self._activated

    def get(self, key: GraphKey) -> CompiledGraph | None:
        return self._graphs.get(key)

    def require_bound_cache_context(self):
        context = self._bound_cache_context()
        if context is None:
            raise RuntimeError("Paged KV cache must be allocated and bound before compilation")
        return context

    def compile(
        self,
        key: GraphKey,
        invoke: Callable[[Any], Any],
        *,
        output_spec: Callable[[Any], OutputSpec] = OutputSpec.from_value,
        trace_eligible: bool = True,
    ) -> CompiledGraph:
        """Compile one concrete graph and release its transient output.

        ``invoke`` receives the manager's read-only bound-cache context.  The
        compiler synchronizes, records the output contract, releases only the
        returned TT tensors, and synchronizes again before any trace-persistent
        input may be allocated.
        """
        self._ensure_live()
        self._ensure_no_compile_orphans()
        existing = self._graphs.get(key)
        if existing is not None:
            if existing.trace_eligible != trace_eligible:
                raise ValueError(f"Graph {key!r} was compiled with different trace eligibility")
            return existing
        if self._activated or self._capture_in_progress:
            raise RuntimeError(f"Cannot compile unseen graph {key!r} after trace activation")

        cache_context = self.require_bound_cache_context()
        output = invoke(cache_context)
        try:
            ttnn.synchronize_device(self.mesh_device)
            spec = output_spec(output)
        except BaseException as primary:
            cleanup_failures = self._release_or_retain_compile_output(output)
            try:
                ttnn.synchronize_device(self.mesh_device)
            except BaseException as error:
                cleanup_failures.append(error)
            _attach_cleanup_failures(primary, cleanup_failures)
            raise

        cleanup_failures = self._release_or_retain_compile_output(output)
        post_sync_error = None
        try:
            ttnn.synchronize_device(self.mesh_device)
        except BaseException as error:
            post_sync_error = error
        if post_sync_error is not None:
            _attach_cleanup_failures(post_sync_error, cleanup_failures)
            raise post_sync_error
        if cleanup_failures:
            error = RuntimeError(f"Failed to deallocate {len(cleanup_failures)} compile output resource(s)")
            _attach_cleanup_failures(error, cleanup_failures)
            raise error from cleanup_failures[0]

        graph = CompiledGraph(key=key, output_spec=spec, trace_eligible=trace_eligible)
        self._graphs[key] = graph
        return graph

    def capture_all(self, plans: Mapping[GraphKey, TraceCapturePlan]) -> None:
        """Allocate every persistent input before capturing the first trace."""
        self._ensure_live()
        self._ensure_no_compile_orphans()
        if self._activated:
            return
        if self._capture_in_progress:
            raise RuntimeError("Trace capture is already in progress")

        selected = [
            graph for graph in self._graphs.values() if graph.trace_eligible and self._trace_enabled(graph.key.mode)
        ]
        missing = [graph.key for graph in selected if graph.key not in plans]
        if missing:
            raise KeyError(f"Missing trace capture plans for compiled graphs: {missing!r}")
        if not selected:
            return

        prepared: dict[GraphKey, tuple[PersistentInputs, TraceCapturePlan]] = {}
        captured_keys: set[GraphKey] = set()
        self._capture_in_progress = True
        try:
            for graph in selected:
                plan = plans[graph.key]
                values = plan.prepare_inputs()
                persistent = values if isinstance(values, PersistentInputs) else PersistentInputs(values)
                prepared[graph.key] = (persistent, plan)

            for graph in selected:
                persistent, plan = prepared[graph.key]
                trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                outputs = None
                try:
                    outputs = plan.capture(persistent)
                    ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                    ttnn.synchronize_device(self.mesh_device)
                except BaseException as error:
                    # Once begin succeeds, persistent inputs and any returned
                    # output may be referenced by a partially captured backend
                    # trace. Retain them until release_trace succeeds.
                    graph.trace = TraceArtifact(
                        trace_id=trace_id,
                        persistent_inputs=persistent,
                        outputs=outputs,
                        refresh_policy=plan.refresh_policy,
                    )
                    graph.state = GraphState.CAPTURED
                    captured_keys.add(graph.key)
                    _attach_cleanup_failures(error, self._release_graph_trace(graph))
                    raise
                graph.trace = TraceArtifact(
                    trace_id=trace_id,
                    persistent_inputs=persistent,
                    outputs=outputs,
                    refresh_policy=plan.refresh_policy,
                )
                graph.state = GraphState.CAPTURED
                captured_keys.add(graph.key)
            ttnn.synchronize_device(self.mesh_device)
            self._activated = True
            _trim_host_allocator()
        except BaseException as error:
            cleanup_failures = self._release_trace_resources()
            for key, (persistent, _) in prepared.items():
                if key not in captured_keys:
                    orphan = _RollbackOrphan(persistent.values)
                    failures = _best_effort_deallocate_owned(
                        orphan.values,
                        orphan.deallocated_tensor_ids,
                    )
                    cleanup_failures.extend(failures)
                    if failures:
                        self._rollback_orphans.append(orphan)
            # Live traces or incompletely reclaimed inputs seal allocations.
            self._activated = bool(self._rollback_orphans) or any(
                graph.trace is not None for graph in self._graphs.values()
            )
            _attach_cleanup_failures(error, cleanup_failures)
            raise
        finally:
            self._capture_in_progress = False

    def assert_executable(self, key: GraphKey) -> CompiledGraph:
        """Return a precompiled graph or reject an unseen post-activation key."""
        self._ensure_live()
        graph = self._graphs.get(key)
        if graph is None:
            suffix = " after trace activation" if self._activated else ""
            raise RuntimeError(f"Graph {key!r} was not compiled{suffix}")
        return graph

    def replay(
        self,
        key: GraphKey,
        refresh_inputs: Callable[[TraceArtifact, RefreshDecision], None],
        *,
        reset_batch: bool = False,
        device_feedback_enabled: bool = False,
        feedback_compatible: bool = False,
        page_table_changed: bool = False,
    ) -> Any:
        graph = self.assert_executable(key)
        if graph.trace is None:
            raise RuntimeError(f"Graph {key!r} was compiled without a trace")

        policy = graph.trace.refresh_policy
        # The first replay must refresh capture-time template values too.
        switched = self._previous_replay_key != key
        full = (
            (policy.full_on_batch_reset and reset_batch)
            or (policy.full_on_graph_switch and switched)
            or (policy.full_without_device_feedback and not (device_feedback_enabled and feedback_compatible))
        )
        decision = RefreshDecision(
            full=full,
            page_table=policy.refresh_page_table_on_change and page_table_changed,
            fields=policy.every_replay,
        )
        refresh_inputs(graph.trace, decision)
        ttnn.execute_trace(self.mesh_device, graph.trace.trace_id, cq_id=0, blocking=False)
        self._previous_replay_key = key
        return graph.trace.outputs

    def cleanup(self) -> None:
        if self._released:
            return
        failures = self._release_trace_resources()
        failures.extend(self._release_compile_orphans())
        failures.extend(self._release_rollback_orphans())
        if failures:
            error = RuntimeError(f"Failed to release {len(failures)} graph resource(s)")
            _attach_cleanup_failures(error, failures)
            raise error from failures[0]
        for graph in self._graphs.values():
            graph.state = GraphState.RELEASED
        self._released = True

    def _release_trace_resources(self) -> list[BaseException]:
        failures: list[BaseException] = []
        for graph in self._graphs.values():
            failures.extend(self._release_graph_trace(graph))
        return failures

    def _release_or_retain_compile_output(self, output: Any) -> list[BaseException]:
        orphan = _RollbackOrphan(output)
        failures = _best_effort_deallocate_owned(
            orphan.values,
            orphan.deallocated_tensor_ids,
        )
        if failures:
            self._compile_orphans.append(orphan)
        return failures

    def _release_compile_orphans(self) -> list[BaseException]:
        failures = []
        remaining = []
        for orphan in self._compile_orphans:
            orphan_failures = _best_effort_deallocate_owned(
                orphan.values,
                orphan.deallocated_tensor_ids,
            )
            failures.extend(orphan_failures)
            if orphan_failures:
                remaining.append(orphan)
        self._compile_orphans = remaining
        return failures

    def _release_graph_trace(self, graph: CompiledGraph) -> list[BaseException]:
        artifact = graph.trace
        if artifact is None:
            return []

        if not artifact.trace_released:
            try:
                ttnn.release_trace(self.mesh_device, artifact.trace_id)
            except BaseException as error:
                # A live backend trace may still reference all of these
                # buffers. Keep the complete artifact for a later retry.
                return [error]
            artifact.trace_released = True

        failures = _best_effort_deallocate_owned(
            (artifact.persistent_inputs.values, artifact.outputs),
            artifact.deallocated_tensor_ids,
        )
        if failures:
            return failures

        graph.trace = None
        graph.state = GraphState.COMPILED
        return []

    def _release_rollback_orphans(self) -> list[BaseException]:
        failures = []
        remaining = []
        for orphan in self._rollback_orphans:
            orphan_failures = _best_effort_deallocate_owned(
                orphan.values,
                orphan.deallocated_tensor_ids,
            )
            failures.extend(orphan_failures)
            if orphan_failures:
                remaining.append(orphan)
        self._rollback_orphans = remaining
        return failures

    def _trace_enabled(self, mode: str) -> bool:
        if hasattr(self.trace_config, "enables"):
            return bool(self.trace_config.enables(mode))
        configured = getattr(self.trace_config, "mode", self.trace_config)
        configured = getattr(configured, "value", configured)
        return configured == "all" or configured == "decode_only" and mode == "decode"

    def _ensure_live(self) -> None:
        if self._released:
            raise RuntimeError("LLMGraphCompiler has been released")

    def _ensure_no_compile_orphans(self) -> None:
        if self._compile_orphans:
            raise RuntimeError(
                "Cannot compile or capture while unreleased compile outputs remain; clean up this compiler"
            )


def _primary_output(value: Any) -> Any:
    if isinstance(value, tuple):
        for item in value:
            if item is not None:
                return item
    return value


def _deallocate_owned(value: Any, seen: set[int] | None = None) -> None:
    """Deallocate explicit graph-owned TT values exactly once."""
    if value is None:
        return
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    if isinstance(value, ttnn.Tensor):
        ttnn.deallocate(value)
        return
    if isinstance(value, dict):
        for nested in value.values():
            _deallocate_owned(nested, seen)
        return
    if isinstance(value, (list, tuple, set)):
        for nested in value:
            _deallocate_owned(nested, seen)


def _best_effort_deallocate_owned(value: Any, completed: set[int] | None = None) -> list[BaseException]:
    """Deallocate every reachable owned TT tensor without skipping siblings.

    Successfully deallocated tensor identities are retained across cleanup
    retries; failed tensors remain retryable. Containers are only traversal
    structure and are cycle-guarded per attempt.
    """

    if completed is None:
        completed = set()
    failures: list[BaseException] = []
    visiting: set[int] = set()

    def visit(item: Any) -> None:
        if item is None:
            return
        item_id = id(item)
        if isinstance(item, ttnn.Tensor):
            if item_id in completed:
                return
            try:
                ttnn.deallocate(item)
            except BaseException as error:
                failures.append(error)
            else:
                completed.add(item_id)
            return
        if item_id in visiting:
            return
        if isinstance(item, dict):
            visiting.add(item_id)
            for nested in item.values():
                visit(nested)
            visiting.remove(item_id)
        elif isinstance(item, (list, tuple, set)):
            visiting.add(item_id)
            for nested in item:
                visit(nested)
            visiting.remove(item_id)

    visit(value)
    return failures


def _attach_cleanup_failures(error: BaseException, failures: list[BaseException]) -> None:
    if not failures:
        return
    previous = tuple(getattr(error, "cleanup_failures", ()))
    error.cleanup_failures = previous + tuple(failures)
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(f"Trace rollback also encountered {len(failures)} cleanup failure(s)")
