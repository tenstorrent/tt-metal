# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Trace capture, replay, and persistent-resource ownership."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

import ttnn
from models.common.llm_runtime.program_compiler import (
    ProgramCompiler,
    ProgramKey,
    signature_digest,
    validate_sha256_digest,
)
from models.common.llm_runtime.tensor_resources import (
    TensorResourceOrphan,
    attach_cleanup_failures,
    best_effort_deallocate_owned_tensors,
    raise_cleanup_failures,
    release_orphans,
)

_TRACE_KEY_DOMAIN = "tttv2.llm-runtime.trace"
_TRACE_KEY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TraceKey:
    """Full content digest for one operation-produced trace signature."""

    digest: str

    def __post_init__(self) -> None:
        validate_sha256_digest(self.digest, "trace")

    @classmethod
    def from_signature(cls, signature: Any) -> "TraceKey":
        return cls(signature_digest(_TRACE_KEY_DOMAIN, _TRACE_KEY_SCHEMA_VERSION, signature))


@dataclass
class PersistentInputs:
    """Trace-owned persistent replay inputs opaque to public runtime APIs."""

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


@dataclass(frozen=True)
class TraceCapturePlan:
    """Operation-produced specification for one trace-capable compiled program."""

    program_key: ProgramKey
    trace_signature: Any
    operation: str
    prepare_inputs: Callable[[], PersistentInputs | Any]
    capture: Callable[[PersistentInputs], Any]
    refresh_policy: InputRefreshPolicy = InputRefreshPolicy()
    schema_fingerprint: Any = None
    prepare_workspace: Callable[[], Any] | None = None
    workspace_fingerprint: Any = None
    prime: Callable[[PersistentInputs], Any] | None = None
    release_prime_output: Callable[[Any], list[BaseException]] | None = None

    def __post_init__(self) -> None:
        if self.operation not in ("prefill", "decode"):
            raise ValueError(f"Unsupported trace operation: {self.operation!r}")
        if (self.prime is None) is not (self.release_prime_output is None):
            raise ValueError("trace capture prime and output releaser must be configured together")


@dataclass
class TraceRecord:
    signature: Any
    operation: str
    artifact: TraceArtifact | None = None


@dataclass
class TraceAliasRecord:
    """Program-local postprocess state kept outside the shared hidden trace."""

    trace_key: TraceKey
    workspace_fingerprint: Any
    prepare_workspace: Callable[[], Any] | None = field(default=None, repr=False)
    workspace: Any = None
    deallocated_tensor_ids: set[int] = field(default_factory=set, repr=False)


class TraceCompiler:
    """Register, capture, replay, and release traces for compiled programs.

    ``TracedExecutor.compile_*`` first compiles an eager program and calls
    `register_capture_plan`. `WarmupCoordinator` calls
    `capture_all` only after the complete configured program set exists.
    Forward execution then calls `replay` with operation-owned refresh
    logic. The compiler owns trace artifacts and persistent inputs, while the
    composed ``ProgramCompiler`` remains the sole program registry.
    """

    def __init__(self, program_compiler: ProgramCompiler):
        if not isinstance(program_compiler, ProgramCompiler):
            raise TypeError("program_compiler must be a ProgramCompiler")
        self.program_compiler = program_compiler
        self.mesh_device = program_compiler.mesh_device
        self._traces: dict[TraceKey, TraceRecord] = {}
        self._plans: dict[TraceKey, TraceCapturePlan] = {}
        self._program_to_trace: dict[ProgramKey, TraceKey] = {}
        self._aliases: dict[ProgramKey, TraceAliasRecord] = {}
        self._rollback_orphans: list[TensorResourceOrphan] = []
        self._capture_in_progress = False
        self._activated = False
        self._released = False
        self._previous_replay_key: TraceKey | None = None
        self._replay_count = 0
        self._replay_counts = {"prefill": 0, "decode": 0}

    # Public API

    @property
    def trace_active(self) -> bool:
        return self._activated

    @property
    def replay_count(self) -> int:
        """Return successfully submitted trace replays across all operations."""

        return self._replay_count

    @property
    def replay_counts(self) -> dict[str, int]:
        """Return a snapshot of successfully submitted replays by operation."""

        return dict(self._replay_counts)

    @property
    def trace_count(self) -> int:
        """Return the number of semantic hidden traces in the registry."""

        return len(self._traces)

    @property
    def trace_association_count(self) -> int:
        """Return the number of compiled-program aliases associated to traces."""

        return len(self._program_to_trace)

    def registered_coverage(self, operation: str) -> tuple[tuple[TraceKey, Any], ...]:
        """Return registered trace keys/signatures for one operation."""

        if operation not in ("prefill", "decode"):
            raise ValueError(f"Unsupported trace operation: {operation!r}")
        return tuple(
            (trace_key, record.signature) for trace_key, record in self._traces.items() if record.operation == operation
        )

    def get(self, key: TraceKey) -> TraceRecord | None:
        """Return the record needed to finish an operation-specific replay."""

        return self._traces.get(key)

    def trace_key_for_program(self, program_key: ProgramKey) -> TraceKey | None:
        """Return the registered trace association for one compiled program."""

        return self._program_to_trace.get(program_key)

    def workspace_for_program(self, program_key: ProgramKey) -> Any:
        """Return one alias's postprocess workspace after capture allocation."""

        alias = self._aliases.get(program_key)
        if alias is None:
            raise RuntimeError(f"Program key {program_key.digest} has no trace alias workspace")
        return alias.workspace

    def register_capture_plan(self, plan: TraceCapturePlan) -> TraceKey:
        """Validate a compiled source and register one explicit trace association."""

        self._ensure_live()
        if self._capture_in_progress or self._activated:
            raise RuntimeError("Cannot register trace capture plans during capture or after trace activation")
        self.program_compiler.require_compiled(plan.program_key)

        trace_key = TraceKey.from_signature(plan.trace_signature)
        existing_association = self._program_to_trace.get(plan.program_key)
        if existing_association is not None and existing_association != trace_key:
            raise ValueError(f"Program key {plan.program_key.digest} already has a different trace association")
        existing_alias = self._aliases.get(plan.program_key)
        if existing_alias is not None and existing_alias.workspace_fingerprint != plan.workspace_fingerprint:
            raise ValueError(
                f"Program key {plan.program_key.digest} was registered with a different workspace fingerprint"
            )

        record = self._traces.get(trace_key)
        if record is None:
            record = TraceRecord(
                signature=plan.trace_signature,
                operation=plan.operation,
            )
            self._traces[trace_key] = record
            self._plans[trace_key] = plan
        else:
            if record.signature != plan.trace_signature:
                raise RuntimeError(f"Trace key collision for digest {trace_key.digest}: retained signature differs")
            if record.operation != plan.operation:
                raise RuntimeError(f"Trace key collision for digest {trace_key.digest}: operation differs")
            if self._plans[trace_key].refresh_policy != plan.refresh_policy:
                raise ValueError(f"Trace key {trace_key.digest} was registered with a different refresh policy")
            if self._plans[trace_key].schema_fingerprint != plan.schema_fingerprint:
                raise ValueError(f"Trace key {trace_key.digest} was registered with a different schema fingerprint")
        self._program_to_trace[plan.program_key] = trace_key
        if existing_alias is None:
            self._aliases[plan.program_key] = TraceAliasRecord(
                trace_key=trace_key,
                workspace_fingerprint=plan.workspace_fingerprint,
                prepare_workspace=plan.prepare_workspace,
            )
        return trace_key

    def capture_all(self) -> None:
        """Allocate every persistent input before beginning the first capture."""

        self._ensure_live()
        if self._activated:
            return
        if self._capture_in_progress:
            raise RuntimeError("Trace capture is already in progress")
        if not self._plans:
            return
        if self.program_compiler.compile_orphan_count:
            raise RuntimeError("Cannot capture while unreleased compile outputs remain")

        prepared: dict[TraceKey, tuple[PersistentInputs, TraceCapturePlan]] = {}
        captured_keys: set[TraceKey] = set()
        self._capture_in_progress = True
        try:
            for trace_key, plan in self._plans.items():
                self.program_compiler.require_compiled(plan.program_key)
                values = plan.prepare_inputs()
                persistent = values if isinstance(values, PersistentInputs) else PersistentInputs(values)
                prepared[trace_key] = (persistent, plan)

            for program_key, alias in self._aliases.items():
                if alias.prepare_workspace is not None:
                    alias.workspace = alias.prepare_workspace()

            capture_order = sorted(
                prepared,
                key=lambda trace_key: self._traces[trace_key].operation == "prefill",
            )
            for trace_key in capture_order:
                persistent, plan = prepared[trace_key]
                record = self._traces[trace_key]
                # Program signatures intentionally describe padded trace
                # identity, not active-row cardinality. Selected operation
                # plans therefore prime their exact persistent-input body
                # immediately before capturing that same body. No unrelated
                # trace can perturb allocator/program state between the prime
                # and ``begin_trace_capture``.
                if plan.prime is not None:
                    prime_output = None
                    try:
                        prime_output = plan.prime(persistent)
                        ttnn.synchronize_device(self.mesh_device)
                    except BaseException as primary:
                        cleanup_failures = plan.release_prime_output(prime_output)
                        try:
                            ttnn.synchronize_device(self.mesh_device)
                        except BaseException as error:
                            cleanup_failures.append(error)
                        attach_cleanup_failures(primary, cleanup_failures)
                        raise
                    release_failures = plan.release_prime_output(prime_output)
                    try:
                        ttnn.synchronize_device(self.mesh_device)
                    except BaseException as error:
                        release_failures.append(error)
                    if release_failures:
                        raise_cleanup_failures(release_failures)
                    logger.info(f"Primed {plan.operation} trace capture body: signature={plan.trace_signature!r}")

                self.program_compiler.set_trace_capture_in_progress(True)
                trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
                outputs = None
                try:
                    outputs = plan.capture(persistent)
                    ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
                    ttnn.synchronize_device(self.mesh_device)
                except BaseException as primary:
                    record.artifact = TraceArtifact(
                        trace_id=trace_id,
                        persistent_inputs=persistent,
                        outputs=outputs,
                        refresh_policy=plan.refresh_policy,
                    )
                    captured_keys.add(trace_key)
                    attach_cleanup_failures(primary, self._release_trace(record))
                    raise
                record.artifact = TraceArtifact(
                    trace_id=trace_id,
                    persistent_inputs=persistent,
                    outputs=outputs,
                    refresh_policy=plan.refresh_policy,
                )
                logger.info(f"Captured {plan.operation} trace: signature={plan.trace_signature!r}")
                captured_keys.add(trace_key)
                self.program_compiler.set_trace_capture_in_progress(False)

            ttnn.synchronize_device(self.mesh_device)
            self._capture_in_progress = False
            self.program_compiler.set_trace_capture_in_progress(False)
            self._activated = True
            self.program_compiler.set_trace_active(True)
            _trim_host_allocator()
        except BaseException as primary:
            cleanup_failures = self._release_trace_resources()
            cleanup_failures.extend(self._release_alias_workspaces())
            for trace_key, (persistent, _) in prepared.items():
                if trace_key in captured_keys:
                    continue
                orphan = TensorResourceOrphan(persistent.values)
                orphan_failures = best_effort_deallocate_owned_tensors(
                    orphan.values,
                    orphan.deallocated_tensor_ids,
                )
                cleanup_failures.extend(orphan_failures)
                if orphan_failures:
                    self._rollback_orphans.append(orphan)

            self._activated = (
                bool(self._rollback_orphans)
                or any(record.artifact is not None for record in self._traces.values())
                or any(alias.workspace is not None for alias in self._aliases.values())
            )
            self._capture_in_progress = False
            self.program_compiler.set_trace_capture_in_progress(False)
            self.program_compiler.set_trace_active(self._activated)
            attach_cleanup_failures(primary, cleanup_failures)
            raise

    def replay(
        self,
        program_key: ProgramKey,
        refresh_inputs: Callable[[TraceArtifact, RefreshDecision], None],
        *,
        reset_batch: bool = False,
        device_feedback_enabled: bool = False,
        feedback_compatible: bool = False,
        page_table_changed: bool = False,
    ) -> Any:
        """Refresh persistent inputs and enqueue one non-blocking trace replay."""

        self._ensure_live()
        self.program_compiler.require_compiled(program_key)
        trace_key = self._program_to_trace.get(program_key)
        if trace_key is None:
            raise RuntimeError(f"Program key {program_key.digest} has no trace association")
        record = self._traces[trace_key]
        artifact = record.artifact
        if artifact is None:
            raise RuntimeError(f"Trace key {trace_key.digest} has not been captured")

        policy = artifact.refresh_policy
        switched = self._previous_replay_key != trace_key
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
        refresh_inputs(artifact, decision)
        ttnn.execute_trace(self.mesh_device, artifact.trace_id, cq_id=0, blocking=False)
        self._replay_count += 1
        self._replay_counts[record.operation] += 1
        self._previous_replay_key = trace_key
        return artifact.outputs

    def cleanup(self) -> None:
        """Release traces and persistent inputs, then reopen the program gate."""

        if self._released:
            return
        failures = self._release_trace_resources()
        failures.extend(self._release_alias_workspaces())
        failures.extend(release_orphans(self._rollback_orphans))
        if failures:
            self._activated = True
            self.program_compiler.set_trace_active(True)
            error = RuntimeError(f"Failed to release {len(failures)} trace resource(s)")
            attach_cleanup_failures(error, failures)
            raise error from failures[0]
        self._capture_in_progress = False
        self._activated = False
        self._previous_replay_key = None
        self.program_compiler.set_trace_capture_in_progress(False)
        self.program_compiler.set_trace_active(False)
        self._released = True

    # Private implementation

    def _release_trace_resources(self) -> list[BaseException]:
        failures: list[BaseException] = []
        for record in self._traces.values():
            failures.extend(self._release_trace(record))
        return failures

    def _release_alias_workspaces(self) -> list[BaseException]:
        failures: list[BaseException] = []
        for alias in self._aliases.values():
            if alias.workspace is None:
                continue
            alias_failures = best_effort_deallocate_owned_tensors(
                alias.workspace,
                alias.deallocated_tensor_ids,
            )
            failures.extend(alias_failures)
            if not alias_failures:
                alias.workspace = None
        return failures

    def _release_trace(self, record: TraceRecord) -> list[BaseException]:
        artifact = record.artifact
        if artifact is None:
            return []
        if not artifact.trace_released:
            try:
                ttnn.release_trace(self.mesh_device, artifact.trace_id)
            except BaseException as error:
                return [error]
            artifact.trace_released = True

        failures = best_effort_deallocate_owned_tensors(
            (artifact.persistent_inputs.values, artifact.outputs),
            artifact.deallocated_tensor_ids,
        )
        if failures:
            return failures
        record.artifact = None
        return []

    def _ensure_live(self) -> None:
        if self._released:
            raise RuntimeError("TraceCompiler has been released")


def _trim_host_allocator() -> None:
    """Return released trace-capture staging arenas to the OS when supported."""

    try:
        malloc_trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError):
        return
    malloc_trim.argtypes = (ctypes.c_size_t,)
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)
