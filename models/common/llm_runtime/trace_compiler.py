# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Trace capture, replay, and persistent-resource ownership."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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
    release_orphans,
    trim_host_allocator,
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

    @property
    def short(self) -> str:
        return self.digest[:12]


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
    trace_eligible: bool = True

    def __post_init__(self) -> None:
        if self.operation not in ("prefill", "decode"):
            raise ValueError(f"Unsupported trace operation: {self.operation!r}")


@dataclass
class TraceRecord:
    key: TraceKey
    signature: Any
    source_program_key: ProgramKey
    operation: str
    artifact: TraceArtifact | None = None


class TraceCompiler:
    """Own trace state while composing the executor's exact ProgramCompiler."""

    def __init__(self, program_compiler: ProgramCompiler, mesh_device: Any, trace_config: Any):
        if not isinstance(program_compiler, ProgramCompiler):
            raise TypeError("program_compiler must be a ProgramCompiler")
        self.program_compiler = program_compiler
        self.mesh_device = mesh_device
        self.trace_config = trace_config
        self._traces: dict[TraceKey, TraceRecord] = {}
        self._plans: dict[TraceKey, TraceCapturePlan] = {}
        self._program_to_trace: dict[ProgramKey, TraceKey] = {}
        self._rollback_orphans: list[TensorResourceOrphan] = []
        self._capture_in_progress = False
        self._activated = False
        self._released = False
        self._previous_replay_key: TraceKey | None = None

    @property
    def traces(self) -> Mapping[TraceKey, TraceRecord]:
        return self._traces.copy()

    @property
    def program_to_trace(self) -> Mapping[ProgramKey, TraceKey]:
        return self._program_to_trace.copy()

    @property
    def trace_active(self) -> bool:
        return self._activated

    @property
    def capture_in_progress(self) -> bool:
        return self._capture_in_progress

    @property
    def rollback_orphan_count(self) -> int:
        return len(self._rollback_orphans)

    def key_for(self, signature: Any) -> TraceKey:
        return TraceKey.from_signature(signature)

    def get(self, key: TraceKey) -> TraceRecord | None:
        return self._traces.get(key)

    def trace_key_for_program(self, program_key: ProgramKey) -> TraceKey | None:
        return self._program_to_trace.get(program_key)

    def register_capture_plan(self, plan: TraceCapturePlan) -> TraceKey | None:
        """Validate a compiled source and register one explicit trace association."""

        self._ensure_live()
        if self._capture_in_progress or self._activated:
            raise RuntimeError("Cannot register trace capture plans during capture or after trace activation")
        self.program_compiler.require_compiled(plan.program_key)
        if not plan.trace_eligible or not self._trace_enabled(plan.operation):
            return None

        trace_key = self.key_for(plan.trace_signature)
        existing_association = self._program_to_trace.get(plan.program_key)
        if existing_association is not None and existing_association != trace_key:
            raise ValueError(f"Program key {plan.program_key.digest} already has a different trace association")

        record = self._traces.get(trace_key)
        if record is None:
            record = TraceRecord(
                key=trace_key,
                signature=plan.trace_signature,
                source_program_key=plan.program_key,
                operation=plan.operation,
            )
            self._traces[trace_key] = record
            self._plans[trace_key] = plan
        else:
            self._ensure_matching_signature(trace_key, record.signature, plan.trace_signature)
            if record.operation != plan.operation:
                raise RuntimeError(f"Trace key collision for digest {trace_key.digest}: operation differs")
            if self._plans[trace_key].refresh_policy != plan.refresh_policy:
                raise ValueError(f"Trace key {trace_key.digest} was registered with a different refresh policy")
        self._program_to_trace[plan.program_key] = trace_key
        return trace_key

    def capture_all(self, plans: tuple[TraceCapturePlan, ...] | list[TraceCapturePlan] | None = None) -> None:
        """Allocate every persistent input before beginning the first capture."""

        self._ensure_live()
        if plans is not None:
            for plan in plans:
                self.register_capture_plan(plan)
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
        self.program_compiler.set_trace_capture_in_progress(True)
        self._capture_in_progress = True
        try:
            for trace_key, plan in self._plans.items():
                self.program_compiler.require_compiled(plan.program_key)
                values = plan.prepare_inputs()
                persistent = values if isinstance(values, PersistentInputs) else PersistentInputs(values)
                prepared[trace_key] = (persistent, plan)

            capture_order = sorted(
                prepared,
                key=lambda trace_key: self._traces[trace_key].operation == "prefill",
            )
            for trace_key in capture_order:
                persistent, plan = prepared[trace_key]
                record = self._traces[trace_key]
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
                logger.info(f"Captured {plan.operation} trace")
                captured_keys.add(trace_key)

            ttnn.synchronize_device(self.mesh_device)
            self._capture_in_progress = False
            self.program_compiler.set_trace_capture_in_progress(False)
            self._activated = True
            self.program_compiler.set_trace_active(True)
            trim_host_allocator()
        except BaseException as primary:
            cleanup_failures = self._release_trace_resources()
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

            self._activated = bool(self._rollback_orphans) or any(
                record.artifact is not None for record in self._traces.values()
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
        self._previous_replay_key = trace_key
        return artifact.outputs

    def cleanup(self) -> None:
        if self._released:
            return
        failures = self._release_trace_resources()
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

    def _release_trace_resources(self) -> list[BaseException]:
        failures: list[BaseException] = []
        for record in self._traces.values():
            failures.extend(self._release_trace(record))
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

    def _trace_enabled(self, operation: str) -> bool:
        if hasattr(self.trace_config, "enables"):
            return bool(self.trace_config.enables(operation))
        configured = getattr(self.trace_config, "mode", self.trace_config)
        configured = getattr(configured, "value", configured)
        return configured == "all" or configured == "decode_only" and operation == "decode"

    @staticmethod
    def _ensure_matching_signature(trace_key: TraceKey, retained: Any, candidate: Any) -> None:
        if retained != candidate:
            raise RuntimeError(f"Trace key collision for digest {trace_key.digest}: retained signature differs")

    def _ensure_live(self) -> None:
        if self._released:
            raise RuntimeError("TraceCompiler has been released")
