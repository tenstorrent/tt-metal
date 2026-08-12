# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Concrete eager and traced execution by direct composition."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from loguru import logger

from models.common.llm_runtime.decode import DecodeRuntime
from models.common.llm_runtime.decode import InvocationResult as DecodeInvocationResult
from models.common.llm_runtime.prefill.plan import PrefillRequest
from models.common.llm_runtime.prefill.runtime import PrefillRuntime
from models.common.llm_runtime.program_compiler import CompiledProgram, OutputSpec, ProgramCompiler
from models.common.llm_runtime.trace_compiler import InputRefreshPolicy, TraceCapturePlan, TraceCompiler


class TraceCoverageError(RuntimeError):
    """Actionable strict-trace miss with construction coverage context."""


@dataclass(frozen=True)
class PrefillReplayEvidence:
    """Structured evidence for one successfully submitted prefill trace."""

    operation: str
    variant: str
    sampling_path: str
    execution: str
    active_batch_size: int
    padded_batch_size: int
    padded_sequence_length: int
    lane: int
    rank: int
    program_key: str
    trace_key: str
    replay_steps: int


class EagerExecutor:
    """Compile and execute prepared requests through the eager TT path.

    ``Llama3Executor`` owns one instance and exposes it to
    ``Llama3Generator`` as the non-traced execution target. Callers normally
    use `compile_prefill`, `prefill_forward`,
    `compile_decode`, and `decode_forward`; request preparation and
    program-registry mechanics remain private to this composition.
    """

    def __init__(self, *, prefill: PrefillRuntime, decode: DecodeRuntime, program_compiler: ProgramCompiler) -> None:
        if not isinstance(prefill, PrefillRuntime):
            raise TypeError("prefill must be a PrefillRuntime")
        if not isinstance(decode, DecodeRuntime):
            raise TypeError("decode must be a DecodeRuntime")
        if not isinstance(program_compiler, ProgramCompiler):
            raise TypeError("program_compiler must be a ProgramCompiler")
        self.prefill = prefill
        self.decode = decode
        self.program_compiler = program_compiler
        self._eager_prefill_count = 0

    # Public API

    @property
    def eager_prefill_count(self) -> int:
        """Return successfully submitted eager prefill requests."""

        return self._eager_prefill_count

    def runtime_summary(self) -> dict[str, Any]:
        """Return serving-gate counters owned by the eager execution path."""

        return {
            "eager_prefill_executions": self._eager_prefill_count,
            "semantic_program_count": len(self.program_compiler.compiled_programs),
            "rejected_post_activation_compile_attempts": (self.program_compiler.post_activation_compile_rejections),
            "ttnn_program_cache_count": _program_cache_entries(self.program_compiler.mesh_device),
        }

    def compile_prefill(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        sampling_params: Any = None,  # ↓ Sampling
    ) -> tuple[CompiledProgram, ...]:
        """Prepare and compile every eager program needed by one prefill call."""

        programs = []
        for prepared in self._prepare_prefill(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
        ):
            programs.extend(self._compile_prefill(prepared))
        return tuple(programs)

    def prefill_forward(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        sampling_params: Any = None,  # ↓ Sampling
    ):
        """Prepare, execute, and assemble one eager prefill call."""

        prepared = self._prepare_prefill(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
        )
        results = tuple((request, self._execute_prefill(request)) for request in prepared)
        return self.prefill.assemble(
            results,
            batch_size=int(tokens.shape[0]),
            sampling_params=sampling_params,
        )

    def compile_decode(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
    ) -> CompiledProgram:
        """Prepare and compile the eager program needed by one decode call."""

        return self._compile_decode(
            self._prepare_decode(
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table,
                sampling_params=sampling_params,
                reset_batch=reset_batch,
            )
        )

    def decode_forward(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
        read_from_device: bool = True,  # ↓ Output policy
    ):
        """Prepare and execute one eager decode call."""

        prepared = self._prepare_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )
        return self._execute_decode(prepared, read_from_device=read_from_device)

    # Private implementation

    def _prepare_prefill(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        sampling_params: Any = None,  # ↓ Sampling
    ):
        return self.prefill.prepare(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
        )

    def _compile_prefill(self, prepared: Any):
        programs = []
        for signature in prepared.program_signatures:
            programs.append(
                self.program_compiler.compile(
                    signature,
                    lambda _context, prepared=prepared: self.prefill.invoke(prepared),
                    output_spec=lambda result: OutputSpec.from_value(result.value),
                    release_output=lambda result: result.owned,
                )
            )
        return tuple(programs)

    def _execute_prefill(self, prepared: Any):
        self._require_ready_after_trace_gate(prepared.program_signatures)
        result = self.prefill.invoke(prepared)
        self._eager_prefill_count += 1
        return result

    def _prepare_decode(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
    ):
        return self.decode.prepare(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )

    def _compile_decode(self, prepared: Any):
        return self.program_compiler.compile(
            self.decode.program_signature(prepared),
            lambda _context: self.decode.invoke(prepared, device_feedback=prepared.device_feedback),
            output_spec=lambda result: OutputSpec.from_value(result.value),
            release_output=lambda result: result.owned,
        )

    def _execute_decode(self, prepared: Any, *, read_from_device: bool = True):
        if self._program_gate_active():
            self._require_ready_after_trace_gate((self.decode.program_signature(prepared),))
        result = self.decode.invoke(prepared, device_feedback=False)
        return self.decode.consume(result, read_from_device=read_from_device)

    def _require_ready_after_trace_gate(self, signatures: Any) -> None:
        if not self._program_gate_active():
            return
        for signature in signatures:
            key = self.program_compiler.key_for(signature)
            self.program_compiler.require_compiled(key, signature)

    def _program_gate_active(self) -> bool:
        return self.program_compiler.trace_capture_in_progress or self.program_compiler.trace_active


class TracedExecutor:
    """Compile and replay traces over one exact `EagerExecutor`.

    ``Llama3Generator`` selects this target only when the requested operation
    is configured and eligible for tracing. This class never chooses an eager
    fallback; a caller that wants eager execution uses
    `eager_executor` directly.
    """

    def __init__(self, *, eager: EagerExecutor, trace_compiler: TraceCompiler, trace_mode: str = "all") -> None:
        if not isinstance(eager, EagerExecutor):
            raise TypeError("eager must be an EagerExecutor")
        if not isinstance(trace_compiler, TraceCompiler):
            raise TypeError("trace_compiler must be a TraceCompiler")
        if trace_compiler.program_compiler is not eager.program_compiler:
            raise ValueError("trace_compiler must compose eager.program_compiler")
        if trace_mode not in ("decode_only", "all"):
            raise ValueError("TracedExecutor trace_mode must be 'decode_only' or 'all'")
        self.eager_executor = eager
        self.trace_compiler = trace_compiler
        self.trace_mode = trace_mode
        self._coverage_miss_count = 0
        self._recent_prefill_replay_evidence: tuple[PrefillReplayEvidence, ...] = ()

    @property
    def coverage_miss_count(self) -> int:
        """Return strict operation coverage misses rejected before replay."""

        return self._coverage_miss_count

    @property
    def recent_prefill_replay_evidence(self) -> tuple[PrefillReplayEvidence, ...]:
        """Return evidence emitted by the most recent prepared public call."""

        return self._recent_prefill_replay_evidence

    def runtime_summary(self) -> dict[str, Any]:
        """Return the end-of-run counters required by serving qualification."""

        summary = self.eager_executor.runtime_summary()
        summary.update(
            {
                "successful_trace_replays": self.trace_compiler.replay_count,
                "trace_replays_by_operation": self.trace_compiler.replay_counts,
                "strict_coverage_misses": self._coverage_miss_count,
                "semantic_trace_count": self.trace_compiler.trace_count,
                "trace_association_count": self.trace_compiler.trace_association_count,
            }
        )
        return summary

    def log_runtime_summary(self, *, phase: str | None = None) -> dict[str, Any]:
        """Emit and return one structured serving-lifecycle summary."""

        summary = self.runtime_summary()
        if phase is not None:
            summary["phase"] = phase
        logger.info("TTTV2_RUNTIME_SUMMARY {}", json.dumps(summary, sort_keys=True))
        return summary

    # Public API

    def compile_prefill(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        sampling_params: Any = None,  # ↓ Sampling
    ) -> tuple[CompiledProgram, ...]:
        """Compile eager prefill programs and register their trace plans."""

        programs = []
        for prepared in self.eager_executor._prepare_prefill(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
        ):
            programs.extend(self._compile_prefill(prepared))
        return tuple(programs)

    def prefill_forward(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        sampling_params: Any = None,  # ↓ Sampling
    ):
        """Replay traced prefill and assemble the results."""

        prepared = self.prepare_prefill(
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
        )
        preflighted = self.preflight_prefill(prepared)
        return self.execute_prepared_prefill(
            preflighted,
            batch_size=int(tokens.shape[0]),
            sampling_params=sampling_params,
        )

    def prepare_prefill(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,
        sampling_params: Any = None,
    ) -> tuple[Any, ...]:
        """Prepare one traced public call without submitting device work."""

        return tuple(
            self.eager_executor._prepare_prefill(
                tokens=tokens,
                page_table=page_table,
                prompt_lens=prompt_lens,
                start_pos=start_pos,
                empty_slots=empty_slots,
                sampling_params=sampling_params,
            )
        )

    def preflight_prefill(self, prepared: Sequence[Any]) -> tuple[tuple[Any, Any], ...]:
        """Resolve complete trace coverage for already-prepared requests."""

        # Validate the complete public call before the first replay can write
        # KV. In particular, a later bucket/chunk trace miss must not leave an
        # earlier prepared item partially committed.
        return tuple((request, self._preflight_prefill(request)) for request in prepared)

    def execute_prepared_prefill(
        self,
        preflighted: Sequence[tuple[Any, Any]],
        *,
        batch_size: int,
        sampling_params: Any = None,
        lane: int = 0,
    ):
        """Replay an exact prepared/preflighted call without replanning it."""

        evidence: list[PrefillReplayEvidence] = []
        self._recent_prefill_replay_evidence = ()
        # A trace record owns one persistent output buffer. Consume each replay
        # before the next request with the same trace overwrites that buffer.
        results = (
            (request, self._execute_prefill(request, coverage, lane=lane, evidence=evidence))
            for request, coverage in preflighted
        )
        result = self.eager_executor.prefill.assemble(
            results,
            batch_size=batch_size,
            sampling_params=sampling_params,
        )
        self._recent_prefill_replay_evidence = tuple(evidence)
        return result

    def compile_decode(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
    ) -> CompiledProgram:
        """Compile the eager decode program and register its trace plan."""

        return self._compile_decode(
            self.eager_executor._prepare_decode(
                tokens=tokens,
                start_pos=start_pos,
                page_table=page_table,
                sampling_params=sampling_params,
                reset_batch=reset_batch,
            )
        )

    def decode_forward(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        sampling_params: Any = None,  # ↓ Sampling
        reset_batch: bool = False,  # ↓ State transition
        read_from_device: bool = True,  # ↓ Output policy
    ):
        """Replay one traced decode step and consume its output."""

        prepared = self.eager_executor._prepare_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )
        return self._execute_decode(
            prepared,
            read_from_device=read_from_device,
        )

    # Private implementation

    def _compile_prefill(self, prepared: Any):
        programs = self.eager_executor._compile_prefill(prepared)
        for program in programs:
            if self.trace_compiler.trace_key_for_program(program.key) is not None:
                continue
            operation_plan = self.eager_executor.prefill.capture_plan(prepared)
            self.trace_compiler.register_capture_plan(
                TraceCapturePlan(
                    program_key=program.key,
                    trace_signature=operation_plan.signature,
                    operation="prefill",
                    prepare_inputs=operation_plan.prepare_inputs,
                    capture=lambda persistent, plan=operation_plan: plan.capture(persistent.values),
                    refresh_policy=InputRefreshPolicy(every_replay=operation_plan.refresh_fields),
                    schema_fingerprint=getattr(operation_plan, "schema_fingerprint", None),
                    prepare_workspace=getattr(operation_plan, "prepare_workspace", None),
                    workspace_fingerprint=getattr(operation_plan, "workspace_fingerprint", None),
                )
            )
        return programs

    def _preflight_prefill(self, prepared: Any):
        # A compiled eager program can share geometry with a request that is not
        # trace-eligible. Program-key equality alone must never authorize replay.
        if prepared.trace_signature is None:
            self._raise_prefill_coverage_error(
                prepared,
                reason="the prepared request is not trace-eligible",
            )
        coverage = []
        for signature in prepared.program_signatures:
            program_key = self.eager_executor.program_compiler.key_for(signature)
            trace_key = self.trace_compiler.trace_key_for_program(program_key)
            record = self.trace_compiler.get(trace_key) if trace_key is not None else None
            if record is None or record.artifact is None:
                self._raise_prefill_coverage_error(
                    prepared,
                    signature=signature,
                    program_key=program_key,
                    trace_key=trace_key,
                    reason="the required trace is not registered and captured",
                )
            coverage.append((program_key, record))
        return tuple(coverage)

    def _execute_prefill(
        self,
        prepared: Any,
        coverage: Any = None,
        *,
        lane: int = 0,
        evidence: list[PrefillReplayEvidence] | None = None,
    ):
        coverage = self._preflight_prefill(prepared) if coverage is None else coverage
        if len(coverage) != 1:
            raise RuntimeError("Traced chunk replay requires one shared program geometry per prepared request")
        program_key, record = coverage[0]
        prefill = self.eager_executor.prefill
        canonical_workspace = hasattr(prepared, "request")
        workspace = (
            self.trace_compiler.workspace_for_program(program_key)
            if canonical_workspace
            else record.artifact.persistent_inputs.values
        )
        steps = prepared.request.chunks if hasattr(prepared, "request") else (None,)
        hidden = None
        for chunk in steps:
            hidden = self.trace_compiler.replay(
                program_key,
                lambda artifact, _decision, chunk=chunk: (
                    prefill.refresh_trace(prepared, artifact.persistent_inputs.values, workspace, chunk)
                    if canonical_workspace and chunk is not None
                    else prefill.refresh_trace(prepared, artifact.persistent_inputs.values, workspace)
                    if canonical_workspace
                    else prefill.refresh_trace(prepared, artifact.persistent_inputs.values)
                ),
                reset_batch=True,
            )
        if hidden is None:
            raise RuntimeError("Prepared prefill trace sequence contained no replay steps")
        result = self.eager_executor.prefill.finish_trace(
            prepared,
            hidden,
            workspace,
        )
        if isinstance(getattr(prepared, "request", None), PrefillRequest):
            request = prepared.request
            trace_key = self.trace_compiler.trace_key_for_program(program_key)
            signature = prepared.program_signatures[0]
            item = PrefillReplayEvidence(
                operation="prefill",
                variant=str(signature.operation_variant),
                sampling_path=str(prepared.sampling_path),
                execution="trace_replay",
                active_batch_size=len(request.source_rows),
                padded_batch_size=int(request.padded_batch_size),
                padded_sequence_length=int(request.padded_sequence_length),
                lane=int(lane),
                rank=int(lane),
                program_key=program_key.digest,
                trace_key="unassociated" if trace_key is None else trace_key.digest,
                replay_steps=len(steps),
            )
            if evidence is not None:
                evidence.append(item)
            logger.info("TTTV2_RUNTIME_EVIDENCE {}", json.dumps(asdict(item), sort_keys=True))
        return result

    def _raise_prefill_coverage_error(
        self,
        prepared: Any,
        *,
        reason: str,
        signature: Any = None,
        program_key: Any = None,
        trace_key: Any = None,
    ) -> None:
        self._coverage_miss_count += 1
        model = getattr(getattr(self.eager_executor.prefill, "config", None), "model", None)
        model_identity = (
            f"{type(model).__module__}.{type(model).__qualname__}"
            if model is not None
            else type(self.eager_executor.prefill).__qualname__
        )
        exact_signature = signature if signature is not None else getattr(prepared, "trace_signature", None)
        if exact_signature is None:
            exact_signature = tuple(getattr(prepared, "program_signatures", ()))
        material = _signature_material(exact_signature)
        configured = tuple(
            {
                "trace_key": key.digest,
                "signature": _signature_material(registered_signature),
            }
            for key, registered_signature in self.trace_compiler.registered_coverage("prefill")
        )
        digest = getattr(program_key, "digest", "unavailable")
        associated_trace = getattr(trace_key, "digest", "unavailable")
        raise TraceCoverageError(
            "Required prefill trace is unavailable: "
            f"reason={reason}; operation=prefill; trace_mode={self.trace_mode}; model={model_identity}; "
            f"signature_material={material!r}; signature_digest={digest}; "
            f"program_key={digest}; trace_key={associated_trace}; configured_coverage={configured!r}. "
            "Add the missing signature to construction-time trace coverage, or rerun with "
            "TraceConfig(mode='none') for debugging."
        )

    def _compile_decode(self, prepared: Any):
        program = self.eager_executor._compile_decode(prepared)
        if self.trace_compiler.trace_key_for_program(program.key) is None:
            operation_plan = self.eager_executor.decode.capture_plan(prepared)
            self.trace_compiler.register_capture_plan(
                TraceCapturePlan(
                    program_key=program.key,
                    trace_signature=self.eager_executor.decode.trace_signature(prepared),
                    operation="decode",
                    prepare_inputs=operation_plan.prepare_inputs,
                    capture=lambda persistent, plan=operation_plan: plan.capture(persistent.values),
                    refresh_policy=InputRefreshPolicy(
                        every_replay=operation_plan.refresh_policy.every_replay,
                        full_on_batch_reset=operation_plan.refresh_policy.full_on_batch_reset,
                        full_on_graph_switch=operation_plan.refresh_policy.full_on_graph_switch,
                        full_without_device_feedback=operation_plan.refresh_policy.full_without_device_feedback,
                        refresh_page_table_on_change=operation_plan.refresh_policy.refresh_page_table_on_change,
                    ),
                )
            )
        return program

    def _execute_decode(self, prepared: Any, *, read_from_device: bool = True):
        decode = self.eager_executor.decode
        signature = decode.program_signature(prepared)
        program_key = self.eager_executor.program_compiler.key_for(signature)
        trace_key = self.trace_compiler.trace_key_for_program(program_key)
        record = self.trace_compiler.get(trace_key) if trace_key is not None else None
        if record is None or record.artifact is None:
            self._coverage_miss_count += 1
            model = getattr(getattr(decode, "config", None), "model", None)
            model_identity = (
                f"{type(model).__module__}.{type(model).__qualname__}"
                if model is not None
                else type(decode).__qualname__
            )
            configured = tuple(
                {
                    "trace_key": key.digest,
                    "signature": _signature_material(registered_signature),
                }
                for key, registered_signature in self.trace_compiler.registered_coverage("decode")
            )
            associated_trace = "unavailable" if trace_key is None else trace_key.digest
            raise TraceCoverageError(
                "Required decode trace is unavailable: operation=decode; "
                f"trace_mode={self.trace_mode}; model={model_identity}; "
                f"signature_material={_signature_material(signature)!r}; "
                f"signature_digest={program_key.digest}; program_key={program_key.digest}; "
                f"trace_key={associated_trace}; configured_coverage={configured!r}. "
                "Add the missing signature to construction-time trace coverage, or rerun with "
                "TraceConfig(mode='none') for debugging."
            )
        output = self.trace_compiler.replay(
            program_key,
            lambda artifact, decision: decode.refresh_trace(artifact, prepared, decision),
            reset_batch=prepared.reset_batch,
            device_feedback_enabled=decode.config.position_feedback_capable,
            feedback_compatible=prepared.device_feedback,
            page_table_changed=prepared.page_table_changed,
        )
        decode.note_submitted(prepared)
        result = DecodeInvocationResult(
            value=output,
            owned=None,
            is_tokens=prepared.sampling_params is not None,
        )
        return decode.consume(result, read_from_device=read_from_device)


def _signature_material(signature: Any) -> Any:
    """Return stable diagnostic material without changing registry identity."""

    if isinstance(signature, tuple):
        return tuple(_signature_material(value) for value in signature)
    material = getattr(signature, "key_material", None)
    if material is None:
        return repr(signature)
    return material() if callable(material) else material


def _program_cache_entries(mesh_device: Any) -> int | None:
    """Read TTNN program-cache size when the concrete mesh exposes it."""

    devices = mesh_device.get_devices() if hasattr(mesh_device, "get_devices") else (mesh_device,)
    counts = []
    for device in devices:
        count = getattr(device, "num_program_cache_entries", None)
        if not callable(count):
            return None
        counts.append(int(count()))
    return sum(counts)
