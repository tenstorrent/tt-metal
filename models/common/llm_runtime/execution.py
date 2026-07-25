# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Concrete eager and traced execution by direct composition."""

from __future__ import annotations

from typing import Any

from models.common.llm_runtime.decode import DecodeRuntime
from models.common.llm_runtime.decode import InvocationResult as DecodeInvocationResult
from models.common.llm_runtime.prefill.runtime import PrefillRuntime
from models.common.llm_runtime.program_compiler import OutputSpec, ProgramCompiler
from models.common.llm_runtime.trace_compiler import InputRefreshPolicy, TraceCapturePlan, TraceCompiler


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

    # Public API

    def compile_prefill(self, **kwargs: Any) -> None:
        """Prepare and compile every eager program needed by one prefill call."""

        kwargs.pop("kv_cache", None)
        for prepared in self._prepare_prefill(**kwargs):
            self._compile_prefill(prepared)

    def prefill_forward(self, **kwargs: Any):
        """Prepare, execute, and assemble one eager prefill call."""

        prepared = self._prepare_prefill(**kwargs)
        results = tuple((request, self._execute_prefill(request)) for request in prepared)
        return self.prefill.assemble(
            results,
            batch_size=int(kwargs["tokens"].shape[0]),
            sampling_params=kwargs.get("sampling_params"),
        )

    def compile_decode(self, **kwargs: Any) -> None:
        """Prepare and compile the eager program needed by one decode call."""

        kwargs.pop("kv_cache", None)
        self._compile_decode(self._prepare_decode(**kwargs))

    def decode_forward(
        self,
        *,
        read_from_device: bool = True,
        **kwargs: Any,
    ):
        """Prepare and execute one eager decode call."""

        prepared = self._prepare_decode(**kwargs)
        return self._execute_decode(prepared, read_from_device=read_from_device)

    # Private implementation

    def _prepare_prefill(self, **kwargs: Any):
        return self.prefill.prepare(**kwargs)

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
        return self.prefill.invoke(prepared)

    def _prepare_decode(self, **kwargs: Any):
        return self.decode.prepare(**kwargs)

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

    def __init__(self, *, eager: EagerExecutor, trace_compiler: TraceCompiler) -> None:
        if not isinstance(eager, EagerExecutor):
            raise TypeError("eager must be an EagerExecutor")
        if not isinstance(trace_compiler, TraceCompiler):
            raise TypeError("trace_compiler must be a TraceCompiler")
        if trace_compiler.program_compiler is not eager.program_compiler:
            raise ValueError("trace_compiler must compose eager.program_compiler")
        self.eager_executor = eager
        self.trace_compiler = trace_compiler

    # Public API

    def compile_prefill(self, **kwargs: Any) -> None:
        """Compile eager prefill programs and register their trace plans."""

        kwargs.pop("kv_cache", None)
        for prepared in self.eager_executor._prepare_prefill(**kwargs):
            self._compile_prefill(prepared)

    def prefill_forward(self, **kwargs: Any):
        """Replay traced prefill and assemble the results."""

        prepared = self.eager_executor._prepare_prefill(**kwargs)
        results = tuple((request, self._execute_prefill(request)) for request in prepared)
        return self.eager_executor.prefill.assemble(
            results,
            batch_size=int(kwargs["tokens"].shape[0]),
            sampling_params=kwargs.get("sampling_params"),
        )

    def compile_decode(self, **kwargs: Any) -> None:
        """Compile the eager decode program and register its trace plan."""

        kwargs.pop("kv_cache", None)
        self._compile_decode(self.eager_executor._prepare_decode(**kwargs))

    def decode_forward(
        self,
        *,
        read_from_device: bool = True,
        **kwargs: Any,
    ):
        """Replay one traced decode step and consume its output."""

        prepared = self.eager_executor._prepare_decode(**kwargs)
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
                )
            )
        return programs

    def _execute_prefill(self, prepared: Any):
        signature = prepared.program_signatures[0]
        program_key = self.eager_executor.program_compiler.key_for(signature)
        hidden = self.trace_compiler.replay(
            program_key,
            lambda artifact, _decision: self.eager_executor.prefill.refresh_trace(
                prepared,
                artifact.persistent_inputs.values,
            ),
            reset_batch=True,
        )
        trace_key = self.trace_compiler.trace_key_for_program(program_key)
        record = self.trace_compiler.get(trace_key) if trace_key is not None else None
        if record is None or record.artifact is None:
            raise RuntimeError(f"Required prefill trace for program {program_key.digest} is unavailable")
        return self.eager_executor.prefill.finish_trace(
            prepared,
            hidden,
            record.artifact.persistent_inputs.values,
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
        program_key = self.eager_executor.program_compiler.key_for(decode.program_signature(prepared))
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
