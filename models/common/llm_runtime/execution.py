# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Eager execution and trace dispatch by direct composition."""

from __future__ import annotations

from typing import Any

from models.common.llm_runtime.decode import DecodeRuntime
from models.common.llm_runtime.decode import InvocationResult as DecodeInvocationResult
from models.common.llm_runtime.prefill import PrefillRuntime
from models.common.llm_runtime.program_compiler import OutputSpec, ProgramCompiler
from models.common.llm_runtime.trace_compiler import InputRefreshPolicy, TraceCapturePlan, TraceCompiler


class EagerExecutor:
    """Single eager semantic path and the fallback target for traced execution."""

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

    def prepare_prefill(self, **kwargs: Any):
        return self.prefill.prepare(**kwargs)

    def compile_prefill_prepared(self, prepared: Any):
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

    def execute_prefill_prepared(self, prepared: Any):
        self._require_ready_after_trace_gate(prepared.program_signatures)
        return self.prefill.invoke(prepared)

    def compile_prefill(self, *, enable_trace: bool | None = None, **kwargs: Any) -> None:
        kwargs.pop("kv_cache", None)
        for prepared in self.prepare_prefill(**kwargs):
            self.compile_prefill_prepared(prepared)

    def prefill_forward(self, *, enable_trace: bool | None = None, **kwargs: Any):
        prepared = self.prepare_prefill(**kwargs)
        results = tuple((request, self.execute_prefill_prepared(request)) for request in prepared)
        return self.prefill.assemble(
            results,
            batch_size=int(kwargs["tokens"].shape[0]),
            sampling_params=kwargs.get("sampling_params"),
        )

    def prepare_decode(self, **kwargs: Any):
        return self.decode.prepare(**kwargs)

    def compile_decode_prepared(self, prepared: Any):
        return self.program_compiler.compile(
            self.decode.program_signature(prepared),
            lambda _context: self.decode.invoke(prepared, device_feedback=prepared.device_feedback),
            output_spec=lambda result: OutputSpec.from_value(result.value),
            release_output=lambda result: result.owned,
        )

    def execute_decode_prepared(self, prepared: Any, *, read_from_device: bool = True):
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

    def compile_decode(self, *, enable_trace: bool | None = None, **kwargs: Any) -> None:
        kwargs.pop("kv_cache", None)
        self.compile_decode_prepared(self.prepare_decode(**kwargs))

    def decode_forward(
        self,
        *,
        read_from_device: bool = True,
        enable_trace: bool | None = None,
        **kwargs: Any,
    ):
        prepared = self.prepare_decode(**kwargs)
        return self.execute_decode_prepared(prepared, read_from_device=read_from_device)


class TracedExecutor:
    """Trace dispatch that delegates all semantic preparation and fallback to eager."""

    def __init__(self, *, eager: EagerExecutor, trace_compiler: TraceCompiler) -> None:
        if not isinstance(eager, EagerExecutor):
            raise TypeError("eager must be an EagerExecutor")
        if not isinstance(trace_compiler, TraceCompiler):
            raise TypeError("trace_compiler must be a TraceCompiler")
        if trace_compiler.program_compiler is not eager.program_compiler:
            raise ValueError("trace_compiler must compose eager.program_compiler")
        self.eager = eager
        self.trace_compiler = trace_compiler

    @property
    def prefill(self) -> PrefillRuntime:
        return self.eager.prefill

    @property
    def decode(self) -> DecodeRuntime:
        return self.eager.decode

    @property
    def program_compiler(self) -> ProgramCompiler:
        return self.eager.program_compiler

    def prepare_prefill(self, **kwargs: Any):
        return self.eager.prepare_prefill(**kwargs)

    def compile_prefill_prepared(self, prepared: Any, *, enable_trace: bool):
        programs = self.eager.compile_prefill_prepared(prepared)
        if not enable_trace or not prepared.trace_eligible:
            return programs
        for program in programs:
            if self.trace_compiler.trace_key_for_program(program.key) is not None:
                continue
            operation_plan = self.prefill.capture_plan(prepared)
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

    def execute_prefill_prepared(self, prepared: Any, *, enable_trace: bool):
        if not enable_trace or not prepared.trace_eligible:
            return self.eager.execute_prefill_prepared(prepared)
        signature = prepared.program_signatures[0]
        program_key = self.program_compiler.key_for(signature)
        hidden = self.trace_compiler.replay(
            program_key,
            lambda artifact, _decision: self.prefill.refresh_trace(
                prepared,
                artifact.persistent_inputs.values,
            ),
            reset_batch=True,
        )
        trace_key = self.trace_compiler.trace_key_for_program(program_key)
        record = self.trace_compiler.get(trace_key) if trace_key is not None else None
        if record is None or record.artifact is None:
            raise RuntimeError(f"Required prefill trace for program {program_key.digest} is unavailable")
        return self.prefill.finish_trace(prepared, hidden, record.artifact.persistent_inputs.values)

    def compile_prefill(self, *, enable_trace: bool | None = None, **kwargs: Any) -> None:
        kwargs.pop("kv_cache", None)
        trace = _resolve_trace_hint(self.trace_compiler.trace_config, "prefill", enable_trace)
        for prepared in self.prepare_prefill(**kwargs):
            self.compile_prefill_prepared(prepared, enable_trace=trace)

    def prefill_forward(self, *, enable_trace: bool | None = None, **kwargs: Any):
        trace = _resolve_trace_hint(self.trace_compiler.trace_config, "prefill", enable_trace)
        prepared = self.prepare_prefill(**kwargs)
        results = tuple((request, self.execute_prefill_prepared(request, enable_trace=trace)) for request in prepared)
        return self.prefill.assemble(
            results,
            batch_size=int(kwargs["tokens"].shape[0]),
            sampling_params=kwargs.get("sampling_params"),
        )

    def prepare_decode(self, **kwargs: Any):
        return self.eager.prepare_decode(**kwargs)

    def compile_decode_prepared(self, prepared: Any, *, enable_trace: bool):
        program = self.eager.compile_decode_prepared(prepared)
        if enable_trace and self.trace_compiler.trace_key_for_program(program.key) is None:
            operation_plan = self.decode.capture_plan(prepared)
            self.trace_compiler.register_capture_plan(
                TraceCapturePlan(
                    program_key=program.key,
                    trace_signature=self.decode.trace_signature(prepared),
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

    def execute_decode_prepared(self, prepared: Any, *, enable_trace: bool, read_from_device: bool = True):
        if not enable_trace:
            return self.eager.execute_decode_prepared(prepared, read_from_device=read_from_device)
        program_key = self.program_compiler.key_for(self.decode.program_signature(prepared))
        output = self.trace_compiler.replay(
            program_key,
            lambda artifact, decision: self.decode.refresh_trace(artifact, prepared, decision),
            reset_batch=prepared.reset_batch,
            device_feedback_enabled=self.decode.device_feedback_enabled,
            feedback_compatible=prepared.device_feedback,
            page_table_changed=prepared.page_table_changed,
        )
        self.decode.note_submitted(prepared)
        result = DecodeInvocationResult(
            value=output,
            owned=None,
            is_tokens=prepared.sampling_params is not None,
        )
        return self.decode.consume(result, read_from_device=read_from_device)

    def compile_decode(self, *, enable_trace: bool | None = None, **kwargs: Any) -> None:
        kwargs.pop("kv_cache", None)
        trace = _resolve_trace_hint(self.trace_compiler.trace_config, "decode", enable_trace)
        self.compile_decode_prepared(self.prepare_decode(**kwargs), enable_trace=trace)

    def decode_forward(
        self,
        *,
        read_from_device: bool = True,
        enable_trace: bool | None = None,
        **kwargs: Any,
    ):
        trace = _resolve_trace_hint(self.trace_compiler.trace_config, "decode", enable_trace)
        prepared = self.prepare_decode(**kwargs)
        return self.execute_decode_prepared(
            prepared,
            enable_trace=trace,
            read_from_device=read_from_device,
        )


def _resolve_trace_hint(trace_config: Any, operation: str, enable_trace: bool | None) -> bool:
    configured = bool(getattr(trace_config, f"{operation}_enabled"))
    if enable_trace is None:
        return configured
    if not isinstance(enable_trace, bool):
        raise TypeError("enable_trace must be bool or None")
    if enable_trace and not configured:
        raise ValueError(f"enable_trace=True disagrees with static {operation} trace policy")
    return enable_trace
