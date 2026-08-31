# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Decode preparation, invocation, feedback, readback, and local resources."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
from dataclasses import dataclass, field
from typing import Any

import torch
from loguru import logger

import ttnn
from models.common.llm_runtime.config import PageTableLayout
from models.common.llm_runtime.output_reader import OutputReader, PendingRead
from models.common.llm_runtime.tensor_resources import (
    TensorResourceOrphan,
    attach_cleanup_failures,
    best_effort_deallocate_owned_tensors,
    raise_cleanup_failures,
    release_orphans,
)
from models.common.modules.sampling.params import (
    PreparedSamplingParams,
    place_prepared_sampling_params,
    prepare_sampling_params,
    slice_sampling_params,
)
from models.common.modules.sampling.seed_manager_1d import SeedManager1D
from models.common.sampling.sampling_params import SamplingParams


@dataclass(frozen=True)
class DecodeProgramSignature:
    """Material identity of one decode eager-program variant."""

    batch_size: int
    page_table_width: int
    sampling_path: str
    device_feedback: bool
    penalties_enabled: bool = False
    logprobs_enabled: bool = False

    def key_material(self) -> tuple[tuple[str, Any], ...]:
        material = [
            ("operation", "decode"),
            ("batch_size", self.batch_size),
            ("page_table_width", self.page_table_width),
            ("sampling_path", self.sampling_path),
            ("device_feedback", self.device_feedback),
        ]
        if self.penalties_enabled:
            material.append(("penalties_enabled", True))
        if self.logprobs_enabled:
            material.append(("logprobs_enabled", True))
        return tuple(material)


@dataclass(frozen=True)
class DecodeTraceSignature:
    """Material identity of one full-step decode trace."""

    batch_size: int
    page_table_width: int
    sampling_path: str
    device_feedback: bool
    penalties_enabled: bool = False
    logprobs_enabled: bool = False

    def key_material(self) -> tuple[tuple[str, Any], ...]:
        material = [
            ("operation", "decode"),
            ("batch_size", self.batch_size),
            ("page_table_width", self.page_table_width),
            ("sampling_path", self.sampling_path),
            ("device_feedback", self.device_feedback),
        ]
        if self.penalties_enabled:
            material.append(("penalties_enabled", True))
        if self.logprobs_enabled:
            material.append(("logprobs_enabled", True))
        return tuple(material)


@dataclass(frozen=True)
class DecodeHostInputs:
    tokens: Any
    positions: Any
    rotary_indices: Any
    page_table: Any

    def values(self) -> tuple[Any, Any, Any, Any]:
        return self.tokens, self.positions, self.rotary_indices, self.page_table


@dataclass(frozen=True)
class DecodeDeviceInputs:
    tokens: Any
    positions: Any
    rotary_indices: Any
    page_table: Any

    def values(self) -> tuple[Any, Any, Any, Any]:
        return self.tokens, self.positions, self.rotary_indices, self.page_table

    def owned_tensor_values(self) -> tuple[Any, Any, Any, Any]:
        return self.values()


@dataclass(frozen=True)
class PreparedDecode:
    """One validated and normalized decode request, prepared exactly once."""

    tokens: torch.Tensor
    start_pos: torch.Tensor
    page_table: torch.Tensor
    sampling_params: SamplingParams | None
    prepared_sampling: PreparedSamplingParams | None
    sampling_path: str
    reset_batch: bool
    device_feedback: bool
    page_table_changed: bool

    @property
    def sampling_values(self):
        """Compatibility view over the native prepared structure."""

        sampling = self.prepared_sampling
        if sampling is None:
            return None
        return (
            sampling.top_k,
            sampling.top_p,
            sampling.temperature,
            sampling.all_active_rows_greedy,
        )

    @property
    def sampling_seeds(self):
        return None if self.prepared_sampling is None else self.prepared_sampling.seeds


@dataclass(frozen=True)
class InvocationResult:
    value: Any
    owned: Any
    is_tokens: bool


@dataclass(frozen=True)
class DecodeRefreshPolicy:
    every_replay: tuple[str, ...] = ("sampling",)
    full_on_batch_reset: bool = True
    full_on_graph_switch: bool = True
    full_without_device_feedback: bool = True
    refresh_page_table_on_change: bool = True


@dataclass(frozen=True)
class DecodeCapturePlan:
    """Operation callbacks consumed by the trace compiler by duck typing."""

    prepare_inputs: Any
    capture: Any
    refresh_policy: DecodeRefreshPolicy = DecodeRefreshPolicy()


@dataclass(frozen=True)
class DecodePersistentInputs:
    device_inputs: DecodeDeviceInputs
    kpt: tuple[Any, Any, Any] | None
    kpt_signature: list[Any] | None = None
    seed_buffer: Any | None = None

    def owned_tensor_values(self) -> tuple[Any, ...]:
        return self.device_inputs.values(), self.kpt


@dataclass
class DecodeOutputLease:
    raw_value: Any
    owned_values: Any
    host_value: Any = None
    pending: PendingRead | None = None
    released: bool = False
    deallocated_tensor_ids: set[int] = field(default_factory=set, repr=False)


@dataclass(frozen=True)
class DecodeRuntimeConfig:
    """Fully resolved, immutable decode policy and borrowed collaborators."""

    model: Any
    mesh_device: Any
    output_reader: OutputReader
    lane_capacity: int
    page_table_layout: PageTableLayout  # Current geometry; may be replaced once before execution.
    page_table_layout_ceiling: PageTableLayout  # Construction-time upper bound retained across replacement.
    cluster_shape: tuple[int, int]
    num_devices: int
    vocab_size: int
    device_sampling_enabled: bool
    force_greedy_top_k: bool
    allow_force_argmax: bool
    max_device_top_k: int
    sampling_batch_size: int
    position_feedback_capable: bool
    sampling_state_controller: Any
    sampling_state: Any

    def __post_init__(self) -> None:
        _validate_resolved_decode_config(self)

    @classmethod
    def resolve(
        cls,
        *,
        model: Any,
        output_reader: OutputReader,
        lane_capacity: int,
        page_table_layout: PageTableLayout,
        device_sampling_enabled: bool,
        force_greedy_top_k: bool = False,
        sampling_state_controller: Any = None,
        sampling_state: Any = None,
    ) -> "DecodeRuntimeConfig":
        if not isinstance(output_reader, OutputReader):
            raise TypeError("output_reader must be an OutputReader")
        mesh_device = output_reader.mesh_device
        model_mesh = getattr(getattr(model, "config", None), "mesh_device", None)
        if model_mesh is not None and model_mesh is not mesh_device:
            raise ValueError("model and decode runtime must use the same mesh_device")
        if not isinstance(lane_capacity, int) or isinstance(lane_capacity, bool) or lane_capacity <= 0:
            raise ValueError("lane_capacity must be a positive integer")
        if lane_capacity > 32:
            raise ValueError("decode token input padding supports at most 32 lane slots")
        if not isinstance(device_sampling_enabled, bool):
            raise TypeError("device_sampling_enabled must be bool")
        if not isinstance(force_greedy_top_k, bool):
            raise TypeError("force_greedy_top_k must be bool")
        _validate_page_table_layout(page_table_layout)

        try:
            cluster_shape = tuple(int(value) for value in mesh_device.shape)
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError("mesh_device must provide a two-dimensional shape") from error
        if len(cluster_shape) != 2 or any(value <= 0 for value in cluster_shape):
            raise ValueError("mesh_device shape must contain two positive dimensions")
        model_config = getattr(model, "config", None)
        num_devices = getattr(model_config, "num_devices", None)
        if num_devices is None:
            num_devices = getattr(model, "num_devices", cluster_shape[0] * cluster_shape[1])
        if not isinstance(num_devices, int) or isinstance(num_devices, bool) or num_devices <= 0:
            raise ValueError("model num_devices must be a positive integer")
        if num_devices != cluster_shape[0] * cluster_shape[1]:
            raise ValueError("model num_devices must match the decode mesh shape")
        vocab_size = getattr(model, "vocab_size", None)
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
            raise ValueError("model vocab_size must be a positive integer")

        sampling = getattr(model, "sampling", None)
        sampling_config = getattr(sampling, "config", None)
        allow_force_argmax = getattr(sampling_config, "allow_force_argmax", False)
        max_device_top_k = getattr(sampling_config, "max_top_k", 0)
        sampling_batch_size = getattr(sampling_config, "max_batch_size", lane_capacity)
        if device_sampling_enabled:
            if not callable(getattr(sampling, "decode_forward", None)):
                raise TypeError("device sampling requires model.sampling.decode_forward()")
            if not isinstance(allow_force_argmax, bool):
                raise TypeError("model sampling allow_force_argmax must be bool")
            if not isinstance(max_device_top_k, int) or isinstance(max_device_top_k, bool) or max_device_top_k <= 0:
                raise ValueError("model sampling max_top_k must be a positive integer")
            if (
                not isinstance(sampling_batch_size, int)
                or isinstance(sampling_batch_size, bool)
                or sampling_batch_size < lane_capacity
            ):
                raise ValueError("model sampling max_batch_size must cover the decode lane capacity")
        else:
            allow_force_argmax = False
            max_device_top_k = 0
            sampling_batch_size = lane_capacity

        return cls(
            model=model,
            mesh_device=mesh_device,
            output_reader=output_reader,
            lane_capacity=lane_capacity,
            page_table_layout=page_table_layout,
            cluster_shape=cluster_shape,
            num_devices=num_devices,
            vocab_size=vocab_size,
            device_sampling_enabled=device_sampling_enabled,
            force_greedy_top_k=force_greedy_top_k,
            allow_force_argmax=allow_force_argmax,
            max_device_top_k=max_device_top_k,
            sampling_batch_size=sampling_batch_size,
            position_feedback_capable=callable(getattr(model, "increment_positions", None)),
            sampling_state_controller=sampling_state_controller,
            sampling_state=sampling_state,
            page_table_layout_ceiling=page_table_layout,
        )

    def with_page_table_layout(self, layout: PageTableLayout) -> "DecodeRuntimeConfig":
        """Return a validated geometry replacement within the original ceiling."""

        _validate_page_table_layout(layout)
        if layout.block_size != self.page_table_layout.block_size:
            raise ValueError("replacement page-table layout cannot change block_size")
        if layout.raw_capacity_width > self.page_table_layout_ceiling.raw_capacity_width:
            raise ValueError("replacement page-table capacity exceeds the construction-time ceiling")
        if layout.decode_width > self.page_table_layout_ceiling.decode_width:
            raise ValueError("replacement decode width exceeds the construction-time ceiling")
        return dataclasses.replace(self, page_table_layout=layout)


class DecodeRuntime:
    """Prepare, execute, trace, and consume decode for one execution lane.

    The eager call chain is
    `EagerExecutor.decode_forward()` → `prepare` → `invoke` →
    `consume`. Trace warmup uses `capture_plan`; replay calls
    `refresh_trace`, `note_submitted`, and `consume`.
    `Llama3Executor` also exposes `read_decode_output` and
    `process_decode_output_host` for vLLM's asynchronous output path.

    The model, mesh, output reader, sampler, and KV-backed page-table values are
    borrowed. Only staged invocation tensors, raw outputs, output leases, and
    retryable decode transients are released here.

    Explicit ``SamplingParams.seed`` routing is a decode-top-k contract. The
    common prefill runtime retains its existing sampling behavior and does not
    consume this decode RNG stream; callers that need one coherent controlled
    stream must keep prefill on the logits path and begin sampling in decode.
    """

    def __init__(self, config: DecodeRuntimeConfig):
        if not isinstance(config, DecodeRuntimeConfig):
            raise TypeError("config must be a DecodeRuntimeConfig")
        self.config = config
        self._previous_page_table: torch.Tensor | None = None
        self._normalization_source: torch.Tensor | None = None
        self._normalization_copy_blocks: tuple[int, ...] | None = None
        self._normalization_layout: tuple[int, int, int] | None = None
        self._normalized_page_table: torch.Tensor | None = None
        self._external_by_raw_id: dict[int, DecodeOutputLease] = {}
        self._external_by_host_id: dict[int, DecodeOutputLease] = {}
        self._transient_orphans: list[TensorResourceOrphan] = []
        self._sampling_state_controller = config.sampling_state_controller
        self._sampling_state = config.sampling_state
        sampling_config = getattr(getattr(config.model, "sampling", None), "config", None)
        seed_buffer = getattr(sampling_config, "seeds", None)
        has_mutable_seed_buffer = all(
            callable(getattr(seed_buffer, name, None)) for name in ("update", "get_device_buffer")
        ) and isinstance(getattr(seed_buffer, "source", None), torch.Tensor)
        self._seed_manager = (
            self._sampling_state_controller.seed_manager
            if self._sampling_state_controller is not None
            else SeedManager1D(sampling_config)
            if config.device_sampling_enabled and has_mutable_seed_buffer
            else None
        )
        self._seed_state = (
            self._sampling_state.seed_state
            if self._sampling_state is not None
            else self._seed_manager.create_state()
            if self._seed_manager is not None
            else None
        )
        if self._seed_state is not None and self._seed_state.capacity < config.lane_capacity:
            raise ValueError("model sampling seed buffer is smaller than the decode lane capacity")

    # Public API

    @property
    def transient_orphan_count(self) -> int:
        """Return the number of failed transient releases awaiting cleanup."""

        return len(self._transient_orphans)

    def configure_page_table_layout(self, layout: PageTableLayout) -> None:
        """Install final physical-capacity geometry before allocation."""

        self.config = self.config.with_page_table_layout(layout)

    def prepare(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        sampling_params: Any = None,  # ↓ Sampling
        prompt_tokens: Any = None,  # ↓ Request-owned sampling state
        output_tokens: Any = None,
        slot_remap: Any = None,
        reset_batch: bool = False,  # ↓ State transition
    ) -> PreparedDecode:
        """Normalize one host decode request into an immutable prepared value."""

        self._ensure_usable()
        self._validate_inputs(tokens, start_pos, page_table)
        self._validate_sampling_request(sampling_params)
        feedback = self._classify_feedback(sampling_params)
        prepared_sampling = None
        if sampling_params is not None:
            active_slots = tuple(slot for slot, position in enumerate(start_pos) if int(position) >= 0)
            if not active_slots:
                raise ValueError("decode sampling requires at least one active slot")
            request_sampling = slice_sampling_params(sampling_params, active_slots)
            prepared_sampling = prepare_sampling_params(
                request_sampling,
                self.config.sampling_batch_size,
                max_device_top_k=self.config.max_device_top_k,
                allow_force_argmax=self.config.allow_force_argmax and not self.config.force_greedy_top_k,
                prompt_tokens=_select_decode_request_state(
                    prompt_tokens,
                    active_slots=active_slots,
                    lane_capacity=self.config.lane_capacity,
                ),
                output_tokens=_select_decode_request_state(
                    output_tokens,
                    active_slots=active_slots,
                    lane_capacity=self.config.lane_capacity,
                ),
                slot_remap=_normalize_decode_slot_remap(
                    slot_remap,
                    lane_capacity=self.config.lane_capacity,
                    sampling_batch_size=self.config.sampling_batch_size,
                ),
            )
            prepared_sampling = place_prepared_sampling_params(
                prepared_sampling,
                active_slots,
            )
        normalized = self._normalize_page_table(
            page_table,
            start_pos,
            allow_one_step_feedback_lag=feedback,
        )
        return PreparedDecode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=normalized,
            sampling_params=sampling_params,
            prepared_sampling=prepared_sampling,
            sampling_path=self._classify_sampling_path(prepared_sampling),
            reset_batch=bool(reset_batch),
            device_feedback=feedback,
            page_table_changed=(
                self._previous_page_table is None or not torch.equal(self._previous_page_table, normalized)
            ),
        )

    def program_signature(self, prepared: PreparedDecode) -> DecodeProgramSignature:
        """Return the eager program identity for a prepared decode request."""

        self._require_prepared(prepared)
        return self._program_signature(prepared)

    def trace_signature(self, prepared: PreparedDecode) -> DecodeTraceSignature:
        """Return the trace identity for a prepared decode request."""

        self._require_prepared(prepared)
        program = self._program_signature(prepared)
        return DecodeTraceSignature(
            batch_size=program.batch_size,
            page_table_width=program.page_table_width,
            sampling_path=program.sampling_path,
            penalties_enabled=program.penalties_enabled,
            logprobs_enabled=program.logprobs_enabled,
            device_feedback=program.device_feedback,
        )

    def invoke(
        self,
        prepared: PreparedDecode,
        *,
        device_feedback: bool = False,
        count_tokens: bool = True,
    ) -> InvocationResult:
        """Stage and execute one prepared request eagerly."""

        self._ensure_usable()
        self._require_prepared(prepared)
        host_inputs = self._prepare_inputs_host(prepared)
        device_inputs, kpt = self._stage_inputs_and_kpt(host_inputs, prepared)
        owned = (device_inputs, kpt)
        compile_only_state = not count_tokens and (
            self._sampling_state_controller is not None or self._seed_manager is not None
        )
        try:
            if compile_only_state:
                sampling = prepared.prepared_sampling
                if self._sampling_state_controller is not None and sampling is not None:
                    self._sampling_state_controller.reset(
                        self._sampling_state,
                        dataclasses.replace(sampling, slot_remap=None),
                    )
                elif self._seed_manager is not None:
                    # Program warmup compiles sampled decode before any real
                    # prefill/admission boundary.  Give the fallback seed
                    # manager a temporary replacement batch so its normal
                    # strict serving checks do not mistake synthetic warmup
                    # rows for live requests.
                    self._refresh_sampling_seeds(dataclasses.replace(prepared, reset_batch=True))
            else:
                self._refresh_sampling_seeds(prepared)
            with _validate_module_inputs(self.config.model):
                output = self._run_body(
                    device_inputs,
                    prepared,
                    kpt,
                    device_feedback=device_feedback and prepared.device_feedback,
                    count_tokens=count_tokens,
                    advance_seeds=count_tokens,
                )
        except BaseException as primary:
            if compile_only_state:
                try:
                    self._reset_compile_only_sampling_state()
                except BaseException as cleanup_error:
                    attach_cleanup_failures(primary, (cleanup_error,))
            failures = self._release_or_retain_transient(owned)
            attach_cleanup_failures(primary, failures)
            raise
        if compile_only_state:
            self._reset_compile_only_sampling_state()
        self._note_submitted(prepared)
        return InvocationResult(
            value=output,
            owned=(output, owned),
            is_tokens=prepared.sampling_params is not None,
        )

    def capture_plan(self, prepared: PreparedDecode) -> DecodeCapturePlan:
        """Describe persistent inputs and capture work for one decode trace."""

        self._require_prepared(prepared)

        def prepare_inputs() -> DecodePersistentInputs:
            if self._sampling_state_controller is not None and prepared.prepared_sampling is not None:
                self._sampling_state_controller.reset(
                    self._sampling_state,
                    dataclasses.replace(prepared.prepared_sampling, slot_remap=None),
                )
            host_inputs = self._prepare_inputs_host(prepared)
            device_inputs, kpt = self._stage_inputs_and_kpt(host_inputs, prepared)
            sampling = prepared.prepared_sampling
            signature = [(sampling.top_k, sampling.top_p, sampling.temperature)] if kpt is not None else None
            return DecodePersistentInputs(
                device_inputs=device_inputs,
                kpt=kpt,
                kpt_signature=signature,
                seed_buffer=self._seed_device_handle(),
            )

        def capture(persistent: Any) -> Any:
            values = _persistent_values(persistent)
            return self._run_body(
                values.device_inputs,
                prepared,
                values.kpt,
                device_feedback=prepared.device_feedback,
                advance_seeds=False,
            )

        return DecodeCapturePlan(prepare_inputs=prepare_inputs, capture=capture)

    def refresh_trace(
        self,
        artifact: Any,
        prepared: PreparedDecode,
        decision: Any,
    ) -> None:
        self._require_prepared(prepared)
        values = _persistent_values(artifact)
        self._validate_trace_seed_handle(values)
        # Sampling1D's trace captures the stable model-owned seed tensor handle.
        # Refresh its contents before replay so the trace cannot observe stale
        # request state.
        self._refresh_sampling_seeds(prepared)
        if bool(decision.full):
            host_inputs = self._prepare_inputs_host(prepared)
            _copy_host_to_device(host_inputs.values(), values.device_inputs.values())
        elif bool(decision.page_table):
            host_inputs = self._prepare_inputs_host(prepared)
            ttnn.copy_host_to_device_tensor(host_inputs.page_table, values.device_inputs.page_table)
        if prepared.sampling_path == "topk":
            sampling = prepared.prepared_sampling
            if sampling is None:
                raise RuntimeError("top-k decode trace is missing prepared sampling parameters")
            signature = sampling.top_k, sampling.top_p, sampling.temperature
            if values.kpt_signature is None or values.kpt_signature[0] != signature:
                self._refresh_kpt(values.kpt, prepared)
                if values.kpt_signature is not None:
                    values.kpt_signature[0] = signature
        elif values.kpt is not None:
            raise RuntimeError("non-top-k decode trace unexpectedly owns sampling inputs")

    def note_submitted(self, prepared: PreparedDecode) -> None:
        """Advance feedback comparison state immediately after device submission."""
        self._require_prepared(prepared)
        self._note_submitted(prepared)

    def consume(self, result: InvocationResult, *, read_from_device: bool = True) -> Any:
        """Read and normalize an invocation or transfer it to an external lease."""
        if not isinstance(result, InvocationResult):
            raise TypeError("result must be an InvocationResult")
        if not read_from_device:
            if result.owned is not None:
                lease = DecodeOutputLease(raw_value=result.value, owned_values=result.owned)
                self._external_by_raw_id[id(result.value)] = lease
            return result.value
        try:
            host = self.config.output_reader.read(result.value, blocking=True)
            normalized = self._normalize_host_output(
                host,
                is_tokens=result.is_tokens,
            )
        except BaseException as primary:
            failures = self._release_or_retain_transient(result.owned)
            attach_cleanup_failures(primary, failures)
            raise
        failures = self._release_or_retain_transient(result.owned)
        if failures:
            raise_cleanup_failures(failures)
        return normalized

    def read_decode_output(self, tt_out: Any, *, async_read: bool = False) -> Any:
        """Read a raw externally leased decode output, optionally asynchronously."""

        if not async_read:
            host = self.config.output_reader.read(tt_out, blocking=True)
            self._release_external_lease(self._external_by_raw_id.get(id(tt_out)))
            return host
        pending = self.config.output_reader.submit(tt_out)
        lease = self._external_by_raw_id.get(id(tt_out))
        if lease is not None:
            lease.host_value = pending.value
            lease.pending = pending
            self._external_by_host_id[id(pending.value)] = lease
        return pending.value, list(pending.events)

    def process_decode_output_host(self, tt_out: Any, *, is_tokens: bool = False) -> tuple[Any, Any]:
        """Complete and normalize a host value returned by async decode read."""

        completed = self.config.output_reader.complete(tt_out)
        self._release_external_lease(self._external_by_host_id.get(id(tt_out)))
        return self._normalize_host_output(completed, is_tokens=is_tokens)

    def drain_external_outputs(self) -> None:
        """Synchronize and release every outstanding externally owned output."""

        failures = []
        for lease in tuple(self._external_by_raw_id.values()):
            try:
                if lease.pending is None:
                    ttnn.synchronize_device(self.config.mesh_device)
                self._release_external_lease(lease)
            except BaseException as error:
                failures.append(error)
        if failures:
            raise_cleanup_failures(failures)

    def cleanup_transients(self) -> None:
        """Retry every transient tensor release that previously failed."""

        failures = release_orphans(self._transient_orphans)
        if failures:
            raise_cleanup_failures(failures)

    # Private implementation

    def _validate_sampling_request(self, sampling_params: SamplingParams | None) -> None:
        if sampling_params is not None and not self.config.device_sampling_enabled:
            raise ValueError("sampling parameters were supplied while device sampling is disabled")

    def _classify_sampling_path(self, prepared_sampling: PreparedSamplingParams | None) -> str:
        if prepared_sampling is None:
            return "logits"
        return prepared_sampling.sampling_path

    def _classify_feedback(self, sampling_params: SamplingParams | None) -> bool:
        return sampling_params is not None and self.config.position_feedback_capable

    def _convert_logits(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            output = value.float()
        elif self.config.num_devices == 1:
            output = ttnn.to_torch(value).float()
        else:
            output = _concat_host_output(value, self.config.cluster_shape).float()
        return self._slice_logits(output)

    def _slice_logits(self, output: torch.Tensor) -> torch.Tensor:
        config = self.config
        return output[:, :, : config.lane_capacity, : config.vocab_size].contiguous().view(config.lane_capacity, 1, -1)

    def _program_signature(self, prepared: PreparedDecode) -> DecodeProgramSignature:
        return DecodeProgramSignature(
            batch_size=self.config.lane_capacity,
            page_table_width=int(prepared.page_table.shape[-1]),
            sampling_path=prepared.sampling_path,
            penalties_enabled=(
                prepared.prepared_sampling.penalties_enabled if prepared.prepared_sampling is not None else False
            ),
            logprobs_enabled=(
                prepared.prepared_sampling.log_probs_enabled if prepared.prepared_sampling is not None else False
            ),
            device_feedback=prepared.device_feedback,
        )

    def _note_submitted(self, prepared: PreparedDecode) -> None:
        self._previous_page_table = prepared.page_table.clone()

    def _normalize_host_output(self, host_output: Any, *, is_tokens: bool) -> tuple[Any, Any]:
        if isinstance(host_output, tuple):
            if len(host_output) != 2:
                raise TypeError("runtime output tuple must contain (output, log_probs)")
            output, log_probs = host_output
        else:
            output, log_probs = host_output, None
        if is_tokens:
            tokens = _process_output_tokens(output, self.config.lane_capacity, self.config.cluster_shape)
            return tokens.to(torch.int64), _process_sampled_log_probs(log_probs, self.config.lane_capacity)
        return self._convert_logits(output), log_probs

    def _normalize_page_table(self, page_table, start_pos, *, allow_one_step_feedback_lag):
        layout = self.config.page_table_layout
        raw_width = layout.raw_capacity_width
        decode_width = layout.decode_width
        block_size = layout.block_size
        copy_blocks_by_row = []
        for row, position_value in enumerate(start_pos):
            position = int(position_value)
            used_blocks = _num_blocks(max(0, position + 1), block_size)
            if used_blocks > raw_width:
                raise ValueError("decode position exceeds the configured paged-KV capacity")
            if int(page_table.shape[1]) < used_blocks:
                raise ValueError(f"page table is too narrow for decode row {row}")
            copy_blocks = used_blocks
            if allow_one_step_feedback_lag and position >= 0 and (position + 1) % block_size == 0:
                copy_blocks = min(used_blocks + 1, raw_width, int(page_table.shape[1]))
            copy_blocks_by_row.append(copy_blocks)

        layout = (raw_width, decode_width, block_size)
        copy_blocks_by_row = tuple(copy_blocks_by_row)
        source = self._normalization_source
        if (
            source is not None
            and self._normalization_layout == layout
            and self._normalization_copy_blocks == copy_blocks_by_row
            and source.shape == page_table.shape
            and source.device == page_table.device
            and source.dtype == page_table.dtype
            and torch.equal(source, page_table)
        ):
            assert self._normalized_page_table is not None
            return self._normalized_page_table

        normalized = torch.zeros((int(page_table.shape[0]), decode_width), dtype=torch.int32, device=page_table.device)
        for row, copy_blocks in enumerate(copy_blocks_by_row):
            normalized[row, :copy_blocks] = page_table[row, :copy_blocks].to(torch.int32)
        self._normalization_source = page_table.clone()
        self._normalization_copy_blocks = copy_blocks_by_row
        self._normalization_layout = layout
        self._normalized_page_table = normalized
        return normalized

    def _prepare_inputs_host(self, prepared: PreparedDecode) -> DecodeHostInputs:
        config = self.config
        padded = torch.nn.functional.pad(prepared.tokens.reshape(-1), (0, 32 - config.lane_capacity))
        tokens_tt = ttnn.unsqueeze_to_4D(
            ttnn.from_torch(
                padded,
                device=None,
                dtype=ttnn.uint32,
                mesh_mapper=ttnn.ReplicateTensorToMesh(config.mesh_device),
            )
        )
        nonnegative = torch.maximum(prepared.start_pos, torch.zeros_like(prepared.start_pos))
        rotary = config.model.rope_setup.get_rot_idxs(nonnegative, on_host=True)
        mapper = ttnn.ShardTensor2dMesh(
            config.mesh_device,
            dims=(None, None),
            mesh_shape=config.cluster_shape,
        )
        positions = ttnn.from_torch(prepared.start_pos, device=None, dtype=ttnn.int32, mesh_mapper=mapper)
        page_table = ttnn.from_torch(prepared.page_table, device=None, dtype=ttnn.int32, mesh_mapper=mapper)
        return DecodeHostInputs(tokens_tt, positions, rotary, page_table)

    def _stage_inputs_and_kpt(self, host_inputs, prepared):
        device_inputs = None
        try:
            raw = _copy_host_to_device(host_inputs.values(), mesh_device=self.config.mesh_device)
            device_inputs = DecodeDeviceInputs(*raw)
            kpt = self._make_device_kpt(prepared)
        except BaseException as primary:
            failures = self._release_or_retain_transient(device_inputs)
            attach_cleanup_failures(primary, failures)
            raise
        return device_inputs, kpt

    def _run_body(
        self,
        inputs,
        prepared,
        kpt,
        *,
        device_feedback,
        count_tokens=True,
        advance_seeds=True,
    ):
        model = self.config.model
        rot_mats = model.rope_setup.get_rot_mats(inputs.rotary_indices)
        logits = model.decode_forward(
            model.embed_decode(inputs.tokens),
            inputs.positions,
            rot_mats,
            page_table=inputs.page_table,
        )
        sampling = prepared.prepared_sampling
        if sampling is None:
            return model.gather_and_untilize_logits(logits), None
        if self._sampling_state_controller is not None:
            sampling = dataclasses.replace(sampling, slot_remap=None)
            output = self._sampling_state_controller.decode_forward(
                logits,
                self._sampling_state,
                sampling,
                k=None if kpt is None else kpt[0],
                p=None if kpt is None else kpt[1],
                temp=None if kpt is None else kpt[2],
                positions=prepared.start_pos,
                tt_out_tok=None,
                count_tokens=count_tokens,
                advance_seeds=advance_seeds,
            )
        else:
            output = self._sample_device(logits, kpt, sampling)
        if device_feedback:
            sampled_tokens = ttnn.reshape(output[0], inputs.tokens.shape)
            ttnn.copy(input_a=sampled_tokens, input_b=inputs.tokens)
            model.increment_positions(inputs.positions, inputs.rotary_indices)
        return output

    def _sample_device(self, logits, kpt, sampling: PreparedSamplingParams):
        if kpt is None:
            return self.config.model.sampling.decode_forward(
                logits,
                tt_out_tok=None,
                enable_log_probs=sampling.enable_log_probs,
            )
        return self.config.model.sampling.decode_forward(
            logits,
            k=kpt[0],
            p=kpt[1],
            temp=kpt[2],
            tt_out_tok=None,
            enable_log_probs=sampling.enable_log_probs,
        )

    def _seed_device_handle(self):
        if self._seed_manager is None:
            return None
        return self._seed_manager.get_seed_device_buffer()

    def _validate_trace_seed_handle(self, persistent: DecodePersistentInputs) -> None:
        if self._seed_manager is None:
            if persistent.seed_buffer is not None:
                raise RuntimeError("decode trace unexpectedly captured a seed buffer")
            return
        current = self._seed_device_handle()
        if persistent.seed_buffer is not current:
            raise RuntimeError("decode trace seed buffer handle changed after capture")

    def _refresh_sampling_seeds(self, prepared: PreparedDecode) -> None:
        if self._sampling_state_controller is not None:
            sampling = prepared.prepared_sampling
            if sampling is None:
                self._sampling_state_controller.seed_manager.restore_defaults(self._sampling_state.seed_state)
                return
            sampling = self._sampling_state_controller.synchronize_decode(
                self._sampling_state,
                sampling,
                reset_batch=prepared.reset_batch,
            )
            self._sampling_state_controller.refresh_dynamic_inputs(
                self._sampling_state,
                sampling,
                positions=prepared.start_pos,
            )
            return
        manager = self._seed_manager
        state = self._seed_state
        sampling = prepared.prepared_sampling
        active_slots = [slot for slot, position in enumerate(prepared.start_pos) if int(position) >= 0]
        if manager is None or state is None:
            if sampling is not None and any(sampling.seeds[slot] is not None for slot in active_slots):
                raise TypeError("explicit request seeds require a native mutable Sampling1D seed buffer")
            return
        if sampling is None:
            manager.cleanup(state, active_slots)
            manager.restore_defaults(state)
            return
        if sampling.slot_remap is not None:
            manager.apply_slot_remap(state, sampling.slot_remap)
        manager.synchronize(
            state,
            sampling.seeds,
            active_slots,
            reset_batch=prepared.reset_batch,
        )
        if prepared.sampling_path == "topk":
            manager.refresh(state, active_slots, positions=prepared.start_pos)
        else:
            manager.restore_defaults(state)

    def _reset_compile_only_sampling_state(self) -> None:
        """Discard synthetic compile admissions and restore seed defaults."""

        if self._sampling_state_controller is not None:
            self._sampling_state_controller.reset(self._sampling_state)
        elif self._seed_manager is not None and self._seed_state is not None:
            self._seed_manager.reset(self._seed_state)

    def _make_device_kpt(self, prepared):
        host = self._make_host_kpt(prepared)
        if host is None:
            return None
        return tuple(_copy_host_to_device(host, mesh_device=self.config.mesh_device))

    def _make_host_kpt(self, prepared):
        sampling = prepared.prepared_sampling
        if sampling is None or prepared.sampling_path == "argmax":
            return None
        k, p, temperature = sampling.top_k, sampling.top_p, sampling.temperature
        mapper = ttnn.ReplicateTensorToMesh(self.config.mesh_device)
        return (
            ttnn.from_torch(
                torch.tensor(k, dtype=torch.int32),
                device=None,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.tensor(p, dtype=torch.float32),
                device=None,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                torch.tensor(temperature, dtype=torch.float32),
                device=None,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=mapper,
            ),
        )

    def _refresh_kpt(self, device_kpt, prepared):
        host_kpt = self._make_host_kpt(prepared)
        if (host_kpt is None) != (device_kpt is None):
            raise RuntimeError("sampling parameters changed the compiled sampling path")
        if host_kpt is not None:
            _copy_host_to_device(host_kpt, device_kpt)

    def _validate_inputs(self, tokens, start_pos, page_table):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1:
            raise ValueError("decode tokens must be a rank-1 torch.Tensor")
        if not isinstance(start_pos, torch.Tensor) or start_pos.ndim != 1:
            raise ValueError("decode start_pos must be a rank-1 torch.Tensor")
        if not isinstance(page_table, torch.Tensor) or page_table.ndim != 2:
            raise ValueError("decode page_table must be a rank-2 torch.Tensor")
        lane_capacity = self.config.lane_capacity
        if int(tokens.shape[0]) != lane_capacity:
            raise ValueError(f"decode batch {tokens.shape[0]} must equal lane capacity {lane_capacity}")
        if int(start_pos.shape[0]) != lane_capacity or int(page_table.shape[0]) != lane_capacity:
            raise ValueError("decode tokens, start_pos, and page_table batches must match")

    def _require_prepared(self, prepared):
        if not isinstance(prepared, PreparedDecode):
            raise TypeError("prepared must be a PreparedDecode")

    def _ensure_usable(self):
        if self._transient_orphans:
            raise RuntimeError("DecodeRuntime has unreleased transient resources; clean up this runtime")

    def _release_external_lease(self, lease):
        if lease is None or lease.released:
            return
        if lease.pending is not None:
            self.config.output_reader.complete(lease.pending)
        failures = []
        if lease.owned_values is not None:
            failures = best_effort_deallocate_owned_tensors(
                (lease.raw_value, lease.owned_values),
                lease.deallocated_tensor_ids,
            )
        if failures:
            raise_cleanup_failures(failures)
        lease.released = True
        self._external_by_raw_id.pop(id(lease.raw_value), None)
        if lease.host_value is not None:
            self._external_by_host_id.pop(id(lease.host_value), None)

    def _release_or_retain_transient(self, values):
        orphan = TensorResourceOrphan(values)
        failures = best_effort_deallocate_owned_tensors(orphan.values, orphan.deallocated_tensor_ids)
        if failures:
            self._transient_orphans.append(orphan)
        return failures


def _persistent_values(value: Any) -> DecodePersistentInputs:
    persistent = getattr(value, "persistent_inputs", value)
    values = getattr(persistent, "values", persistent)
    if isinstance(values, DecodePersistentInputs):
        return values
    if isinstance(values, dict):
        device = values["device_inputs"]
        if not isinstance(device, DecodeDeviceInputs):
            device = DecodeDeviceInputs(*device)
        return DecodePersistentInputs(
            device_inputs=device,
            kpt=values.get("kpt"),
            kpt_signature=values.get("kpt_signature"),
            seed_buffer=values.get("seed_buffer"),
        )
    raise TypeError("decode persistent inputs have an unsupported representation")


def _select_decode_request_state(
    value: Any,
    *,
    active_slots: tuple[int, ...],
    lane_capacity: int,
) -> Any:
    """Convert slot-indexed decode history into active request order."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            raise ValueError("decode sampling history must have a leading request dimension")
        length = int(value.shape[0])
    elif isinstance(value, (list, tuple)):
        length = len(value)
    else:
        raise TypeError("decode sampling history must be a tensor or sequence")
    if length in (1, len(active_slots)):
        return value
    if length < int(lane_capacity):
        raise ValueError(
            f"decode sampling history has {length} rows, expected 1, {len(active_slots)}, "
            f"or at least lane capacity {lane_capacity}"
        )
    rows = (
        torch.tensor(active_slots, dtype=torch.long, device=value.device) if isinstance(value, torch.Tensor) else None
    )
    if rows is not None:
        return value.index_select(0, rows)
    selected = [value[slot] for slot in active_slots]
    return tuple(selected) if isinstance(value, tuple) else selected


def _normalize_decode_slot_remap(
    value: Any,
    *,
    lane_capacity: int,
    sampling_batch_size: int,
) -> Any:
    """Extend a lane-local remap with identity rows for sampler-only padding."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            raise ValueError("decode slot_remap must have a leading slot dimension")
        flat = value.reshape(-1)
        length = int(flat.numel())
    elif isinstance(value, (list, tuple)):
        flat = list(value)
        length = len(flat)
    else:
        raise TypeError("decode slot_remap must be a tensor or sequence")
    if length == int(sampling_batch_size):
        return value
    if length != int(lane_capacity):
        raise ValueError(
            f"decode slot_remap has {length} rows, expected lane capacity {lane_capacity} "
            f"or sampler capacity {sampling_batch_size}"
        )
    sources = [int(source) for source in flat]
    if any(source < 0 or source >= int(lane_capacity) for source in sources):
        raise ValueError("decode slot_remap contains a source outside the lane capacity")
    tail = list(range(int(lane_capacity), int(sampling_batch_size)))
    if isinstance(value, torch.Tensor):
        return torch.cat(
            [
                flat,
                torch.tensor(tail, dtype=value.dtype, device=value.device),
            ]
        )
    extended = sources + tail
    return tuple(extended) if isinstance(value, tuple) else extended


@contextlib.contextmanager
def _validate_module_inputs(model: Any):
    """Instrument one decode forward pass against declared input memory configs."""

    mismatches = []
    originals = []
    for name, module in model.iter_executor_named_modules():
        config = getattr(module, "config", None)
        expected = getattr(config, "decode_input_memcfg", None)
        if expected is None:
            continue
        if not hasattr(module, "decode_forward"):
            raise AttributeError(f"Module {name} has decode_input_memcfg but no decode_forward method")
        original = module.decode_forward
        originals.append((module, original))

        def make_wrapper(orig, module_name, expected_memcfg):
            @functools.wraps(orig)
            def wrapper(x: Any, *args: Any, **kwargs: Any) -> Any:
                if isinstance(x, ttnn.Tensor) and x.is_allocated():
                    actual = x.spec.memory_config
                    if actual != expected_memcfg:
                        mismatches.append((module_name, expected_memcfg, actual))
                return orig(x, *args, **kwargs)

            return wrapper

        module.decode_forward = make_wrapper(original, name, expected)

    try:
        yield
    finally:
        for module, original in originals:
            module.decode_forward = original
        for name, expected, actual in mismatches:
            logger.warning(f"Config mismatch at {name}: declared {expected}, actual {actual}")


def _validate_page_table_layout(layout: Any) -> None:
    if not isinstance(layout, PageTableLayout):
        raise TypeError("page_table_layout must be a PageTableLayout")


def _validate_resolved_decode_config(config: DecodeRuntimeConfig) -> None:
    if not isinstance(config.output_reader, OutputReader):
        raise TypeError("output_reader must be an OutputReader")
    if config.output_reader.mesh_device is not config.mesh_device:
        raise ValueError("output_reader must use the decode mesh_device")
    model_mesh = getattr(getattr(config.model, "config", None), "mesh_device", None)
    if model_mesh is not None and model_mesh is not config.mesh_device:
        raise ValueError("model and decode runtime must use the same mesh_device")
    if (
        not isinstance(config.lane_capacity, int)
        or isinstance(config.lane_capacity, bool)
        or not 0 < config.lane_capacity <= 32
    ):
        raise ValueError("lane_capacity must be an integer from 1 through 32")
    _validate_page_table_layout(config.page_table_layout)
    if (
        not isinstance(config.cluster_shape, tuple)
        or len(config.cluster_shape) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in config.cluster_shape)
    ):
        raise ValueError("cluster_shape must contain two positive integers")
    if tuple(int(value) for value in config.mesh_device.shape) != config.cluster_shape:
        raise ValueError("cluster_shape must match mesh_device.shape")
    if (
        not isinstance(config.num_devices, int)
        or isinstance(config.num_devices, bool)
        or config.num_devices != config.cluster_shape[0] * config.cluster_shape[1]
    ):
        raise ValueError("num_devices must match cluster_shape")
    model_num_devices = getattr(getattr(config.model, "config", None), "num_devices", None)
    if model_num_devices is None:
        model_num_devices = getattr(config.model, "num_devices", config.num_devices)
    if model_num_devices != config.num_devices:
        raise ValueError("num_devices must match the model")
    if not isinstance(config.vocab_size, int) or isinstance(config.vocab_size, bool) or config.vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    if getattr(config.model, "vocab_size", None) != config.vocab_size:
        raise ValueError("vocab_size must match the model")
    for name in (
        "device_sampling_enabled",
        "force_greedy_top_k",
        "allow_force_argmax",
        "position_feedback_capable",
    ):
        if not isinstance(getattr(config, name), bool):
            raise TypeError(f"{name} must be bool")
    sampling = getattr(config.model, "sampling", None)
    sampling_config = getattr(sampling, "config", None)
    expected_argmax = getattr(sampling_config, "allow_force_argmax", None) if config.device_sampling_enabled else False
    if config.device_sampling_enabled:
        if not callable(getattr(sampling, "decode_forward", None)):
            raise TypeError("device sampling requires model.sampling.decode_forward()")
        if not isinstance(expected_argmax, bool):
            raise TypeError("model sampling allow_force_argmax must be bool")
    if config.allow_force_argmax is not expected_argmax:
        raise ValueError("allow_force_argmax must match the resolved model capability")
    expected_top_k = getattr(sampling_config, "max_top_k", 0) if config.device_sampling_enabled else 0
    if config.max_device_top_k != expected_top_k:
        raise ValueError("max_device_top_k must match the resolved model sampler capability")
    expected_sampling_batch_size = (
        getattr(sampling_config, "max_batch_size", config.lane_capacity)
        if config.device_sampling_enabled
        else config.lane_capacity
    )
    if config.sampling_batch_size != expected_sampling_batch_size:
        raise ValueError("sampling_batch_size must match the resolved model sampler capacity")
    if (config.sampling_state_controller is None) != (config.sampling_state is None):
        raise ValueError("sampling_state_controller and sampling_state must be supplied together")
    if config.sampling_state_controller is not None:
        if getattr(config.sampling_state_controller, "sampling", None) is not sampling:
            raise ValueError("sampling state controller must borrow model.sampling")
        if not callable(getattr(config.sampling_state_controller, "decode_forward", None)):
            raise TypeError("sampling state controller must provide decode_forward()")
    if config.position_feedback_capable != callable(getattr(config.model, "increment_positions", None)):
        raise ValueError("position_feedback_capable must match the resolved model capability")
    if not isinstance(config.page_table_layout_ceiling, PageTableLayout):
        raise TypeError("page_table_layout_ceiling must be a PageTableLayout")
    if config.page_table_layout.block_size != config.page_table_layout_ceiling.block_size:
        raise ValueError("page_table_layout_ceiling cannot change block_size")
    if config.page_table_layout.raw_capacity_width > config.page_table_layout_ceiling.raw_capacity_width:
        raise ValueError("page_table_layout_ceiling must cover page_table_layout capacity")
    if config.page_table_layout.decode_width > config.page_table_layout_ceiling.decode_width:
        raise ValueError("page_table_layout_ceiling must cover decode page-table geometry")


def _copy_host_to_device(host_tensors, device_tensors=None, mesh_device=None):
    if device_tensors is None:
        if mesh_device is None:
            raise ValueError("mesh_device is required for device allocation")
        allocated = []
        try:
            for host in host_tensors:
                allocated.append(ttnn.to_device(host, device=mesh_device) if host is not None else None)
        except BaseException as primary:
            failures = best_effort_deallocate_owned_tensors(allocated)
            attach_cleanup_failures(primary, failures)
            raise
        return allocated
    for host, device in zip(host_tensors, device_tensors):
        if host is None:
            if device is not None:
                raise ValueError("host/device optional tensor structure changed")
            continue
        ttnn.copy_host_to_device_tensor(host, device)
    return device_tensors


def _formatted_sampling_values(
    sampling_params,
    batch_size,
    *,
    max_device_top_k=32,
    allow_force_argmax=True,
):
    """Compatibility test helper backed by native exact preparation."""

    formatted_size = ((int(batch_size) + 31) // 32) * 32
    prepared = prepare_sampling_params(
        sampling_params,
        formatted_size,
        max_device_top_k=max_device_top_k,
        allow_force_argmax=allow_force_argmax,
    )
    return (
        prepared.top_k,
        prepared.top_p,
        prepared.temperature,
        prepared.all_active_rows_greedy,
    )


def _concat_host_output(value, cluster_shape):
    if isinstance(value, torch.Tensor):
        return value
    tensors = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(value)]
    rows, columns = cluster_shape
    mesh = [tensors[index : index + columns] for index in range(0, len(tensors), columns)]
    return torch.cat([torch.cat(row, dim=-1) for row in mesh], dim=1)


def _process_output_tokens(value, batch_size, cluster_shape):
    output = _concat_host_output(value, cluster_shape)
    if output.ndim >= 4:
        if int(output.shape[2]) >= batch_size:
            output = output[0, 0, :batch_size, 0]
        elif int(output.shape[3]) >= batch_size:
            output = output[0, 0, 0, :batch_size]
    return output.reshape(-1)[:batch_size].to(torch.int64)


def _process_sampled_log_probs(value, batch_size):
    """Normalize replicated sampled-token logprobs to one row-major tensor."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        output = value
    elif isinstance(value, ttnn.Tensor):
        replicas = ttnn.get_device_tensors(value)
        output = ttnn.to_torch(replicas[0] if replicas else value)
    else:
        # Preserve opaque compatibility payloads used by callers that own their
        # own logprob representation. Native Sampling1D returns a TT tensor.
        return value
    flat = output.reshape(-1)
    if int(flat.numel()) < int(batch_size):
        raise ValueError(f"sampled-token logprobs contain {flat.numel()} rows, expected at least {batch_size}")
    return flat[: int(batch_size)].to(torch.float32)


def _num_blocks(sequence_length, block_size):
    return (int(sequence_length) + int(block_size) - 1) // int(block_size)
