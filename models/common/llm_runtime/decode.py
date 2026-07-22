# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Decode preparation, invocation, feedback, readback, and local resources."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch

import ttnn
from models.common.llm_runtime.module_input_validation import (
    suspend_module_input_validation,
    validate_module_input_configs,
)
from models.common.llm_runtime.output_reader import OutputReader, PendingRead
from models.common.llm_runtime.tensor_resources import (
    TensorResourceOrphan,
    attach_cleanup_failures,
    best_effort_deallocate_owned_tensors,
    raise_cleanup_failures,
    release_orphans,
)
from models.common.sampling import SamplingParams, format_sampling_params


@dataclass(frozen=True)
class DecodeProgramSignature:
    """Material identity of one decode eager-program variant."""

    batch_size: int
    page_table_width: int
    sampling_path: str
    device_feedback: bool

    def key_material(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("operation", "decode"),
            ("batch_size", self.batch_size),
            ("page_table_width", self.page_table_width),
            ("sampling_path", self.sampling_path),
            ("device_feedback", self.device_feedback),
        )


@dataclass(frozen=True)
class DecodeTraceSignature:
    """Material identity of one full-step decode trace."""

    batch_size: int
    page_table_width: int
    sampling_path: str
    device_feedback: bool

    def key_material(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("operation", "decode"),
            ("batch_size", self.batch_size),
            ("page_table_width", self.page_table_width),
            ("sampling_path", self.sampling_path),
            ("device_feedback", self.device_feedback),
        )


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
    sampling_values: tuple[tuple[int, ...], tuple[float, ...], tuple[float, ...], bool] | None
    sampling_path: str
    reset_batch: bool
    device_feedback: bool
    page_table_changed: bool


@dataclass(frozen=True)
class InvocationResult:
    value: Any
    owned: Any
    is_tokens: bool


@dataclass(frozen=True)
class DecodeRefreshIntent:
    full: bool
    page_table: bool
    reason: str | None = None


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


class DecodeRuntime:
    """Own the complete decode vertical slice for one Llama execution lane.

    The model, mesh, output reader, sampler, and KV-backed page-table values are
    borrowed. Only staged invocation tensors, raw outputs, output leases, and
    retryable decode transients are released here.
    """

    def __init__(
        self,
        model: Any,
        mesh_device: Any,
        output_reader: OutputReader,
        *,
        lane_capacity: int,
        page_table_layout: Any,
        device_sampling_enabled: bool,
        force_greedy_top_k: bool = False,
    ):
        if not isinstance(output_reader, OutputReader):
            raise TypeError("output_reader must be an OutputReader")
        if int(lane_capacity) <= 0:
            raise ValueError("lane_capacity must be positive")
        if int(lane_capacity) > 32:
            raise ValueError("decode token input padding supports at most 32 lane slots")
        self.model = model
        self.mesh_device = mesh_device
        self.output_reader = output_reader
        self.lane_capacity = int(lane_capacity)
        self.page_table_layout = page_table_layout
        self.device_sampling_enabled = bool(device_sampling_enabled)
        self.force_greedy_top_k = bool(force_greedy_top_k)
        self.device_feedback_enabled = True
        self._previous_page_table: torch.Tensor | None = None
        self._normalization_source: torch.Tensor | None = None
        self._normalization_copy_blocks: tuple[int, ...] | None = None
        self._normalization_layout: tuple[int, int, int] | None = None
        self._normalized_page_table: torch.Tensor | None = None
        self._external_by_raw_id: dict[int, DecodeOutputLease] = {}
        self._external_by_host_id: dict[int, DecodeOutputLease] = {}
        self._transient_orphans: list[TensorResourceOrphan] = []

    @property
    def cluster_shape(self) -> list[int]:
        return list(self.mesh_device.shape)

    @property
    def previous_page_table(self) -> torch.Tensor | None:
        value = self._previous_page_table
        return None if value is None else value.clone()

    @property
    def external_lease_count(self) -> int:
        return len(self._external_by_raw_id)

    @property
    def transient_orphan_count(self) -> int:
        return len(self._transient_orphans)

    def prepare(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        sampling_params: SamplingParams | None = None,
        *,
        reset_batch: bool = False,
    ) -> PreparedDecode:
        self._ensure_usable()
        self._validate_inputs(tokens, start_pos, page_table)
        if sampling_params is not None and not self.device_sampling_enabled:
            raise ValueError("sampling parameters were supplied while device sampling is disabled")
        feedback = self._use_device_feedback(sampling_params)
        sampling_values = (
            None if sampling_params is None else _formatted_sampling_values(sampling_params, self.lane_capacity)
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
            sampling_values=sampling_values,
            sampling_path=self._sampling_path(sampling_values),
            reset_batch=bool(reset_batch),
            device_feedback=feedback,
            page_table_changed=(
                self._previous_page_table is None or not torch.equal(self._previous_page_table, normalized)
            ),
        )

    def program_signature(self, prepared: PreparedDecode) -> DecodeProgramSignature:
        self._require_prepared(prepared)
        return DecodeProgramSignature(
            batch_size=self.lane_capacity,
            page_table_width=int(prepared.page_table.shape[-1]),
            sampling_path=prepared.sampling_path,
            device_feedback=prepared.device_feedback,
        )

    def trace_signature(self, prepared: PreparedDecode) -> DecodeTraceSignature:
        program = self.program_signature(prepared)
        return DecodeTraceSignature(
            batch_size=program.batch_size,
            page_table_width=program.page_table_width,
            sampling_path=program.sampling_path,
            device_feedback=program.device_feedback,
        )

    def invoke(self, prepared: PreparedDecode, *, device_feedback: bool = False) -> InvocationResult:
        self._ensure_usable()
        self._require_prepared(prepared)
        host_inputs = self._prepare_inputs_host(prepared)
        device_inputs, kpt = self._stage_inputs_and_kpt(host_inputs, prepared)
        owned = (device_inputs, kpt)
        try:
            with self._validation_context():
                output = self._run_body(
                    device_inputs,
                    prepared.sampling_params,
                    kpt,
                    device_feedback=device_feedback and prepared.device_feedback,
                )
        except BaseException as primary:
            failures = self._release_or_retain_transient(owned)
            attach_cleanup_failures(primary, failures)
            raise
        self.note_submitted(prepared)
        return InvocationResult(
            value=output,
            owned=(output, owned),
            is_tokens=prepared.sampling_params is not None,
        )

    def capture_plan(self, prepared: PreparedDecode) -> DecodeCapturePlan:
        self._require_prepared(prepared)

        def prepare_inputs() -> DecodePersistentInputs:
            host_inputs = self._prepare_inputs_host(prepared)
            device_inputs, kpt = self._stage_inputs_and_kpt(host_inputs, prepared)
            signature = [prepared.sampling_values[:3]] if kpt is not None else None
            return DecodePersistentInputs(device_inputs=device_inputs, kpt=kpt, kpt_signature=signature)

        def capture(persistent: Any) -> Any:
            values = _persistent_values(persistent)
            with suspend_module_input_validation():
                return self._run_body(
                    values.device_inputs,
                    prepared.sampling_params,
                    values.kpt,
                    device_feedback=prepared.device_feedback,
                )

        return DecodeCapturePlan(prepare_inputs=prepare_inputs, capture=capture)

    def refresh_trace(
        self,
        artifact: Any,
        prepared: PreparedDecode,
        decision: DecodeRefreshIntent | Any,
    ) -> None:
        self._require_prepared(prepared)
        values = _persistent_values(artifact)
        if bool(decision.full):
            host_inputs = self._prepare_inputs_host(prepared)
            _copy_host_to_device(host_inputs.values(), values.device_inputs.values())
        elif bool(decision.page_table):
            host_inputs = self._prepare_inputs_host(prepared)
            ttnn.copy_host_to_device_tensor(host_inputs.page_table, values.device_inputs.page_table)
        if prepared.sampling_path == "topk":
            signature = prepared.sampling_values[:3]
            if values.kpt_signature is None or values.kpt_signature[0] != signature:
                self._refresh_kpt(values.kpt, prepared)
                if values.kpt_signature is not None:
                    values.kpt_signature[0] = signature
        elif values.kpt is not None:
            raise RuntimeError("non-top-k decode trace unexpectedly owns sampling inputs")

    def note_submitted(self, prepared: PreparedDecode) -> None:
        """Advance feedback comparison state immediately after device submission."""
        self._require_prepared(prepared)
        self._previous_page_table = prepared.page_table.clone()

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
            host = self.output_reader.read(result.value, blocking=True)
            normalized = self.normalize_host_output(
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

    def read_decode_output(self, tt_out: Any, async_read: bool = False) -> Any:
        if not async_read:
            host = self.output_reader.read(tt_out, blocking=True)
            self._release_external_lease(self._external_by_raw_id.get(id(tt_out)))
            return host
        pending = self.output_reader.submit(tt_out)
        lease = self._external_by_raw_id.get(id(tt_out))
        if lease is not None:
            lease.host_value = pending.value
            lease.pending = pending
            self._external_by_host_id[id(pending.value)] = lease
        return pending.value, list(pending.events)

    def process_decode_output_host(self, tt_out: Any, is_tokens: bool = False) -> tuple[Any, Any]:
        completed = self.output_reader.complete(tt_out)
        self._release_external_lease(self._external_by_host_id.get(id(tt_out)))
        return self.normalize_host_output(completed, is_tokens=is_tokens)

    def normalize_host_output(self, host_output: Any, *, is_tokens: bool) -> tuple[Any, Any]:
        output, log_probs = _split_output(host_output)
        if is_tokens:
            tokens = _process_output_tokens(output, self.lane_capacity, self.cluster_shape)
            return tokens.to(torch.int64), log_probs
        logits = _process_output_decode_logits(
            output,
            self.lane_capacity,
            int(self.model.vocab_size),
            int(self.model.num_devices),
            self.cluster_shape,
        )
        return logits, log_probs

    def drain_external_outputs(self) -> None:
        failures = []
        for lease in tuple(self._external_by_raw_id.values()):
            try:
                if lease.pending is None:
                    ttnn.synchronize_device(self.mesh_device)
                self._release_external_lease(lease)
            except BaseException as error:
                failures.append(error)
        if failures:
            raise_cleanup_failures(failures)

    def cleanup_transients(self) -> None:
        failures = release_orphans(self._transient_orphans)
        if failures:
            raise_cleanup_failures(failures)

    def cleanup(self) -> None:
        failures = []
        for action in (self.drain_external_outputs, self.cleanup_transients):
            try:
                action()
            except BaseException as error:
                failures.append(error)
        if failures:
            raise_cleanup_failures(failures)

    def _normalize_page_table(self, page_table, start_pos, *, allow_one_step_feedback_lag):
        raw_width = _layout_value(self.page_table_layout, "raw_capacity_width", "raw_width")
        decode_width = _layout_value(self.page_table_layout, "decode_width", "canonical_decode_width")
        block_size = _layout_value(self.page_table_layout, "block_size")
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
        padded = torch.nn.functional.pad(prepared.tokens.reshape(-1), (0, 32 - self.lane_capacity))
        tokens_tt = ttnn.unsqueeze_to_4D(
            ttnn.from_torch(
                padded,
                device=None,
                dtype=ttnn.uint32,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        )
        nonnegative = torch.maximum(prepared.start_pos, torch.zeros_like(prepared.start_pos))
        rotary = self.model.rope_setup.get_rot_idxs(nonnegative, on_host=True)
        mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, None), mesh_shape=self.cluster_shape)
        positions = ttnn.from_torch(prepared.start_pos, device=None, dtype=ttnn.int32, mesh_mapper=mapper)
        page_table = ttnn.from_torch(prepared.page_table, device=None, dtype=ttnn.int32, mesh_mapper=mapper)
        return DecodeHostInputs(tokens_tt, positions, rotary, page_table)

    def _stage_inputs_and_kpt(self, host_inputs, prepared):
        device_inputs = None
        try:
            raw = _copy_host_to_device(host_inputs.values(), mesh_device=self.mesh_device)
            device_inputs = DecodeDeviceInputs(*raw)
            kpt = self._make_device_kpt(prepared)
        except BaseException as primary:
            failures = self._release_or_retain_transient(device_inputs)
            attach_cleanup_failures(primary, failures)
            raise
        return device_inputs, kpt

    def _run_body(self, inputs, sampling_params, kpt, *, device_feedback):
        rot_mats = self.model.rope_setup.get_rot_mats(inputs.rotary_indices)
        logits = self.model.decode_forward(
            self.model.embed_decode(inputs.tokens),
            inputs.positions,
            rot_mats,
            page_table=inputs.page_table,
        )
        if sampling_params is None:
            return self.model.gather_and_untilize_logits(logits), None
        output = self._sample_device(logits, kpt)
        if device_feedback:
            sampled_tokens = ttnn.reshape(output[0], inputs.tokens.shape)
            ttnn.copy(input_a=sampled_tokens, input_b=inputs.tokens)
            self.model.increment_positions(inputs.positions, inputs.rotary_indices)
        return output

    def _sample_device(self, logits, kpt):
        if kpt is None:
            return self.model.sampling.decode_forward(logits, tt_out_tok=None)
        return self.model.sampling.decode_forward(logits, k=kpt[0], p=kpt[1], temp=kpt[2], tt_out_tok=None)

    def _make_device_kpt(self, prepared):
        host = self._make_host_kpt(prepared)
        if host is None:
            return None
        return tuple(_copy_host_to_device(host, mesh_device=self.mesh_device))

    def _make_host_kpt(self, prepared):
        if prepared.sampling_values is None:
            return None
        k, p, temperature, greedy = prepared.sampling_values
        if bool(self.model.sampling.config.allow_force_argmax) and greedy and not self.force_greedy_top_k:
            return None
        mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
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

    def _sampling_path(self, sampling_values):
        if sampling_values is None:
            return "logits"
        if bool(self.model.sampling.config.allow_force_argmax) and sampling_values[3] and not self.force_greedy_top_k:
            return "argmax"
        return "topk"

    def _use_device_feedback(self, sampling_params):
        return (
            self.device_feedback_enabled and sampling_params is not None and hasattr(self.model, "increment_positions")
        )

    def _validation_context(self):
        return validate_module_input_configs(
            model=self.model,
            iter_named_modules=lambda model: model.iter_executor_named_modules(),
            mode="decode",
        )

    def _validate_inputs(self, tokens, start_pos, page_table):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1:
            raise ValueError("decode tokens must be a rank-1 torch.Tensor")
        if not isinstance(start_pos, torch.Tensor) or start_pos.ndim != 1:
            raise ValueError("decode start_pos must be a rank-1 torch.Tensor")
        if not isinstance(page_table, torch.Tensor) or page_table.ndim != 2:
            raise ValueError("decode page_table must be a rank-2 torch.Tensor")
        if int(tokens.shape[0]) != self.lane_capacity:
            raise ValueError(f"decode batch {tokens.shape[0]} must equal lane capacity {self.lane_capacity}")
        if int(start_pos.shape[0]) != self.lane_capacity or int(page_table.shape[0]) != self.lane_capacity:
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
            self.output_reader.complete(lease.pending)
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
        )
    raise TypeError("decode persistent inputs have an unsupported representation")


def _layout_value(layout: Any, *names: str) -> int:
    for name in names:
        value = getattr(layout, name, None)
        if value is not None:
            value = int(value)
            if value <= 0:
                raise ValueError(f"page-table layout {name} must be positive")
            return value
    raise TypeError(f"page_table_layout must provide one of {names!r}")


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


def _formatted_sampling_values(sampling_params, batch_size):
    sampling_params = _tensor_sampling_fields_to_python(sampling_params)
    formatted_size = ((int(batch_size) + 31) // 32) * 32
    formatted = format_sampling_params(sampling_params, formatted_size)
    k = tuple(int(value) for value in formatted.top_k)
    p = tuple(float(value) for value in formatted.top_p)
    temperature = tuple(float(value) for value in formatted.temperature)
    greedy = (
        all(value == 1 for value in k) and all(value == 0 for value in p) and all(value == 1 for value in temperature)
    )
    return k, p, temperature, greedy


def _tensor_sampling_fields_to_python(sampling_params):
    updates = {}
    for field in dataclasses.fields(sampling_params):
        value = getattr(sampling_params, field.name)
        if isinstance(value, torch.Tensor):
            updates[field.name] = value.item() if value.ndim == 0 else value.tolist()
    return dataclasses.replace(sampling_params, **updates) if updates else sampling_params


def _split_output(value):
    if isinstance(value, tuple):
        if len(value) != 2:
            raise TypeError("runtime output tuple must contain (output, log_probs)")
        return value
    return value, None


def _concat_host_output(value, cluster_shape):
    if isinstance(value, torch.Tensor):
        return value
    tensors = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(value)]
    rows, columns = cluster_shape
    mesh = [tensors[index : index + columns] for index in range(0, len(tensors), columns)]
    return torch.cat([torch.cat(row, dim=-1) for row in mesh], dim=1)


def _process_output_decode_logits(value, batch_size, vocab_size, num_devices, cluster_shape):
    if isinstance(value, torch.Tensor):
        output = value.float()
    elif num_devices > 1:
        output = _concat_host_output(value, cluster_shape).float()
    else:
        output = ttnn.to_torch(value).float()
    return output[:, :, :batch_size, :vocab_size].contiguous().view(batch_size, 1, -1)


def _process_output_tokens(value, batch_size, cluster_shape):
    output = _concat_host_output(value, cluster_shape)
    if output.ndim >= 4:
        if int(output.shape[2]) >= batch_size:
            output = output[0, 0, :batch_size, 0]
        elif int(output.shape[3]) >= batch_size:
            output = output[0, 0, 0, :batch_size]
    return output.reshape(-1)[:batch_size].to(torch.int64)


def _num_blocks(sequence_length, block_size):
    return (int(sequence_length) + int(block_size) - 1) // int(block_size)
