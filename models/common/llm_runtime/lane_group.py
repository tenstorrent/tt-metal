# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel composition for already-built single-lane executors."""

from __future__ import annotations

import copy
import dataclasses
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
from typing import Any, Callable, Iterator, Sequence, TypedDict

import torch

from models.common.llm_runtime.execution import EagerExecutor, TracedExecutor


class _LanePrefillKwargs(TypedDict):
    tokens: torch.Tensor  # ↓ Core request
    page_table: torch.Tensor
    prompt_lens: torch.Tensor | None  # ↓ Sequence metadata
    start_pos: torch.Tensor | None
    empty_slots: Sequence[int] | None  # ↓ Lane routing
    kv_cache: Any  # ↓ Borrowed resources
    sampling_params: Any  # ↓ Sampling
    prompt_tokens: Any  # ↓ Request-owned sampling state
    output_tokens: Any
    slot_remap: Any
    execution: EagerExecutor | TracedExecutor | None  # ↓ Internal dispatch


class _LaneDecodeKwargs(TypedDict):
    tokens: torch.Tensor  # ↓ Core request
    start_pos: torch.Tensor
    page_table: torch.Tensor
    kv_cache: Any  # ↓ Borrowed resources
    sampling_params: Any  # ↓ Sampling
    prompt_tokens: Any  # ↓ Request-owned sampling state
    output_tokens: Any
    slot_remap: Any
    reset_batch: bool  # ↓ State transition
    execution: EagerExecutor | TracedExecutor | None  # ↓ Internal dispatch


class LaneGroupExecutor:
    """Present several fixed-capacity executors as one DP execution target.

    ``Llama3Generator`` calls the same public surface for a single
    ``Llama3Executor`` or this group. Prefill rows are assigned by slot to
    lanes and restored to source order; decode tensors are split into
    contiguous lane batches and aggregated after execution. Warmup, cache
    configuration, and cleanup are replicated across all lanes.
    """

    requires_prefill_trace_warmup = True

    def __init__(self, lanes: Sequence[Any], *, mesh_device: Any = None) -> None:
        self.lanes = list(lanes)
        if not self.lanes:
            raise ValueError("LaneGroupExecutor requires at least one lane")

        self._terminal = False
        self._cleaned_up = False
        self._lane_cleanup_complete = [False] * len(self.lanes)
        self._pool_cleanup_complete = False
        self._cleanup_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._pending_futures: set[Future[Any]] = set()
        self._output_pool: ThreadPoolExecutor | None = None

        try:
            capacities = [int(lane.model.config.max_batch_size) for lane in self.lanes]
            if any(capacity <= 0 for capacity in capacities):
                raise ValueError(f"Lane capacities must be positive, got {capacities}")
            if len(set(capacities)) != 1:
                raise ValueError(f"Every DP lane must have the same fixed capacity, got {capacities}")

            self.tt_data_parallel = len(self.lanes)
            self.per_lane_max_batch_size = capacities[0]
            self.max_batch_size = self.tt_data_parallel * self.per_lane_max_batch_size
            self.executors = self.lanes
            self.model = [lane.model for lane in self.lanes]
            self.model_args = [getattr(lane, "model_args", None) for lane in self.lanes]
            self.mesh_devices = [getattr(lane, "mesh_device", None) for lane in self.lanes]
            self.mesh_device = tuple(self.mesh_devices) if mesh_device is None else mesh_device
            self.eager_execution = tuple(getattr(lane, "eager_execution", None) for lane in self.lanes)
            traced_prefill = tuple(getattr(lane, "traced_prefill_execution", None) for lane in self.lanes)
            traced_decode = tuple(getattr(lane, "traced_decode_execution", None) for lane in self.lanes)
            self.traced_prefill_execution = traced_prefill if all(traced_prefill) else None
            self.traced_decode_execution = traced_decode if all(traced_decode) else None
            self._output_pool = ThreadPoolExecutor(
                max_workers=self.tt_data_parallel,
                thread_name_prefix="tttv2-dp-output",
            )
        except BaseException as primary:
            cleanup_failures = _cleanup_lanes(self.lanes)
            _attach_failures(primary, cleanup_failures, "cleanup_failures")
            raise

    # Public execution-target API

    @property
    def cache_path(self) -> Any:
        return getattr(self.lanes[0], "cache_path", None)

    @property
    def already_warmed_up_prefill(self) -> bool:
        return all(bool(getattr(lane, "already_warmed_up_prefill", False)) for lane in self.lanes)

    @already_warmed_up_prefill.setter
    def already_warmed_up_prefill(self, value: bool) -> None:
        for lane in self.lanes:
            lane.already_warmed_up_prefill = value

    @property
    def paged_kv_cache_config(self) -> tuple[Any, ...]:
        return tuple(getattr(lane, "paged_kv_cache_config") for lane in self.lanes)

    @property
    def terminal(self) -> bool:
        return self._terminal

    def configure_paged_kv_cache(self, config: Any | Sequence[Any]) -> None:
        """Apply one shared or per-lane resolved KV-cache configuration."""

        def operation() -> None:
            configs = self._lane_configs(config)
            for lane, lane_config in zip(self.lanes, configs):
                lane.configure_paged_kv_cache(lane_config)

        self._run_guarded(operation)

    def allocate_kv_cache(
        self,
        kv_cache_shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        num_layers: int | None = None,
    ) -> list[Any]:
        return self._run_guarded(
            lambda: [
                lane.allocate_kv_cache(
                    kv_cache_shape=kv_cache_shape,
                    dtype=dtype,
                    num_layers=num_layers,
                )
                for lane in self.lanes
            ]
        )

    def compile_prefill(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        prompt_tokens: Any = None,  # ↓ Request-owned sampling state
        output_tokens: Any = None,
        slot_remap: Any = None,
        execution: Sequence[EagerExecutor | TracedExecutor] | None = None,  # ↓ Internal dispatch
    ) -> None:
        def operation() -> None:
            lane_requests = self._prefill_lane_requests(
                tokens,
                page_table,
                prompt_lens=prompt_lens,
                start_pos=start_pos,
                empty_slots=empty_slots,
                kv_cache=kv_cache,
                sampling_params=sampling_params,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                slot_remap=slot_remap,
                execution=execution,
            )
            for lane_idx, _, lane_kwargs in lane_requests:
                self.lanes[lane_idx].compile_prefill(**lane_kwargs)

        self._run_guarded(operation)

    def compile_decode(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        prompt_tokens: Any = None,  # ↓ Request-owned sampling state
        output_tokens: Any = None,
        slot_remap: Any = None,
        reset_batch: bool = False,  # ↓ State transition
        execution: Sequence[EagerExecutor | TracedExecutor] | None = None,  # ↓ Internal dispatch
    ) -> None:
        def operation() -> None:
            for lane_idx, lane_kwargs in self._decode_lane_requests(
                tokens,
                start_pos,
                page_table,
                kv_cache=kv_cache,
                sampling_params=sampling_params,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                slot_remap=slot_remap,
                reset_batch=reset_batch,
                execution=execution,
            ):
                self.lanes[lane_idx].compile_decode(**lane_kwargs)

        self._run_guarded(operation)

    def warmup_model_prefill(
        self,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
    ) -> None:
        def operation() -> None:
            for lane_idx, lane in enumerate(self.lanes):
                lane_kv_cache = None if kv_cache is None else self._lane_value(kv_cache, lane_idx, "KV caches")
                lane.warmup_model_prefill(
                    kv_cache=lane_kv_cache,
                    can_sample_on_device=can_sample_on_device,
                    enable_trace=enable_trace,
                )

        self._run_guarded(lambda: self._run_warmup_barrier(operation, enable_trace=enable_trace))

    def warmup_model_decode(
        self,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        max_batch_size: int,  # ↓ Coverage dimensions
        num_blocks: int,
        can_sample_on_device: bool,  # ↓ Execution policy
        enable_trace: bool,
    ) -> None:
        def operation() -> None:
            for lane_idx, lane in enumerate(self.lanes):
                lane_kv_cache = None if kv_cache is None else self._lane_value(kv_cache, lane_idx, "KV caches")
                lane.warmup_model_decode(
                    kv_cache=lane_kv_cache,
                    max_batch_size=self.per_lane_max_batch_size,
                    num_blocks=num_blocks,
                    can_sample_on_device=can_sample_on_device,
                    enable_trace=enable_trace,
                )

        self._run_guarded(lambda: self._run_warmup_barrier(operation, enable_trace=enable_trace))

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        prompt_tokens: Any = None,  # ↓ Request-owned sampling state
        output_tokens: Any = None,
        slot_remap: Any = None,
        execution: Sequence[EagerExecutor | TracedExecutor] | None = None,  # ↓ Internal dispatch
    ) -> Any:
        """Fan out prefill rows by slot and restore their source-row order."""

        def operation() -> Any:
            lane_requests = tuple(
                self._prefill_lane_requests(
                    tokens,
                    page_table,
                    prompt_lens=prompt_lens,
                    start_pos=start_pos,
                    empty_slots=empty_slots,
                    kv_cache=kv_cache,
                    sampling_params=sampling_params,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    slot_remap=slot_remap,
                    execution=execution,
                )
            )
            traced_calls = self._preflight_prefill_traces(lane_requests)
            lane_results = []
            for lane_idx, rows, lane_kwargs in lane_requests:
                traced_call = traced_calls.get(lane_idx)
                if traced_call is None:
                    result = self.lanes[lane_idx].prefill_forward(**lane_kwargs)
                else:
                    selected, preflighted = traced_call
                    result = selected.execute_prepared_prefill(
                        preflighted,
                        batch_size=int(lane_kwargs["tokens"].shape[0]),
                        sampling_params=lane_kwargs["sampling_params"],
                        lane=lane_idx,
                    )
                lane_results.append((rows, result))
            return _aggregate_prefill_outputs(lane_results, int(tokens.shape[0]))

        return self._run_guarded(operation)

    def can_trace_prefill(
        self,
        *,
        tokens: torch.Tensor,  # ↓ Core request
        prompt_lens: torch.Tensor | None = None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,  # ↓ Lane routing
    ) -> bool:
        """Return true only when every participating lane can use prefill trace."""

        def operation() -> bool:
            if not isinstance(tokens, torch.Tensor) or tokens.ndim < 1:
                raise TypeError("prefill tokens must be a torch.Tensor with a batch dimension")
            batch_size = int(tokens.shape[0])
            slots = list(range(batch_size)) if empty_slots is None else list(empty_slots)
            if len(slots) != batch_size:
                raise ValueError(f"empty_slots length {len(slots)} must match prefill batch {batch_size}")

            for lane_idx, rows, _ in self._prefill_lane_groups(slots):
                if not self.lanes[lane_idx].can_trace_prefill(
                    tokens=_slice_rows(tokens, rows),
                    prompt_lens=None if prompt_lens is None else _slice_rows(prompt_lens, rows),
                    start_pos=None if start_pos is None else _slice_rows(start_pos, rows),
                ):
                    return False
            return True

        return self._run_guarded(operation)

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        kv_cache: Any = None,  # ↓ Borrowed resources
        sampling_params: Any = None,  # ↓ Sampling
        prompt_tokens: Any = None,  # ↓ Request-owned sampling state
        output_tokens: Any = None,
        slot_remap: Any = None,
        reset_batch: bool = False,  # ↓ State transition
        read_from_device: bool = True,  # ↓ Output policy
        execution: Sequence[EagerExecutor | TracedExecutor] | None = None,  # ↓ Internal dispatch
    ) -> Any:
        """Split one decode batch into contiguous lane calls and aggregate it."""

        def operation() -> Any:
            lane_outputs = []
            for lane_idx, lane_kwargs in self._decode_lane_requests(
                tokens,
                start_pos,
                page_table,
                kv_cache=kv_cache,
                sampling_params=sampling_params,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                slot_remap=slot_remap,
                reset_batch=reset_batch,
                execution=execution,
            ):
                lane_outputs.append(
                    self.lanes[lane_idx].decode_forward(
                        read_from_device=read_from_device,
                        **lane_kwargs,
                    )
                )
            if not read_from_device:
                return lane_outputs
            return _aggregate_contiguous_outputs(lane_outputs)

        return self._run_guarded(operation)

    def read_decode_output(self, tt_out: Any, *, async_read: bool = False) -> Any:
        def operation() -> Any:
            self._validate_lane_values(tt_out, "decode outputs")
            results = self._run_concurrently(
                lambda lane_idx: self.lanes[lane_idx].read_decode_output(
                    tt_out=tt_out[lane_idx],
                    async_read=async_read,
                )
            )
            if not async_read:
                return results

            host_outputs = []
            events = []
            for lane_result in results:
                if not isinstance(lane_result, tuple) or len(lane_result) != 2:
                    raise TypeError("Async lane read must return (host_output, events)")
                host_output, lane_events = lane_result
                host_outputs.append(host_output)
                if lane_events is None:
                    continue
                if isinstance(lane_events, (list, tuple)):
                    events.extend(lane_events)
                else:
                    events.append(lane_events)
            return host_outputs, events

        return self._run_guarded(operation)

    def process_decode_output_host(self, tt_out: Any, *, is_tokens: bool = False) -> Any:
        def operation() -> Any:
            self._validate_lane_values(tt_out, "host decode outputs")
            results = self._run_concurrently(
                lambda lane_idx: self.lanes[lane_idx].process_decode_output_host(
                    tt_out=tt_out[lane_idx],
                    is_tokens=is_tokens,
                )
            )
            return _aggregate_contiguous_outputs(results, force_tokens=is_tokens)

        return self._run_guarded(operation)

    def cleanup(self) -> None:
        """Drain work and release every lane and worker-pool resource."""

        failures = self._cleanup_impl()
        if failures:
            primary = failures[0]
            _attach_failures(primary, failures[1:], "cleanup_failures")
            raise primary

    # Private implementation

    def _run_warmup_barrier(self, operation: Callable[[], None], *, enable_trace: bool) -> None:
        if not enable_trace:
            operation()
            return
        coordinators = tuple(getattr(lane, "warmup", None) for lane in self.lanes)
        required = ("defer_capture", "activate_pending_capture", "capture_pending", "trace_activated")
        if any(
            coordinator is None or any(not hasattr(coordinator, name) for name in required)
            for coordinator in coordinators
        ):
            raise RuntimeError("Every DP lane must expose the trace activation barrier")

        with ExitStack() as stack:
            for coordinator in coordinators:
                stack.enter_context(coordinator.defer_capture())
            operation()
            states = tuple(
                (
                    "activated"
                    if coordinator.trace_activated
                    else "pending"
                    if coordinator.capture_pending
                    else "incomplete"
                )
                for coordinator in coordinators
            )
            if len(set(states)) != 1:
                raise RuntimeError(f"DP lanes have mixed trace activation readiness: {states}")
            if states[0] == "pending":
                for coordinator in coordinators:
                    coordinator.activate_pending_capture()

    def _preflight_prefill_traces(
        self,
        lane_requests: Sequence[tuple[int, list[int], _LanePrefillKwargs]],
    ) -> dict[int, tuple[TracedExecutor, tuple[tuple[Any, Any], ...]]]:
        """Prepare and preflight every selected DP trace before any KV write."""

        traced_calls = {}
        for lane_idx, _, lane_kwargs in lane_requests:
            selected = lane_kwargs["execution"]
            lane = self.lanes[lane_idx]
            if selected is None or not (
                isinstance(selected, TracedExecutor) or selected is getattr(lane, "traced_prefill_execution", None)
            ):
                continue
            if isinstance(selected, TracedExecutor):
                validate_cache = getattr(lane, "_validate_bound_cache", None)
                if callable(validate_cache):
                    validate_cache(lane_kwargs["kv_cache"])
                ensure_sampling = getattr(lane, "_ensure_sampling_for", None)
                if callable(ensure_sampling):
                    ensure_sampling(lane_kwargs["sampling_params"])
                prepared = selected.prepare_prefill(
                    tokens=lane_kwargs["tokens"],
                    page_table=lane_kwargs["page_table"],
                    prompt_lens=lane_kwargs["prompt_lens"],
                    start_pos=lane_kwargs["start_pos"],
                    empty_slots=lane_kwargs["empty_slots"],
                    sampling_params=lane_kwargs["sampling_params"],
                    prompt_tokens=lane_kwargs.get("prompt_tokens"),
                    output_tokens=lane_kwargs.get("output_tokens"),
                    slot_remap=lane_kwargs.get("slot_remap"),
                )
                try:
                    traced_calls[lane_idx] = (selected, selected.preflight_prefill(prepared))
                except RuntimeError as error:
                    raise RuntimeError(
                        f"Required traced prefill is unavailable for DP lane {lane_idx}; no DP lane executed: {error}"
                    ) from error
                continue
            can_trace = getattr(lane, "can_trace_prefill", None)
            if not callable(can_trace):
                raise RuntimeError(f"DP lane {lane_idx} cannot preflight required traced prefill")
            if not can_trace(
                tokens=lane_kwargs["tokens"],
                prompt_lens=lane_kwargs["prompt_lens"],
                start_pos=lane_kwargs["start_pos"],
            ):
                raise RuntimeError(
                    f"Required traced prefill is unavailable for DP lane {lane_idx}; no DP lane executed"
                )
        return traced_calls

    def _prefill_lane_requests(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        prompt_lens: torch.Tensor | None,  # ↓ Sequence metadata
        start_pos: torch.Tensor | None,
        empty_slots: Sequence[int] | None,  # ↓ Lane routing
        kv_cache: Any,  # ↓ Borrowed resources
        sampling_params: Any,  # ↓ Sampling
        prompt_tokens: Any,  # ↓ Request-owned sampling state
        output_tokens: Any,
        slot_remap: Any,
        execution: Sequence[EagerExecutor | TracedExecutor] | None,  # ↓ Internal dispatch
    ) -> Iterator[tuple[int, list[int], _LanePrefillKwargs]]:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim < 1:
            raise TypeError("prefill tokens must be a torch.Tensor with a batch dimension")
        if not isinstance(page_table, torch.Tensor) or page_table.ndim < 1:
            raise TypeError("prefill page_table must be a torch.Tensor with a batch dimension")
        batch_size = int(tokens.shape[0])
        if int(page_table.shape[0]) != batch_size:
            raise ValueError("prefill tokens and page_table batch sizes must match")

        slots = list(range(batch_size)) if empty_slots is None else list(empty_slots)
        if len(slots) != batch_size:
            raise ValueError(f"empty_slots length {len(slots)} must match prefill batch {batch_size}")

        for lane_idx, rows, local_slots in self._prefill_lane_groups(slots):
            lane_tokens = _slice_rows(tokens, rows)
            lane_page_table = _slice_rows(page_table, rows)
            lane_prompt_lens = None if prompt_lens is None else _slice_rows(prompt_lens, rows)
            lane_start_pos = None if start_pos is None else _slice_rows(start_pos, rows)
            lane_sampling_params = None if sampling_params is None else _slice_sampling_params(sampling_params, rows)
            lane_prompt_tokens = _slice_prefill_request_state(
                prompt_tokens,
                rows=rows,
                global_slots=[int(slots[row]) for row in rows],
                global_capacity=self.max_batch_size,
                local_slots=local_slots,
                lane_capacity=self.per_lane_max_batch_size,
            )
            lane_output_tokens = _slice_prefill_request_state(
                output_tokens,
                rows=rows,
                global_slots=[int(slots[row]) for row in rows],
                global_capacity=self.max_batch_size,
                local_slots=local_slots,
                lane_capacity=self.per_lane_max_batch_size,
            )
            lane_slot_remap = _slice_lane_slot_remap(
                slot_remap,
                lane_idx=lane_idx,
                lane_capacity=self.per_lane_max_batch_size,
                lane_count=self.tt_data_parallel,
            )
            lane_kv_cache = None if kv_cache is None else self._lane_value(kv_cache, lane_idx, "KV caches")
            lane_execution = None if execution is None else self._lane_value(execution, lane_idx, "executions")
            lane_kwargs: _LanePrefillKwargs = {
                "tokens": lane_tokens,
                "page_table": lane_page_table,
                "prompt_lens": lane_prompt_lens,
                "start_pos": lane_start_pos,
                "empty_slots": local_slots,
                "kv_cache": lane_kv_cache,
                "sampling_params": lane_sampling_params,
                "execution": lane_execution,
            }
            if lane_prompt_tokens is not None:
                lane_kwargs["prompt_tokens"] = lane_prompt_tokens
            if lane_output_tokens is not None:
                lane_kwargs["output_tokens"] = lane_output_tokens
            if lane_slot_remap is not None:
                lane_kwargs["slot_remap"] = lane_slot_remap
            yield lane_idx, rows, lane_kwargs

    def _decode_lane_requests(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        sampling_params: Any,  # ↓ Sampling
        prompt_tokens: Any,  # ↓ Request-owned sampling state
        output_tokens: Any,
        slot_remap: Any,
        reset_batch: bool,  # ↓ State transition
        execution: Sequence[EagerExecutor | TracedExecutor] | None,  # ↓ Internal dispatch
    ) -> Iterator[tuple[int, _LaneDecodeKwargs]]:
        for name, value in (("tokens", tokens), ("start_pos", start_pos), ("page_table", page_table)):
            if not isinstance(value, torch.Tensor) or value.ndim < 1:
                raise TypeError(f"decode {name} must be a torch.Tensor with a batch dimension")
            if int(value.shape[0]) != self.max_batch_size:
                raise ValueError(
                    f"DP decode expects fixed global batch {self.max_batch_size}; " f"{name} has batch {value.shape[0]}"
                )

        for lane_idx in range(self.tt_data_parallel):
            start = lane_idx * self.per_lane_max_batch_size
            end = start + self.per_lane_max_batch_size
            lane_tokens = tokens[start:end]
            lane_start_pos = start_pos[start:end]
            lane_page_table = page_table[start:end]
            lane_sampling_params = (
                None if sampling_params is None else _slice_contiguous_sampling_params(sampling_params, start, end)
            )
            lane_prompt_tokens = _slice_optional_contiguous(prompt_tokens, start, end)
            lane_output_tokens = _slice_optional_contiguous(output_tokens, start, end)
            lane_slot_remap = _slice_lane_slot_remap(
                slot_remap,
                lane_idx=lane_idx,
                lane_capacity=self.per_lane_max_batch_size,
                lane_count=self.tt_data_parallel,
            )
            lane_kv_cache = None if kv_cache is None else self._lane_value(kv_cache, lane_idx, "KV caches")
            lane_execution = None if execution is None else self._lane_value(execution, lane_idx, "executions")
            lane_kwargs: _LaneDecodeKwargs = {
                "tokens": lane_tokens,
                "start_pos": lane_start_pos,
                "page_table": lane_page_table,
                "kv_cache": lane_kv_cache,
                "sampling_params": lane_sampling_params,
                "reset_batch": reset_batch,
                "execution": lane_execution,
            }
            if lane_prompt_tokens is not None:
                lane_kwargs["prompt_tokens"] = lane_prompt_tokens
            if lane_output_tokens is not None:
                lane_kwargs["output_tokens"] = lane_output_tokens
            if lane_slot_remap is not None:
                lane_kwargs["slot_remap"] = lane_slot_remap
            yield lane_idx, lane_kwargs

    def _prefill_lane_groups(self, empty_slots: list[Any]):
        groups: list[list[tuple[int, int]]] = [[] for _ in self.lanes]
        for row, slot_value in enumerate(empty_slots):
            slot = int(slot_value)
            lane_idx = slot // self.per_lane_max_batch_size
            if slot < 0 or lane_idx >= self.tt_data_parallel:
                raise ValueError(f"empty slot {slot} maps to an invalid DP lane")
            groups[lane_idx].append((row, slot % self.per_lane_max_batch_size))
        for lane_idx, row_slots in enumerate(groups):
            if row_slots:
                yield lane_idx, [row for row, _ in row_slots], [slot for _, slot in row_slots]

    def _lane_configs(self, config: Any | Sequence[Any]) -> list[Any]:
        if isinstance(config, (list, tuple)):
            self._validate_lane_values(config, "paged KV configs")
            return list(config)
        return [_clone_config(config) for _ in self.lanes]

    def _lane_value(self, values: Sequence[Any], lane_idx: int, label: str) -> Any:
        self._validate_lane_values(values, label)
        return values[lane_idx]

    def _validate_lane_values(self, values: Sequence[Any], label: str) -> None:
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise TypeError(f"{label} must be a sequence with one value per lane")
        if len(values) != self.tt_data_parallel:
            raise ValueError(f"{label} has {len(values)} entries for {self.tt_data_parallel} lanes")

    def _run_guarded(self, operation: Callable[[], Any]) -> Any:
        self._ensure_active()
        try:
            return operation()
        except BaseException as primary:
            cleanup_failures = self._cleanup_impl()
            _attach_failures(primary, cleanup_failures, "cleanup_failures")
            raise

    def _run_concurrently(self, operation: Callable[[int], Any]) -> list[Any]:
        assert self._output_pool is not None
        with self._pending_lock:
            self._ensure_active()
            futures = [self._output_pool.submit(operation, lane_idx) for lane_idx in range(self.tt_data_parallel)]
            self._pending_futures.update(futures)
        wait(futures)

        results = []
        primary = None
        secondary_failures = []
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    secondary_failures.append(error)
            finally:
                with self._pending_lock:
                    self._pending_futures.discard(future)
        if primary is not None:
            _attach_failures(primary, secondary_failures, "lane_failures")
            raise primary
        return results

    def _ensure_active(self) -> None:
        if self._terminal:
            raise RuntimeError("LaneGroupExecutor is terminal; construct a new group")

    def _cleanup_impl(self) -> list[BaseException]:
        with self._cleanup_lock:
            self._terminal = True
            if self._cleaned_up:
                return []
            failures: list[BaseException] = []
            with self._pending_lock:
                pending = list(self._pending_futures)
            if pending:
                wait(pending)
                for future in pending:
                    try:
                        future.result()
                    except BaseException as error:
                        failures.append(error)
                with self._pending_lock:
                    self._pending_futures.difference_update(pending)

            for lane_idx, lane in enumerate(self.lanes):
                if self._lane_cleanup_complete[lane_idx]:
                    continue
                try:
                    lane.cleanup()
                except BaseException as error:
                    failures.append(error)
                else:
                    self._lane_cleanup_complete[lane_idx] = True

            if not self._pool_cleanup_complete:
                if self._output_pool is None:
                    self._pool_cleanup_complete = True
                else:
                    try:
                        self._output_pool.shutdown(wait=True)
                    except BaseException as error:
                        failures.append(error)
                    else:
                        self._pool_cleanup_complete = True

            self._cleaned_up = all(self._lane_cleanup_complete) and self._pool_cleanup_complete
            return failures


def _slice_prefill_request_state(
    value: Any,
    *,
    rows: list[int],
    global_slots: list[int],
    global_capacity: int,
    local_slots: list[int],
    lane_capacity: int,
) -> Any:
    if value is None:
        return None
    length = _row_scoped_length(value)
    if length == 1:
        selected = _slice_rows(value, [0] * len(rows))
    elif length == global_capacity:
        selected = _slice_rows(value, global_slots)
    else:
        selected = _slice_rows(value, rows)
    return _place_prefill_request_state(
        selected,
        local_slots=local_slots,
        lane_capacity=lane_capacity,
    )


def _place_prefill_request_state(
    value: Any,
    *,
    local_slots: list[int],
    lane_capacity: int,
) -> Any:
    if len(local_slots) != _row_scoped_length(value):
        raise ValueError("prefill sampling state row count must match destination slots")
    if isinstance(value, torch.Tensor):
        fill = False if value.dtype == torch.bool else -1
        placed = torch.full(
            (int(lane_capacity), *value.shape[1:]),
            fill,
            dtype=value.dtype,
            device=value.device,
        )
        indices = torch.tensor(local_slots, dtype=torch.long, device=value.device)
        placed.index_copy_(0, indices, value)
        return placed
    values = list(value)
    exemplar = values[0]
    if isinstance(exemplar, tuple):
        inactive = tuple(-1 for _ in exemplar)
    elif isinstance(exemplar, list):
        inactive = [-1 for _ in exemplar]
    else:
        inactive = -1
    placed = [inactive for _ in range(int(lane_capacity))]
    for row, slot in enumerate(local_slots):
        placed[int(slot)] = values[row]
    return tuple(placed) if isinstance(value, tuple) else placed


def _slice_optional_contiguous(value: Any, start: int, end: int) -> Any:
    if value is None:
        return None
    length = _row_scoped_length(value)
    count = end - start
    if length == 1:
        return _slice_rows(value, [0] * count)
    if length < end:
        raise ValueError(f"row-scoped sampling state has {length} rows, expected at least {end}")
    return _slice_rows(value, list(range(start, end)))


def _slice_lane_slot_remap(
    value: Any,
    *,
    lane_idx: int,
    lane_capacity: int,
    lane_count: int,
) -> Any:
    if value is None:
        return None
    length = _row_scoped_length(value)
    global_capacity = lane_capacity * lane_count
    lane_start = lane_idx * lane_capacity
    lane_end = lane_start + lane_capacity
    if length == lane_capacity:
        local = _slice_rows(value, list(range(lane_capacity)))
        sources = local.reshape(-1).tolist() if isinstance(local, torch.Tensor) else list(local)
        if any(int(source) < 0 or int(source) >= lane_capacity for source in sources):
            raise ValueError("lane-local slot_remap contains an invalid source slot")
        return local
    if length != global_capacity:
        raise ValueError(f"slot_remap has {length} rows, expected {lane_capacity} or {global_capacity}")
    selected = _slice_rows(value, list(range(lane_start, lane_end)))
    sources = selected.reshape(-1).tolist() if isinstance(selected, torch.Tensor) else list(selected)
    if any(int(source) < lane_start or int(source) >= lane_end for source in sources):
        raise ValueError("slot_remap cannot move request-owned sampling state across DP lanes")
    if isinstance(selected, torch.Tensor):
        return selected - lane_start
    if isinstance(selected, tuple):
        return tuple(int(source) - lane_start for source in sources)
    return [int(source) - lane_start for source in sources]


def _row_scoped_length(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return 1
        return int(value.shape[0])
    if isinstance(value, (list, tuple)):
        return len(value)
    raise TypeError(f"Cannot slice row-scoped value of type {type(value).__name__}")


def _slice_rows(value: Any, rows: list[int]) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            if any(row != 0 for row in rows):
                raise ValueError("cannot select multiple distinct rows from a scalar tensor")
            return value.expand(len(rows))
        indices = torch.tensor(rows, dtype=torch.long, device=value.device)
        return value.index_select(0, indices)
    if isinstance(value, list):
        return [value[row] for row in rows]
    if isinstance(value, tuple):
        return tuple(value[row] for row in rows)
    raise TypeError(f"Cannot slice row-scoped value of type {type(value).__name__}")


def _slice_sampling_params(sampling_params: Any, rows: list[int]) -> Any:
    def slice_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value
            selected_rows = [0] * len(rows) if int(value.shape[0]) == 1 else rows
            return _slice_rows(value, selected_rows)
        if isinstance(value, list):
            if len(value) == 1:
                return [value[0] for _ in rows]
            return [value[row] for row in rows]
        if isinstance(value, tuple):
            if len(value) == 1:
                return tuple(value[0] for _ in rows)
            return tuple(value[row] for row in rows)
        return value

    if dataclasses.is_dataclass(sampling_params) and not isinstance(sampling_params, type):
        updates = {
            field.name: slice_value(getattr(sampling_params, field.name))
            for field in dataclasses.fields(sampling_params)
        }
        return dataclasses.replace(sampling_params, **updates)
    if isinstance(sampling_params, dict):
        return sampling_params.__class__((key, slice_value(value)) for key, value in sampling_params.items())
    raise TypeError("sampling_params must be a dataclass or mapping")


def _slice_contiguous_sampling_params(sampling_params: Any, start: int, end: int) -> Any:
    count = end - start

    def slice_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return value
            if int(value.shape[0]) == 1:
                return value.expand((count, *value.shape[1:]))
            return value[start:end]
        if isinstance(value, list):
            if len(value) == 1:
                return value * count
            return value[start:end]
        if isinstance(value, tuple):
            if len(value) == 1:
                return value * count
            return value[start:end]
        return value

    if dataclasses.is_dataclass(sampling_params) and not isinstance(sampling_params, type):
        updates = {
            field.name: slice_value(getattr(sampling_params, field.name))
            for field in dataclasses.fields(sampling_params)
        }
        return dataclasses.replace(sampling_params, **updates)
    if isinstance(sampling_params, dict):
        return sampling_params.__class__((key, slice_value(value)) for key, value in sampling_params.items())
    raise TypeError("sampling_params must be a dataclass or mapping")


def _aggregate_prefill_outputs(lane_results: list[tuple[list[int], Any]], batch_size: int) -> Any:
    if not lane_results:
        return torch.empty((0,), dtype=torch.int64)
    unwrapped = []
    lane_log_probs = []
    had_tuple = False
    for rows, result in lane_results:
        output, log_probs, was_tuple = _unwrap_output(result)
        had_tuple = had_tuple or was_tuple
        unwrapped.append((rows, output))
        lane_log_probs.append((rows, log_probs))
    if all(output is None for _, output in unwrapped):
        return (None, _aggregate_log_probs_by_rows(lane_log_probs, batch_size)) if had_tuple else None
    if not all(isinstance(output, torch.Tensor) for _, output in unwrapped):
        output = [output for _, output in unwrapped]
        return (output, _aggregate_log_probs_by_rows(lane_log_probs, batch_size)) if had_tuple else output

    first = unwrapped[0][1]
    assert isinstance(first, torch.Tensor)
    if _is_token_tensor(first):
        output = torch.empty((batch_size,), dtype=torch.int64, device=first.device)
        for rows, lane_output in unwrapped:
            output[rows] = lane_output.reshape(-1).to(torch.int64)
    else:
        output = torch.empty((batch_size, *first.shape[1:]), dtype=first.dtype, device=first.device)
        for rows, lane_output in unwrapped:
            output[rows] = lane_output
    return (output, _aggregate_log_probs_by_rows(lane_log_probs, batch_size)) if had_tuple else output


def _aggregate_contiguous_outputs(lane_results: list[Any], *, force_tokens: bool = False) -> Any:
    unwrapped = []
    lane_log_probs = []
    had_tuple = False
    for result in lane_results:
        output, log_probs, was_tuple = _unwrap_output(result)
        had_tuple = had_tuple or was_tuple
        unwrapped.append(output)
        lane_log_probs.append(log_probs)
    if all(output is None for output in unwrapped):
        return (None, _aggregate_log_probs_contiguous(lane_log_probs, unwrapped)) if had_tuple else None
    if not all(isinstance(output, torch.Tensor) for output in unwrapped):
        return (unwrapped, _aggregate_log_probs_contiguous(lane_log_probs, unwrapped)) if had_tuple else unwrapped

    first = unwrapped[0]
    assert isinstance(first, torch.Tensor)
    if force_tokens or _is_token_tensor(first):
        output = torch.cat([lane_output.reshape(-1) for lane_output in unwrapped], dim=0).to(torch.int64)
    else:
        output = torch.cat(unwrapped, dim=0)
    return (output, _aggregate_log_probs_contiguous(lane_log_probs, unwrapped)) if had_tuple else output


def _aggregate_log_probs_contiguous(values: list[Any], lane_outputs: list[Any]) -> Any:
    if all(value is None for value in values):
        return None
    merged = []
    for value, lane_output in zip(values, lane_outputs):
        if not isinstance(lane_output, torch.Tensor) or lane_output.ndim == 0:
            raise TypeError("lane output must expose its row count when logprobs are present")
        lane_rows = int(lane_output.shape[0])
        if value is None:
            merged.append(torch.ones(lane_rows, dtype=torch.float32))
        else:
            merged.append(_sampled_log_probs_for_rows(value, lane_rows))
    return torch.cat(merged, dim=0)


def _aggregate_log_probs_by_rows(values: list[tuple[list[int], Any]], batch_size: int) -> Any:
    if all(value is None for _, value in values):
        return None
    ordered = torch.ones(int(batch_size), dtype=torch.float32)
    for rows, value in values:
        if value is None:
            continue
        payload = _sampled_log_probs_for_rows(value, len(rows))
        indices = torch.tensor(tuple(int(row) for row in rows), dtype=torch.long)
        ordered.index_copy_(0, indices, payload)
    return ordered


def _sampled_log_probs_for_rows(value: Any, row_count: int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        output = value.reshape(-1)
    elif isinstance(value, (float, int)):
        output = torch.full((int(row_count),), float(value), dtype=torch.float32)
    elif isinstance(value, (list, tuple)):
        output = torch.as_tensor(value).reshape(-1)
    else:
        raise TypeError("lane logprobs must be a Torch tensor or numeric sequence")
    if int(output.numel()) == 1 and int(row_count) > 1:
        output = output.expand(int(row_count))
    if int(output.numel()) < int(row_count):
        raise ValueError(f"lane logprobs contain {output.numel()} rows, expected at least {row_count}")
    return output[: int(row_count)].to(torch.float32)


def _unwrap_output(result: Any) -> tuple[Any, Any, bool]:
    if isinstance(result, tuple):
        if len(result) != 2:
            raise TypeError("Lane output tuple must contain (output, log_probs)")
        return result[0], result[1], True
    return result, None, False


def _is_token_tensor(tensor: torch.Tensor) -> bool:
    return not tensor.is_floating_point() and not tensor.is_complex()


def _clone_config(config: Any) -> Any:
    return dataclasses.replace(config) if dataclasses.is_dataclass(config) else copy.copy(config)


def _cleanup_lanes(lanes: Sequence[Any]) -> list[BaseException]:
    failures = []
    for lane in lanes:
        try:
            lane.cleanup()
        except BaseException as error:
            failures.append(error)
    return failures


def _attach_failures(primary: BaseException, failures: Sequence[BaseException], attribute: str) -> None:
    if not failures:
        return
    previous = tuple(getattr(primary, attribute, ()))
    try:
        setattr(primary, attribute, previous + tuple(failures))
    except BaseException:
        pass
    add_note = getattr(primary, "add_note", None)
    if add_note is not None:
        for failure in failures:
            add_note(f"{attribute}: {type(failure).__name__}: {failure}")
