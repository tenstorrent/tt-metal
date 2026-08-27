# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel composition for already-built single-lane executors."""

from __future__ import annotations

import copy
import dataclasses
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
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
    execution: EagerExecutor | TracedExecutor | None  # ↓ Internal dispatch


class _LaneDecodeKwargs(TypedDict):
    tokens: torch.Tensor  # ↓ Core request
    start_pos: torch.Tensor
    page_table: torch.Tensor
    kv_cache: Any  # ↓ Borrowed resources
    sampling_params: Any  # ↓ Sampling
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

        self._run_guarded(operation)

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

        self._run_guarded(operation)

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
        execution: Sequence[EagerExecutor | TracedExecutor] | None = None,  # ↓ Internal dispatch
    ) -> Any:
        """Fan out prefill rows by slot and restore their source-row order."""

        def operation() -> Any:
            lane_results = []
            for lane_idx, rows, lane_kwargs in self._prefill_lane_requests(
                tokens,
                page_table,
                prompt_lens=prompt_lens,
                start_pos=start_pos,
                empty_slots=empty_slots,
                kv_cache=kv_cache,
                sampling_params=sampling_params,
                execution=execution,
            ):
                result = self.lanes[lane_idx].prefill_forward(**lane_kwargs)
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
            yield lane_idx, rows, lane_kwargs

    def _decode_lane_requests(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        kv_cache: Any,  # ↓ Borrowed resources
        sampling_params: Any,  # ↓ Sampling
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


def _slice_rows(value: Any, rows: list[int]) -> Any:
    if isinstance(value, torch.Tensor):
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
    had_tuple = False
    for rows, result in lane_results:
        output, log_probs, was_tuple = _unwrap_output(result)
        if log_probs is not None:
            raise NotImplementedError("DP log probabilities are not implemented")
        had_tuple = had_tuple or was_tuple
        unwrapped.append((rows, output))
    if all(output is None for _, output in unwrapped):
        return (None, None) if had_tuple else None
    if not all(isinstance(output, torch.Tensor) for _, output in unwrapped):
        return [output for _, output in unwrapped]

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
    return (output, None) if had_tuple else output


def _aggregate_contiguous_outputs(lane_results: list[Any], *, force_tokens: bool = False) -> Any:
    unwrapped = []
    had_tuple = False
    for result in lane_results:
        output, log_probs, was_tuple = _unwrap_output(result)
        if log_probs is not None:
            raise NotImplementedError("DP log probabilities are not implemented")
        had_tuple = had_tuple or was_tuple
        unwrapped.append(output)
    if all(output is None for output in unwrapped):
        return (None, None) if had_tuple else None
    if not all(isinstance(output, torch.Tensor) for output in unwrapped):
        return unwrapped

    first = unwrapped[0]
    assert isinstance(first, torch.Tensor)
    if force_tokens or _is_token_tensor(first):
        output = torch.cat([lane_output.reshape(-1) for lane_output in unwrapped], dim=0).to(torch.int64)
    else:
        output = torch.cat(unwrapped, dim=0)
    return (output, None) if had_tuple else output


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
