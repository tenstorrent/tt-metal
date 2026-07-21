# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel composition for already-built single-lane executors."""

from __future__ import annotations

import copy
import dataclasses
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable, Sequence

import torch


class LaneGroupExecutor:
    """Compose fixed-capacity executor lanes behind the execution-target surface."""

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
            self._output_pool = ThreadPoolExecutor(
                max_workers=self.tt_data_parallel,
                thread_name_prefix="tttv2-dp-output",
            )
        except BaseException as primary:
            cleanup_failures = _cleanup_lanes(self.lanes)
            _attach_failures(primary, cleanup_failures, "cleanup_failures")
            raise

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
        def operation() -> None:
            configs = self._lane_configs(config)
            for lane, lane_config in zip(self.lanes, configs):
                lane.configure_paged_kv_cache(lane_config)

        self._run_guarded(operation)

    def allocate_kv_cache(self, *args: Any, **kwargs: Any) -> list[Any]:
        return self._run_guarded(lambda: [lane.allocate_kv_cache(*args, **kwargs) for lane in self.lanes])

    def compile_prefill(self, *args: Any, **kwargs: Any) -> None:
        bound = _bind_positional(args, kwargs, ("tokens", "page_table"), "compile_prefill")
        self._run_guarded(lambda: self._prefill_fanout("compile_prefill", bound))

    def compile_decode(self, *args: Any, **kwargs: Any) -> None:
        bound = _bind_positional(args, kwargs, ("tokens", "start_pos", "page_table"), "compile_decode")
        self._run_guarded(lambda: self._decode_fanout("compile_decode", bound, return_raw=False))

    def warmup_model_prefill(self, *args: Any, **kwargs: Any) -> None:
        self._run_guarded(lambda: self._replicate_warmup("warmup_model_prefill", args, kwargs))

    def warmup_model_decode(self, *args: Any, **kwargs: Any) -> None:
        def operation() -> None:
            lane_kwargs = dict(kwargs)
            if "max_batch_size" in lane_kwargs:
                lane_kwargs["max_batch_size"] = self.per_lane_max_batch_size
            self._replicate_warmup("warmup_model_decode", args, lane_kwargs)

        self._run_guarded(operation)

    def prefill_forward(self, *args: Any, **kwargs: Any) -> Any:
        bound = _bind_positional(args, kwargs, ("tokens", "page_table"), "prefill_forward")
        return self._run_guarded(lambda: self._prefill_fanout("prefill_forward", bound))

    def decode_forward(self, *args: Any, **kwargs: Any) -> Any:
        bound = _bind_positional(args, kwargs, ("tokens", "start_pos", "page_table"), "decode_forward")
        return self._run_guarded(
            lambda: self._decode_fanout(
                "decode_forward",
                bound,
                return_raw=not bound.get("read_from_device", True),
            )
        )

    def read_decode_output(self, lane_outputs: Sequence[Any], async_read: bool = False) -> Any:
        def operation() -> Any:
            self._validate_lane_values(lane_outputs, "decode outputs")
            results = self._run_concurrently(
                lambda lane_idx: self.lanes[lane_idx].read_decode_output(
                    lane_outputs[lane_idx],
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

    def process_decode_output_host(self, lane_outputs: Sequence[Any], is_tokens: bool = False) -> Any:
        def operation() -> Any:
            self._validate_lane_values(lane_outputs, "host decode outputs")
            results = self._run_concurrently(
                lambda lane_idx: self.lanes[lane_idx].process_decode_output_host(
                    lane_outputs[lane_idx],
                    is_tokens=is_tokens,
                )
            )
            return _aggregate_contiguous_outputs(results, force_tokens=is_tokens)

        return self._run_guarded(operation)

    def cleanup(self) -> None:
        failures = self._cleanup_impl()
        if failures:
            primary = failures[0]
            _attach_failures(primary, failures[1:], "cleanup_failures")
            raise primary

    def _prefill_fanout(self, method_name: str, kwargs: dict[str, Any]) -> Any:
        tokens = kwargs.get("tokens")
        page_table = kwargs.get("page_table")
        if not isinstance(tokens, torch.Tensor) or tokens.ndim < 1:
            raise TypeError("prefill tokens must be a torch.Tensor with a batch dimension")
        if not isinstance(page_table, torch.Tensor) or page_table.ndim < 1:
            raise TypeError("prefill page_table must be a torch.Tensor with a batch dimension")
        batch_size = int(tokens.shape[0])
        if int(page_table.shape[0]) != batch_size:
            raise ValueError("prefill tokens and page_table batch sizes must match")

        empty_slots = kwargs.get("empty_slots")
        if empty_slots is None:
            empty_slots = list(range(batch_size))
        else:
            empty_slots = list(empty_slots)
        if len(empty_slots) != batch_size:
            raise ValueError(f"empty_slots length {len(empty_slots)} must match prefill batch {batch_size}")

        lane_results: list[tuple[list[int], Any]] = []
        for lane_idx, rows, local_slots in self._prefill_lane_groups(empty_slots):
            lane_kwargs = dict(kwargs)
            lane_kwargs["tokens"] = _slice_rows(tokens, rows)
            lane_kwargs["page_table"] = _slice_rows(page_table, rows)
            lane_kwargs["empty_slots"] = local_slots
            for key in ("prompt_lens", "start_pos"):
                if lane_kwargs.get(key) is not None:
                    lane_kwargs[key] = _slice_rows(lane_kwargs[key], rows)
            if lane_kwargs.get("sampling_params") is not None:
                lane_kwargs["sampling_params"] = _slice_sampling_params(lane_kwargs["sampling_params"], rows)
            if lane_kwargs.get("kv_cache") is not None:
                lane_kwargs["kv_cache"] = self._lane_value(lane_kwargs["kv_cache"], lane_idx, "KV caches")
            result = getattr(self.lanes[lane_idx], method_name)(**lane_kwargs)
            lane_results.append((rows, result))

        return _aggregate_prefill_outputs(lane_results, batch_size)

    def _decode_fanout(self, method_name: str, kwargs: dict[str, Any], *, return_raw: bool) -> Any:
        tokens = kwargs.get("tokens")
        start_pos = kwargs.get("start_pos")
        page_table = kwargs.get("page_table")
        for name, value in (("tokens", tokens), ("start_pos", start_pos), ("page_table", page_table)):
            if not isinstance(value, torch.Tensor) or value.ndim < 1:
                raise TypeError(f"decode {name} must be a torch.Tensor with a batch dimension")
            if int(value.shape[0]) != self.max_batch_size:
                raise ValueError(
                    f"DP decode expects fixed global batch {self.max_batch_size}; " f"{name} has batch {value.shape[0]}"
                )

        lane_outputs = []
        for lane_idx in range(self.tt_data_parallel):
            start = lane_idx * self.per_lane_max_batch_size
            end = start + self.per_lane_max_batch_size
            rows = list(range(start, end))
            lane_kwargs = dict(kwargs)
            lane_kwargs["tokens"] = _slice_rows(tokens, rows)
            lane_kwargs["start_pos"] = _slice_rows(start_pos, rows)
            lane_kwargs["page_table"] = _slice_rows(page_table, rows)
            if lane_kwargs.get("sampling_params") is not None:
                lane_kwargs["sampling_params"] = _slice_sampling_params(lane_kwargs["sampling_params"], rows)
            if lane_kwargs.get("kv_cache") is not None:
                lane_kwargs["kv_cache"] = self._lane_value(lane_kwargs["kv_cache"], lane_idx, "KV caches")
            lane_outputs.append(getattr(self.lanes[lane_idx], method_name)(**lane_kwargs))

        if return_raw:
            return lane_outputs
        return _aggregate_contiguous_outputs(lane_outputs)

    def _replicate_warmup(self, method_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        for lane_idx, lane in enumerate(self.lanes):
            lane_kwargs = dict(kwargs)
            if lane_kwargs.get("kv_cache") is not None:
                lane_kwargs["kv_cache"] = self._lane_value(lane_kwargs["kv_cache"], lane_idx, "KV caches")
            getattr(lane, method_name)(*args, **lane_kwargs)

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


def _bind_positional(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    names: tuple[str, ...],
    operation: str,
) -> dict[str, Any]:
    if len(args) > len(names):
        raise TypeError(f"{operation} accepts at most {len(names)} positional arguments")
    bound = dict(kwargs)
    for name, value in zip(names, args):
        if name in bound:
            raise TypeError(f"{operation} got multiple values for argument {name!r}")
        bound[name] = value
    return bound


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


def _aggregate_prefill_outputs(lane_results: list[tuple[list[int], Any]], batch_size: int) -> Any:
    if not lane_results:
        return torch.empty((0,), dtype=torch.int32)
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
        output = torch.empty((batch_size,), dtype=torch.int32, device=first.device)
        for rows, lane_output in unwrapped:
            output[rows] = lane_output.reshape(-1).to(torch.int32)
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
        output = torch.cat([lane_output.reshape(-1) for lane_output in unwrapped], dim=0).to(torch.int32)
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
