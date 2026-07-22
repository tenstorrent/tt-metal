# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmup coverage planning and trace-activation coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from loguru import logger

from models.common.sampling import SamplingParams


@dataclass(frozen=True)
class WarmupCase:
    operation: str
    batch_size: int
    sequence_length: int | None
    sampling_path: str
    cached_tokens: int = 0


@dataclass(frozen=True)
class WarmupPlan:
    prefill: tuple[WarmupCase, ...]
    decode: tuple[WarmupCase, ...]


@dataclass(frozen=True)
class WarmupCoverage:
    eager: frozenset[WarmupCase]
    trace_registered: frozenset[WarmupCase]
    traces_captured: bool


class WarmupCoordinator:
    """Own warmup coverage and the compile-all-before-capture barrier."""

    def __init__(
        self,
        *,
        config: Any,
        trace_config: Any,
        execution: Any,
        eager: Any,
        trace_compiler: Any | None,
        model: Any,
        page_table_layout: Any,
        prefill_sequence_lengths: tuple[int, ...],
        device_sampling_enabled: bool,
        ensure_sampling_buffers: Callable[[], None],
        validate_bound_cache: Callable[[Any], None],
    ) -> None:
        self.config = config
        self.trace_config = trace_config
        self.execution = execution
        self.eager = eager
        self.trace_compiler = trace_compiler
        self.model = model
        self.page_table_layout = page_table_layout
        self.prefill_sequence_lengths = tuple(int(value) for value in prefill_sequence_lengths)
        self.device_sampling_enabled = bool(device_sampling_enabled)
        self._ensure_sampling_buffers = ensure_sampling_buffers
        self._validate_bound_cache = validate_bound_cache
        self._eager: set[WarmupCase] = set()
        self._trace_registered: set[WarmupCase] = set()
        self._trace_decisions: dict[str, bool] = {}
        self._captured = False
        self._prefill_trace_postprocess_primed = False

    @property
    def coverage(self) -> WarmupCoverage:
        return WarmupCoverage(frozenset(self._eager), frozenset(self._trace_registered), self._captured)

    @property
    def already_warmed_up_prefill(self) -> bool:
        required = set(self.plan(can_sample_on_device=self.device_sampling_enabled).prefill)
        if not required.issubset(self._eager):
            return False
        if (
            not bool(getattr(self.trace_config, "prefill_enabled", False))
            or self._trace_decisions.get("prefill") is False
        ):
            return True
        return required.issubset(self._trace_registered) and self._captured

    def plan(self, *, can_sample_on_device: bool) -> WarmupPlan:
        sampling_paths = ["logits"]
        if can_sample_on_device:
            sampling_paths.append("topk")
        sampling_config = getattr(getattr(self.model, "sampling", None), "config", None)
        allow_argmax = bool(getattr(sampling_config, "allow_force_argmax", False))
        prefill = []
        max_batch_size = int(self.model.config.max_batch_size)
        for sequence_length in self.prefill_sequence_lengths:
            batches = self.config.prefill_batch_sizes if sequence_length == 128 else (1,)
            for batch_size in batches:
                if batch_size <= max_batch_size:
                    batch_sampling_paths = sampling_paths + (
                        ["argmax"] if can_sample_on_device and allow_argmax and batch_size == 1 else []
                    )
                    prefill.extend(
                        WarmupCase("prefill", batch_size, sequence_length, sampling_path)
                        for sampling_path in batch_sampling_paths
                    )
            cached_prompt_length = int(self.page_table_layout.block_size) + sequence_length
            if cached_prompt_length <= (
                int(self.page_table_layout.raw_capacity_width) * int(self.page_table_layout.block_size)
            ):
                prefill.extend(
                    WarmupCase(
                        "prefill",
                        1,
                        sequence_length,
                        sampling_path,
                        cached_tokens=int(self.page_table_layout.block_size),
                    )
                    for sampling_path in sampling_paths + (["argmax"] if can_sample_on_device and allow_argmax else [])
                )

        decode_paths = ["logits"]
        if can_sample_on_device:
            if allow_argmax:
                decode_paths.append("argmax")
            if not allow_argmax or self.config.include_decode_top_k:
                decode_paths.append("topk")
        decode = tuple(WarmupCase("decode", max_batch_size, None, sampling_path) for sampling_path in decode_paths)
        return WarmupPlan(tuple(prefill), decode)

    def warmup_prefill(
        self,
        *,
        kv_cache: Any,
        enable_trace: bool,
        can_sample_on_device: bool,
        **_: Any,
    ) -> None:
        self._validate_hints("prefill", enable_trace, can_sample_on_device)
        self._trace_decisions["prefill"] = bool(enable_trace)
        if (
            enable_trace
            and self._trace_decisions.get("decode") is False
            and bool(getattr(self.trace_config, "decode_enabled", False))
        ):
            del self._trace_decisions["decode"]
        self._validate_bound_cache(kv_cache)
        if can_sample_on_device:
            self._ensure_sampling_buffers()
        plan = self.plan(can_sample_on_device=can_sample_on_device)
        destination = self._trace_registered if enable_trace else self._eager
        cases = plan.prefill
        if enable_trace and can_sample_on_device:
            # The hidden-body trace is sampling-independent, but its retained
            # post-trace inputs must support both aliases. Register the forced
            # top-k variant first so the shared artifact owns a K/P/T buffer.
            cases = tuple(sorted(cases, key=lambda case: case.sampling_path != "topk"))
        for case in cases:
            if case in destination:
                continue
            sampling = None
            if case.sampling_path == "argmax":
                sampling = _greedy_sampling_params(case.batch_size)
            elif case.sampling_path == "topk":
                sampling = _topk_sampling_params(case.batch_size)
            actual_uncached_lengths = (int(case.sequence_length),)
            if (
                case.batch_size == 1
                and case.sequence_length == 128
                and case.cached_tokens == 0
                and (
                    case.sampling_path == "argmax"
                    or (case.sampling_path == "topk" and int(self.model.config.max_batch_size) >= 32)
                )
            ):
                # Q128 single-user sampled postprocessing has one TT slice
                # program per tile start. Prime all four without expanding the
                # public warmup coverage model.
                actual_uncached_lengths = (32, 64, 96, 128)
            for actual_uncached_length in actual_uncached_lengths:
                prompt_length = case.cached_tokens + actual_uncached_length
                tokens = torch.zeros((case.batch_size, prompt_length), dtype=torch.long)
                prompt_lens = torch.full((case.batch_size,), prompt_length, dtype=torch.long)
                width = _ceil_div(prompt_length, int(self.page_table_layout.block_size))
                page_table = torch.zeros((case.batch_size, width), dtype=torch.int32)
                start_pos = (
                    torch.full((case.batch_size,), case.cached_tokens, dtype=torch.long) if case.cached_tokens else None
                )
                self.execution.compile_prefill(
                    tokens=tokens,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    prompt_lens=prompt_lens,
                    empty_slots=list(range(case.batch_size)),
                    start_pos=start_pos,
                    sampling_params=sampling,
                    enable_trace=enable_trace,
                )
            destination.add(case)
        self._maybe_capture()

    def warmup_decode(
        self,
        *,
        kv_cache: Any,
        enable_trace: bool,
        max_batch_size: int,
        num_blocks: int,
        can_sample_on_device: bool,
        **_: Any,
    ) -> None:
        self._validate_hints("decode", enable_trace, can_sample_on_device)
        self._trace_decisions["decode"] = bool(enable_trace)
        self._validate_bound_cache(kv_cache)
        lane_batch = int(self.model.config.max_batch_size)
        if int(max_batch_size) != lane_batch:
            raise ValueError(f"decode warmup batch {max_batch_size} does not match lane capacity {lane_batch}")
        if int(num_blocks) <= 0:
            raise ValueError("decode warmup num_blocks must be positive")
        if can_sample_on_device:
            self._ensure_sampling_buffers()
        plan = self.plan(can_sample_on_device=can_sample_on_device)
        destination = self._trace_registered if enable_trace else self._eager
        for case in plan.decode:
            if case in destination:
                continue
            sampling = None
            if case.sampling_path == "argmax":
                sampling = _greedy_sampling_params(lane_batch)
            elif case.sampling_path == "topk":
                sampling = _topk_sampling_params(lane_batch)
            self.execution.compile_decode(
                tokens=torch.zeros(lane_batch, dtype=torch.long),
                start_pos=torch.zeros(lane_batch, dtype=torch.long),
                page_table=torch.zeros((lane_batch, int(num_blocks)), dtype=torch.int32),
                kv_cache=kv_cache,
                sampling_params=sampling,
                enable_trace=enable_trace,
            )
            if not enable_trace:
                logger.info("Compiled decode")
                if sampling is not None:
                    logger.info("Compiled on-device sampling")
            destination.add(case)
        self._maybe_capture()

    def _maybe_capture(self) -> None:
        if self.trace_compiler is None or self._captured:
            return
        required = self.plan(can_sample_on_device=self.device_sampling_enabled)
        required_trace: set[WarmupCase] = set()
        if bool(getattr(self.trace_config, "prefill_enabled", False)):
            prefill_decision = self._trace_decisions.get("prefill")
            if prefill_decision is None:
                return
            if prefill_decision:
                required_trace.update(required.prefill)
        if bool(getattr(self.trace_config, "decode_enabled", False)):
            decode_decision = self._trace_decisions.get("decode")
            if decode_decision is None:
                return
            if decode_decision:
                required_trace.update(required.decode)
        if not required_trace:
            return
        if not required_trace.issubset(self._trace_registered):
            return
        self.trace_compiler.capture_all()
        self._captured = True
        self._prime_prefill_trace_postprocess()

    def _prime_prefill_trace_postprocess(self) -> None:
        if (
            self._prefill_trace_postprocess_primed
            or self._trace_decisions.get("prefill") is False
            or not bool(getattr(self.trace_config, "prefill_enabled", False))
        ):
            return
        sampling_config = getattr(getattr(self.model, "sampling", None), "config", None)
        if not self.device_sampling_enabled or not bool(getattr(sampling_config, "allow_force_argmax", False)):
            self._prefill_trace_postprocess_primed = True
            return
        sequence_length = 128 if 128 in self.prefill_sequence_lengths else int(self.prefill_sequence_lengths[0])
        width = _ceil_div(sequence_length, int(self.page_table_layout.block_size))
        self.execution.prefill_forward(
            tokens=torch.zeros((1, sequence_length), dtype=torch.long),
            page_table=torch.zeros((1, width), dtype=torch.int32),
            prompt_lens=torch.full((1,), sequence_length, dtype=torch.long),
            empty_slots=[0],
            start_pos=None,
            sampling_params=_greedy_sampling_params(1),
            enable_trace=True,
        )
        self._prefill_trace_postprocess_primed = True

    def _validate_hints(self, operation: str, enable_trace: bool, can_sample_on_device: bool) -> None:
        if enable_trace and not bool(getattr(self.trace_config, f"{operation}_enabled")):
            raise ValueError(f"{operation} trace warmup exceeds the configured trace policy")
        if can_sample_on_device and not self.device_sampling_enabled:
            raise ValueError("warmup cannot enable device sampling when it is statically disabled")


def _greedy_sampling_params(batch_size: int) -> SamplingParams:
    return SamplingParams(
        temperature=torch.zeros(batch_size),
        top_k=torch.ones(batch_size, dtype=torch.int32),
        top_p=torch.ones(batch_size),
    )


def _topk_sampling_params(batch_size: int) -> SamplingParams:
    return SamplingParams(
        temperature=torch.ones(batch_size),
        top_k=torch.full((batch_size,), 32, dtype=torch.int32),
        top_p=torch.full((batch_size,), 0.08),
    )


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor
