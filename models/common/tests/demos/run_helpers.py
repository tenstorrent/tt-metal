# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reusable teacher-forcing and benchmark run helpers for demos and tests."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger

import ttnn

_SAME_SAMPLING_PARAMS = object()


def make_contiguous_page_table(batch_size: int, max_seq_len: int, block_size: int = 32) -> torch.Tensor:
    """Create a contiguous demo page table with one disjoint block range per user."""
    if min(batch_size, max_seq_len, block_size) <= 0:
        raise ValueError("page-table dimensions must be positive")
    blocks_per_user = (max_seq_len + block_size - 1) // block_size
    return torch.arange(batch_size * blocks_per_user, dtype=torch.int32).reshape(batch_size, blocks_per_user)


@dataclass
class TeacherForceResult:
    """Result from a teacher-forcing evaluation run."""

    predicted_tokens: list[int]
    predicted_tokens_per_user: list[list[int]]
    reference_top5: torch.Tensor
    prefill_time_s: float = 0.0
    compile_decode_time_s: float = 0.0
    decode_times_s: list[float] = field(default_factory=list)
    batch_size: int = 1
    prefill_len: int = 0

    def top1_accuracy(self) -> float:
        matches = sum(
            1 for i, prediction in enumerate(self.predicted_tokens) if self.reference_top5[i, 0].item() == prediction
        )
        return matches / len(self.predicted_tokens)

    def top5_accuracy(self) -> float:
        matches = sum(
            1 for i, prediction in enumerate(self.predicted_tokens) if prediction in self.reference_top5[i, :]
        )
        return matches / len(self.predicted_tokens)

    @property
    def ttft_ms(self) -> float:
        return self.prefill_time_s / self.batch_size * 1000 if self.batch_size else 0.0

    @property
    def prefill_time_to_token_s(self) -> float:
        return self.prefill_time_s / self.batch_size if self.batch_size else 0.0

    @property
    def prefill_tok_s(self) -> float:
        return (self.batch_size * self.prefill_len) / self.prefill_time_s if self.prefill_time_s > 0 else 0.0

    @property
    def decode_tok_s_u(self) -> float:
        return len(self.decode_times_s) / sum(self.decode_times_s) if self.decode_times_s else 0.0

    @property
    def decode_tok_s(self) -> float:
        return self.decode_tok_s_u * self.batch_size


@dataclass
class PerfBenchmarkResult:
    """Result from a performance benchmark run."""

    prefill_time_s: float
    compile_decode_time_s: float
    decode_times_s: list[float]
    batch_size: int
    num_decode_tokens: int
    generated_token_ids: list[list[int]]
    decode_iteration_times_s: list[float] = field(default_factory=list)
    argmax_top2_margins: list[list[float]] | None = None

    @property
    def ttft_ms(self) -> float:
        """TTTv1-style average time to first token per user."""
        return self.prefill_time_s / self.batch_size * 1000

    @property
    def tok_s_u(self) -> float:
        """Tokens per second per user during steady-state decode."""
        if not self.decode_times_s:
            return 0.0
        return len(self.decode_times_s) / sum(self.decode_times_s)

    @property
    def tok_s(self) -> float:
        """Total decode throughput."""
        return self.tok_s_u * self.batch_size

    @property
    def decode_latency_mean_ms(self) -> float:
        if not self.decode_times_s:
            return 0.0
        return sum(self.decode_times_s) / len(self.decode_times_s) * 1000

    def meets_target(self, expected: dict, tolerance: float = 0.05) -> dict[str, bool]:
        """Check benchmark metrics against the expected thresholds."""
        return {
            "tok_s_u": self.tok_s_u >= expected["tok_s_u"] * (1 - tolerance),
            "ttft_ms": self.ttft_ms <= expected["ttft_ms"] * (1 + tolerance),
        }


def _compile_prefill_and_decode(
    execution_target,
    *,
    prefill_tokens: torch.Tensor,
    prefill_page_table: torch.Tensor,
    kv_cache=None,
    prompt_lens: torch.Tensor | None = None,
    empty_slots: list[int] | None = None,
    start_pos: torch.Tensor | None = None,
    sampling_params=None,
    prefill_sampling_params=_SAME_SAMPLING_PARAMS,
    decode_tokens: torch.Tensor | None = None,
    decode_start_pos: torch.Tensor | None = None,
    decode_page_table: torch.Tensor | None = None,
) -> None:
    """Compile the concrete prefill and decode cases through the public target surface."""
    assert prefill_tokens.dim() == 2, f"prefill_tokens must be [batch_size, seq_len], got {prefill_tokens.dim()}D"
    assert (
        prefill_page_table.dim() == 2
    ), f"prefill_page_table must be [batch_size, max_blocks], got {prefill_page_table.dim()}D"

    batch_size = prefill_tokens.shape[0]
    if decode_tokens is None:
        decode_tokens = torch.zeros(batch_size, dtype=torch.long, device=prefill_tokens.device)
    if decode_start_pos is None:
        decode_start_pos = torch.full(
            (decode_tokens.shape[0],),
            prefill_tokens.shape[-1],
            dtype=torch.long,
            device=prefill_tokens.device,
        )
    if decode_page_table is None:
        decode_page_table = prefill_page_table

    if prefill_sampling_params is _SAME_SAMPLING_PARAMS:
        prefill_sampling_params = sampling_params

    if sampling_params is not None:
        execution_target.compile_decode(
            tokens=decode_tokens,
            start_pos=decode_start_pos,
            page_table=decode_page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )
        execution_target.compile_prefill(
            tokens=prefill_tokens,
            page_table=prefill_page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
            sampling_params=prefill_sampling_params,
        )
        return

    execution_target.compile_prefill(
        tokens=prefill_tokens,
        page_table=prefill_page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        empty_slots=empty_slots,
        start_pos=start_pos,
        sampling_params=None,
    )
    execution_target.compile_decode(
        tokens=decode_tokens,
        start_pos=decode_start_pos,
        page_table=decode_page_table,
        kv_cache=kv_cache,
        sampling_params=None,
    )


def _profiler_start(profiler, name: str) -> None:
    if profiler is not None:
        profiler.start(name)


def _profiler_end(profiler, name: str) -> None:
    if profiler is not None:
        profiler.end(name)


def run_teacher_forcing(
    executor,
    *,
    prompt_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    top5_tokens: torch.Tensor,
    kv_cache: list,
    page_table: torch.Tensor,
    max_batch_size: int = 1,
    profiler=None,
) -> TeacherForceResult:
    """Run teacher-forcing accuracy measurement against an execution target."""
    execution_target = executor
    batch_size = prompt_tokens.shape[0]
    assert (
        batch_size == max_batch_size
    ), f"Teacher forcing expects active batch to match max_batch_size, got {batch_size} vs {max_batch_size}"
    prompt_len = prompt_tokens.shape[-1]
    num_target = len(reference_tokens) - prompt_len
    prompt_lens = torch.tensor([prompt_len] * batch_size)
    empty_slots = list(range(batch_size))

    _compile_prefill_and_decode(
        execution_target,
        prefill_tokens=prompt_tokens,
        prefill_page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        empty_slots=empty_slots,
    )

    logger.info(f"Teacher forcing: prefilling {prompt_len} tokens with batch={batch_size}")
    _profiler_start(profiler, "inference_prefill")
    try:
        start_time = time.perf_counter()
        prefill_output = execution_target.prefill_forward(
            prompt_tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
        )
        _synchronize_target(execution_target)
        prefill_time_s = time.perf_counter() - start_time
    finally:
        _profiler_end(profiler, "inference_prefill")
    first_tokens = torch.argmax(prefill_output, dim=-1).view(-1).tolist()
    predicted_tokens_per_user = [[int(token)] for token in first_tokens]

    logger.info(f"Teacher forcing: decoding {num_target - 1} tokens")
    compile_decode_time_s = 0.0
    decode_times_s = []
    _profiler_start(profiler, "inference_decode")
    try:
        for step in range(1, num_target):
            ground_truth_token = reference_tokens[prompt_len + step - 1]
            decode_token = torch.full((batch_size,), ground_truth_token, dtype=torch.long)
            current_pos = torch.full((batch_size,), prompt_len + step - 1, dtype=torch.long)
            start_time = time.perf_counter()
            logits, _ = execution_target.decode_forward(
                decode_token,
                current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                read_from_device=True,
            )
            elapsed = time.perf_counter() - start_time
            if step == 1:
                compile_decode_time_s = elapsed
            else:
                decode_times_s.append(elapsed)

            next_tokens = torch.argmax(logits[:, -1, :], dim=-1).view(-1).tolist()
            for user_id, token in enumerate(next_tokens):
                predicted_tokens_per_user[user_id].append(int(token))
    finally:
        _profiler_end(profiler, "inference_decode")

    return TeacherForceResult(
        predicted_tokens=predicted_tokens_per_user[0],
        predicted_tokens_per_user=predicted_tokens_per_user,
        reference_top5=top5_tokens[:num_target],
        prefill_time_s=prefill_time_s,
        compile_decode_time_s=compile_decode_time_s,
        decode_times_s=decode_times_s,
        batch_size=batch_size,
        prefill_len=prompt_len,
    )


def _split_output(output):
    return output if isinstance(output, tuple) else (output, None)


def _target_mesh_device(execution_target):
    return getattr(execution_target, "mesh_device", None)


def _target_cluster_shape(execution_target):
    cluster_shape = getattr(execution_target, "cluster_shape", None)
    if cluster_shape is not None:
        return list(cluster_shape)
    mesh_device = _target_mesh_device(execution_target)
    return list(mesh_device.shape) if mesh_device is not None else [1, 1]


def _synchronize_target(execution_target):
    mesh_devices = getattr(execution_target, "mesh_devices", None)
    if mesh_devices is not None:
        for mesh_device in mesh_devices:
            if mesh_device is not None:
                ttnn.synchronize_device(mesh_device)
        return
    mesh_device = _target_mesh_device(execution_target)
    if mesh_device is not None:
        ttnn.synchronize_device(mesh_device)


def _concat_host_output(output, cluster_shape):
    output_tensors = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(output)]
    _, columns = cluster_shape
    mesh_rows = [output_tensors[i : i + columns] for i in range(0, len(output_tensors), columns)]
    return torch.cat([torch.cat(row, dim=-1) for row in mesh_rows], dim=1)


def _process_legacy_sampled_tokens(output, batch_size, cluster_shape):
    torch_output = _concat_host_output(output, cluster_shape)
    if torch_output.ndim >= 4:
        if torch_output.shape[2] >= batch_size:
            return torch_output[0, 0, :batch_size, 0]
        if torch_output.shape[3] >= batch_size:
            return torch_output[0, 0, 0, :batch_size]
    return torch_output.reshape(-1)[:batch_size]


def _to_host(value, *, blocking):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.cpu()
    try:
        return value.cpu(blocking=blocking)
    except TypeError:
        return value.cpu()


def _submit_decode_read(execution_target, decode_output):
    read_decode_output = getattr(execution_target, "read_decode_output", None)
    if callable(read_decode_output):
        return read_decode_output(decode_output, async_read=True)

    output, log_probs = _split_output(decode_output)
    host_output = (_to_host(output, blocking=False), _to_host(log_probs, blocking=False))
    return host_output, [ttnn.record_event(_target_mesh_device(execution_target), 0)]


def _synchronize_read_events(events):
    if events is None:
        return
    if not isinstance(events, (list, tuple, set)):
        events = [events]
    for event in events:
        ttnn.event_synchronize(event)


def _consume_sampled_output(
    execution_target,
    host_output,
    batch_size,
    cluster_shape,
    generated_token_ids,
    *,
    process_host_output,
):
    process_decode_output_host = (
        getattr(execution_target, "process_decode_output_host", None) if process_host_output else None
    )
    if callable(process_decode_output_host):
        tokens, _ = process_decode_output_host(host_output, is_tokens=True)
    else:
        tokens, _ = _split_output(host_output)
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.view(-1)[:batch_size].detach().cpu()
        else:
            tokens = _process_legacy_sampled_tokens(tokens, batch_size, cluster_shape)
            tokens = tokens.view(-1)[:batch_size].detach().cpu()

    for user_id, token in enumerate(tokens.tolist()):
        generated_token_ids[user_id].append(int(token))


def _host_argmax_with_margins(logits: torch.Tensor, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return host argmax and top1-minus-top2 margins for eval diagnostics."""

    rows = logits[:, -1, :]
    top2 = torch.topk(rows.float(), k=2, dim=-1)
    tokens = top2.indices[:, 0].view(-1)[:batch_size].detach().cpu()
    margins = (top2.values[:, 0] - top2.values[:, 1]).view(-1)[:batch_size].detach().cpu()
    return tokens, margins


def run_perf_benchmark(
    executor,
    *,
    tokens: torch.Tensor,
    kv_cache: list,
    page_table: torch.Tensor,
    num_decode_tokens: int = 128,
    max_batch_size: int = 1,
    prompt_lens: torch.Tensor | None = None,
    start_pos: list[int] | None = None,
    sampling_params=None,
    prefill_sampling_params=_SAME_SAMPLING_PARAMS,
    pipeline_readback: bool = False,
    profiler=None,
    collect_argmax_diagnostics: bool = False,
) -> PerfBenchmarkResult:
    """Run the timed prefill and decode loop against a public execution target."""
    execution_target = executor
    mesh_device = _target_mesh_device(execution_target)
    has_public_readback = callable(getattr(execution_target, "read_decode_output", None))
    has_legacy_readback = mesh_device is not None and hasattr(ttnn, "record_event")
    can_pipeline_readback = (
        sampling_params is not None
        and pipeline_readback
        and hasattr(ttnn, "event_synchronize")
        and (has_public_readback or has_legacy_readback)
    )
    if sampling_params is not None and pipeline_readback and not can_pipeline_readback:
        logger.warning("PIPELINE_READBACK requested, but this execution target does not expose async readback")

    batch_size, prompt_len = tokens.shape
    max_batch_size = max(max_batch_size, batch_size)
    cluster_shape = _target_cluster_shape(execution_target)
    prompt_lens = prompt_lens if prompt_lens is not None else torch.tensor([prompt_len] * batch_size)
    if prefill_sampling_params is _SAME_SAMPLING_PARAMS:
        prefill_sampling_params = sampling_params
    prefill_kwargs = dict(
        # Decode always carries the full lane-capacity table. Prefill only owns
        # the active request rows, and its public contract requires matching
        # token/page-table batch dimensions.
        page_table=page_table[:batch_size],
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        empty_slots=list(range(batch_size)),
        start_pos=start_pos,
    )

    compile_tokens = torch.zeros(max_batch_size, prompt_len, dtype=tokens.dtype)
    compile_tokens[:batch_size] = tokens
    compile_prompt_lens = torch.zeros(max_batch_size, dtype=prompt_lens.dtype)
    compile_prompt_lens[:batch_size] = prompt_lens
    compile_page_table = page_table
    compile_empty_slots = list(range(batch_size))
    if batch_size < max_batch_size:
        # A partial-cardinality diagnostic still decodes on the model's full
        # lane-capacity program, but prefill owns only its active requests.
        compile_tokens = tokens
        compile_prompt_lens = prompt_lens
        compile_page_table = page_table[:batch_size]
    _compile_prefill_and_decode(
        execution_target,
        prefill_tokens=compile_tokens,
        prefill_page_table=compile_page_table,
        kv_cache=kv_cache,
        prompt_lens=compile_prompt_lens,
        empty_slots=compile_empty_slots,
        start_pos=start_pos,
        sampling_params=sampling_params,
        prefill_sampling_params=prefill_sampling_params,
        decode_tokens=torch.zeros(max_batch_size, dtype=torch.long),
        decode_start_pos=torch.full((max_batch_size,), prompt_len, dtype=torch.long),
        decode_page_table=page_table,
    )

    _profiler_start(profiler, "inference_prefill")
    try:
        start_time = time.perf_counter()
        prefill_output = execution_target.prefill_forward(
            tokens,
            **prefill_kwargs,
            sampling_params=prefill_sampling_params,
        )
        _synchronize_target(execution_target)
        prefill_time = time.perf_counter() - start_time
    finally:
        _profiler_end(profiler, "inference_prefill")

    first_margins = None
    if collect_argmax_diagnostics and sampling_params is None:
        first_token, first_margins = _host_argmax_with_margins(prefill_output, batch_size)
    else:
        first_token = prefill_output[0] if isinstance(prefill_output, tuple) else torch.argmax(prefill_output, dim=-1)
        first_token = first_token.view(-1)[:batch_size].detach().cpu()
    generated_token_ids = [[int(token)] for token in first_token.tolist()]
    argmax_top2_margins = [[float(margin)] for margin in first_margins.tolist()] if first_margins is not None else None

    current_tokens = torch.zeros(max_batch_size, dtype=torch.long)
    current_tokens[:batch_size] = first_token
    current_pos = torch.full((max_batch_size,), -1, dtype=torch.long)
    current_pos[:batch_size] = prompt_lens[:batch_size]

    compile_time = None
    decode_times = []
    decode_iteration_times = []
    sampled_decode_start = None
    pending_host_output = None
    pending_read_events = None

    _profiler_start(profiler, "inference_decode")
    try:
        for iteration in range(num_decode_tokens):
            start_time = time.perf_counter()
            read_from_device = sampling_params is None or not can_pipeline_readback
            if sampling_params is not None and iteration == 1:
                sampled_decode_start = start_time

            decode_output = execution_target.decode_forward(
                current_tokens,
                current_pos,
                page_table=page_table,
                kv_cache=kv_cache,
                read_from_device=read_from_device,
                sampling_params=sampling_params,
                reset_batch=iteration == 0,
            )
            output, _ = _split_output(decode_output)

            completed_host_output = None
            if can_pipeline_readback:
                host_output, read_events = _submit_decode_read(execution_target, decode_output)
                if pending_read_events is not None:
                    _synchronize_read_events(pending_read_events)
                    completed_host_output = pending_host_output
                pending_host_output = host_output
                pending_read_events = read_events

            if sampling_params is not None and iteration == 0:
                _synchronize_target(execution_target)
            elapsed = time.perf_counter() - start_time

            if iteration == 0:
                compile_time = elapsed
            else:
                decode_iteration_times.append(elapsed)
                if sampling_params is None or can_pipeline_readback:
                    decode_times.append(elapsed)

            if completed_host_output is not None:
                _consume_sampled_output(
                    execution_target,
                    completed_host_output,
                    batch_size,
                    cluster_shape,
                    generated_token_ids,
                    process_host_output=True,
                )

            if sampling_params is None:
                if isinstance(output, torch.Tensor) and output.dim() >= 2:
                    if collect_argmax_diagnostics:
                        next_token, next_margins = _host_argmax_with_margins(output, batch_size)
                    else:
                        next_token = torch.argmax(output[:, -1, :], dim=-1)
                else:
                    if collect_argmax_diagnostics:
                        raise TypeError("argmax margin diagnostics require host decode logits")
                    next_token = output
                next_token = next_token.view(-1)[:batch_size].detach().cpu()
                for user_id, token in enumerate(next_token.tolist()):
                    generated_token_ids[user_id].append(int(token))
                    if argmax_top2_margins is not None:
                        argmax_top2_margins[user_id].append(float(next_margins[user_id]))
                current_tokens[:batch_size] = next_token
            elif not can_pipeline_readback:
                _consume_sampled_output(
                    execution_target,
                    decode_output,
                    batch_size,
                    cluster_shape,
                    generated_token_ids,
                    process_host_output=False,
                )
            current_pos[:batch_size] += 1
    finally:
        _profiler_end(profiler, "inference_decode")

    if sampling_params is not None:
        if pending_read_events is not None:
            _synchronize_read_events(pending_read_events)
            _consume_sampled_output(
                execution_target,
                pending_host_output,
                batch_size,
                cluster_shape,
                generated_token_ids,
                process_host_output=True,
            )
        if sampled_decode_start is not None and not can_pipeline_readback:
            _synchronize_target(execution_target)
            sampled_decode_time = time.perf_counter() - sampled_decode_start
            decode_times = [sampled_decode_time / (num_decode_tokens - 1)] * (num_decode_tokens - 1)

    return PerfBenchmarkResult(
        prefill_time_s=prefill_time,
        compile_decode_time_s=compile_time or 0.0,
        decode_times_s=decode_times,
        batch_size=batch_size,
        num_decode_tokens=num_decode_tokens,
        generated_token_ids=generated_token_ids,
        decode_iteration_times_s=decode_iteration_times,
        argmax_top2_margins=argmax_top2_margins,
    )


def _add_token_ids(target: set[int], value) -> None:
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        target.add(int(value))
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _add_token_ids(target, item)


def _stop_token_ids(tokenizer) -> set[int]:
    stop_ids: set[int] = set()
    _add_token_ids(stop_ids, getattr(tokenizer, "eos_token_id", None))
    _add_token_ids(stop_ids, getattr(tokenizer, "stop_tokens", None))
    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        eot_id = convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot_id, int) and eot_id >= 0:
            stop_ids.add(eot_id)
    return stop_ids


def _truncate_at_stop(token_ids: list[int], stop_ids: set[int]) -> list[int]:
    for index, token_id in enumerate(token_ids):
        if token_id in stop_ids:
            return token_ids[:index]
    return token_ids


def assert_no_special_tokens(
    generated_token_ids: list[list[int]],
    tokenizer,
    *,
    case_name: str = "",
    is_ci_env: bool | None = None,
) -> None:
    """Warn locally; fail under CI or ``TT_DEMO_STRICT_SPECIAL_TOKENS=1``."""
    if is_ci_env is None:
        is_ci_env = os.environ.get("CI") == "true" or os.environ.get("TT_DEMO_STRICT_SPECIAL_TOKENS") == "1"

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    stop_ids = _stop_token_ids(tokenizer)
    offending_users = 0
    for token_ids in generated_token_ids:
        output_before_stop = _truncate_at_stop(list(token_ids), stop_ids)
        if any(token_id in special_ids for token_id in output_before_stop):
            offending_users += 1

    if offending_users == 0:
        return

    prefix = f"[{case_name}] " if case_name else ""
    message = f"{prefix}model produced special tokens ({offending_users}/{len(generated_token_ids)} users)"
    logger.warning(message)
    if is_ci_env:
        raise AssertionError(message)


def load_eval_repeat_prompts_batch32() -> list[str]:
    """The 32 numeric sequence-continuation prompts TTTv1's ci-eval-32 uses (parity)."""
    path = Path("models/tt_transformers/demo/sample_prompts/eval_repeat_prompts_batch32.json")
    with open(path) as f:
        data = json.load(f)
    return [entry["prompt"] for entry in data]


def rotate_prompts(all_prompts: list[str], repeat: int) -> list[str]:
    """Rotate the prompt->slot assignment by ``repeat``: slot j holds prompt (j+repeat)%N."""
    n = len(all_prompts)
    return [all_prompts[(j + repeat) % n] for j in range(n)]


def eval_page_table_for_repeat(page_table: torch.Tensor, repeat: int, *, mode: str) -> torch.Tensor:
    """Select physical KV allocation for one rotated eval repeat.

    ``slot-stable`` is the TTTv1 acceptance geometry: page-table row ``j``
    remains at decode slot ``j`` while prompts rotate. ``prompt-stable`` is a
    diagnostic A/B: rows rotate with prompts, so a prompt retains its original
    physical KV blocks even as it moves to another decode slot.
    """

    if mode == "slot-stable":
        return page_table
    if mode == "prompt-stable":
        return torch.roll(page_table, shifts=-repeat, dims=0)
    raise ValueError(f"unsupported eval page-table mode {mode!r}; use 'slot-stable' or 'prompt-stable'")


def eval_decode_trace_mode(mode: str) -> str:
    """Resolve the eval decode execution A/B without changing its default gate."""

    if mode == "traced":
        return "decode_only"
    if mode == "eager":
        return "none"
    raise ValueError(f"unsupported eval decode mode {mode!r}; use 'traced' or 'eager'")


def require_canonical_eval_modes_in_ci(environ) -> None:
    """Prevent diagnostic A/B knobs from replacing a canonical CI gate."""

    if environ.get("CI") != "true":
        return
    noncanonical = []
    if environ.get("EVAL_DECODE_MODE", "traced") != "traced":
        noncanonical.append("EVAL_DECODE_MODE")
    if environ.get("EVAL_PAGE_TABLE_MODE", "slot-stable") != "slot-stable":
        noncanonical.append("EVAL_PAGE_TABLE_MODE")
    noncanonical.extend(name for name in ("EVAL_IDENTICAL_PROMPT_INDEX", "EVAL_ACTIVE_BATCH_SIZE") if name in environ)
    if noncanonical:
        raise RuntimeError("diagnostic eval modes cannot replace the canonical CI gate: " + ", ".join(noncanonical))


def truncate_at_stop(ids: list[int], stop_ids: set[int]) -> list[int]:
    """Prefix of ``ids`` up to (excluding) the first id in ``stop_ids``."""
    out: list[int] = []
    for t in ids:
        if t in stop_ids:
            break
        out.append(t)
    return out


def hf_stop_ids(tokenizer, hf_model_id: str | None = None) -> set[int]:
    """Best-effort stop-token id set for an HF ``AutoTokenizer``.

    Raw HF tokenizers have no ``.stop_tokens`` (that only exists on the TTTv1 wrapped
    tokenizer). Build the set from ``eos_token_id`` (int|list|None), and — when an
    ``hf_model_id`` is supplied — also fold in the model's ``generation_config`` eos ids,
    since chat models (e.g. Llama-3 Instruct) often carry extra eot ids there rather than
    on ``eos_token_id``. Missing/empty -> empty set (truncation simply runs full length).
    """
    stop: set[int] = set()

    def _add(value) -> None:
        if value is None:
            return
        if isinstance(value, bool):  # guard: bool is an int subclass
            return
        if isinstance(value, int):
            stop.add(int(value))
        elif isinstance(value, (list, tuple, set)):
            for e in value:
                _add(e)

    _add(getattr(tokenizer, "eos_token_id", None))
    # tt_transformers ModelArgs.tokenizer augments the HF tokenizer with ``stop_tokens``
    # (eos + any extra eot ids); raw HF AutoTokenizers don't have it (getattr -> None).
    _add(getattr(tokenizer, "stop_tokens", None))
    if hf_model_id is not None:
        try:
            from transformers import GenerationConfig

            gen_cfg = GenerationConfig.from_pretrained(hf_model_id)
            _add(getattr(gen_cfg, "eos_token_id", None))
        except Exception as e:  # generation_config absent / unreadable — eos_token_id is enough
            logger.debug(f"ci-eval-32: could not read generation_config eos ids for {hf_model_id}: {e}")
    return stop


def decode_eval_output(tokenizer, token_ids: list[int], stop_ids: set[int]) -> str:
    """Return the continuation text used by TTTv1's ci-eval-32 comparison.

    Different valid BPE segmentations can decode to the same text.  The TTTv1
    reference stores ``tokenizer.decode(...)`` results and compares those
    strings, so comparing token-id lists here would make the port stricter than
    its source workload and report false slot-rotation failures.
    """
    decoded = tokenizer.decode(truncate_at_stop(token_ids, stop_ids))
    if not isinstance(decoded, str):
        raise TypeError(f"tokenizer.decode must return str, got {type(decoded).__name__}")
    return decoded


def assert_cross_cardinality_consistency(
    outputs_by_cardinality: dict[int, dict[str, str]],
    *,
    expected_cardinalities: tuple[int, ...] = (1, 2, 4, 32),
) -> None:
    """Require each fixed request's decoded output to be invariant as batch cardinality grows."""
    if tuple(outputs_by_cardinality) != expected_cardinalities:
        raise AssertionError(
            f"cross-cardinality experiment expected {expected_cardinalities}, " f"got {tuple(outputs_by_cardinality)}"
        )
    reference: dict[str, tuple[int, str]] = {}
    for cardinality, outputs in outputs_by_cardinality.items():
        if len(outputs) != cardinality:
            raise AssertionError(f"cardinality {cardinality} returned {len(outputs)} request outputs")
        for request_id, output in outputs.items():
            if request_id in reference:
                reference_cardinality, reference_output = reference[request_id]
                if output != reference_output:
                    raise AssertionError(
                        f"request {request_id!r} differs at cardinality {reference_cardinality}->{cardinality}: "
                        f"{reference_output[:120]!r} != {output[:120]!r}"
                    )
            else:
                reference[request_id] = (cardinality, output)


def assert_cross_batch_consistency(
    per_repeat_outputs: list[list[str]],
    *,
    per_repeat_token_ids: list[list[list[int]]] | None = None,
    per_repeat_prompt_lens: list[list[int]] | None = None,
    per_repeat_argmax_margins: list[list[list[float]] | None] | None = None,
) -> None:
    """Assert decoded prompt-position invariance across repeats.

    ``per_repeat_outputs[b][u]`` = decoded continuation for slot ``u`` of repeat ``b``.
    With slot j of repeat b holding prompt (j+b)%N (see ``rotate_prompts``), the same prompt
    sits at slot (offset+1)%N of repeat b and slot offset of repeat b+1 — so those two
    outputs must be identical if no per-user state leaks.
    """
    num_batches = len(per_repeat_outputs)
    assert num_batches >= 2, "cross-batch consistency needs >=2 repeats"
    n = len(per_repeat_outputs[0])
    failed, total = 0, 0
    first_failure = None
    for b in range(num_batches - 1):
        cur, nxt = per_repeat_outputs[b], per_repeat_outputs[b + 1]
        for offset in range(n):
            total += 1
            if cur[(offset + 1) % n] != nxt[offset]:
                failed += 1
                if first_failure is None:
                    current_slot = (offset + 1) % n
                    prompt_index = (offset + b + 1) % n
                    token_detail = ""
                    if per_repeat_token_ids is not None:
                        current_tokens = per_repeat_token_ids[b][current_slot]
                        next_tokens = per_repeat_token_ids[b + 1][offset]
                        common_tokens = 0
                        for current_token, next_token in zip(current_tokens, next_tokens):
                            if current_token != next_token:
                                break
                            common_tokens += 1
                        current_token = current_tokens[common_tokens] if common_tokens < len(current_tokens) else None
                        next_token = next_tokens[common_tokens] if common_tokens < len(next_tokens) else None
                        token_detail = (
                            f"; first token divergence at generation step {common_tokens} "
                            f"({current_token!r} != {next_token!r})"
                        )
                        if per_repeat_argmax_margins is not None:
                            current_repeat_margins = per_repeat_argmax_margins[b]
                            next_repeat_margins = per_repeat_argmax_margins[b + 1]
                            if current_repeat_margins is not None and next_repeat_margins is not None:
                                current_margin = current_repeat_margins[current_slot][common_tokens]
                                next_margin = next_repeat_margins[offset][common_tokens]
                                token_detail += f"; top2 margins {current_margin:.6g} and {next_margin:.6g}"
                    length_detail = ""
                    if per_repeat_prompt_lens is not None:
                        current_length = per_repeat_prompt_lens[b][current_slot]
                        next_length = per_repeat_prompt_lens[b + 1][offset]
                        length_detail = f"; prompt lengths {current_length} and {next_length}"
                    first_failure = (
                        b,
                        offset,
                        current_slot,
                        prompt_index,
                        cur[current_slot],
                        nxt[offset],
                        token_detail,
                        length_detail,
                    )
    assert failed == 0, (
        f"ci-eval-32: {failed}/{total} cross-batch consistency checks failed "
        f"(first at repeat {first_failure[0]} slot {first_failure[2]} -> "
        f"repeat {first_failure[0] + 1} slot {first_failure[1]}, prompt index {first_failure[3]}"
        f"{first_failure[6]}{first_failure[7]}; "
        f"decoded outputs {first_failure[4][:80]!r} != {first_failure[5][:80]!r})"
    )


def assert_within_batch_slot_consistency(
    decoded_outputs: list[str],
    *,
    token_ids: list[list[int]],
    argmax_margins: list[list[float]] | None,
    prompt_index: int,
) -> None:
    """Prove one fixed request is invariant across logical slots in one run."""

    reference = decoded_outputs[0]
    for slot, decoded in enumerate(decoded_outputs[1:], start=1):
        if decoded == reference:
            continue
        reference_tokens = token_ids[0]
        slot_tokens = token_ids[slot]
        common_tokens = 0
        for reference_token, slot_token in zip(reference_tokens, slot_tokens):
            if reference_token != slot_token:
                break
            common_tokens += 1
        reference_token = reference_tokens[common_tokens] if common_tokens < len(reference_tokens) else None
        slot_token = slot_tokens[common_tokens] if common_tokens < len(slot_tokens) else None
        margin_detail = ""
        if argmax_margins is not None:
            margin_detail = (
                f"; top2 margins {argmax_margins[0][common_tokens]:.6g} "
                f"and {argmax_margins[slot][common_tokens]:.6g}"
            )
        raise AssertionError(
            f"ci-eval-32 identical-request diagnostic: prompt index {prompt_index} differs between "
            f"logical slots 0 and {slot} at generation step {common_tokens} "
            f"({reference_token!r} != {slot_token!r}){margin_detail}; "
            f"decoded outputs {reference[:80]!r} != {decoded[:80]!r}"
        )


def run_eval_repeat_batch32(
    *,
    make_executor,
    allocate_kv_cache,
    page_table: torch.Tensor,
    prompts: list[str],
    tokenizer,
    tokenize_fn,
    num_decode_tokens: int,
    max_batch_size: int,
    sampling_params=None,
    repeat_batches: int = 3,
    hf_model_id: str | None = None,
    first_repeat_profiler=None,
    page_table_mode: str = "slot-stable",
    identical_prompt_index: int | None = None,
    active_batch_size: int | None = None,
) -> PerfBenchmarkResult:
    """Drive the ci-eval-32 determinism case, building a fresh traced executor per repeat.

    Each repeat builds its own traced executor (``make_executor()``) and its own zeroed KV
    cache (``allocate_kv_cache(executor)``), so the rotated batches are fully independent —
    no shared device or host state can leak across repeats. The executor is cleaned up after
    each repeat. (The model is bit-deterministic across repeats either way; fresh-per-repeat
    is simply the cleanest independence guarantee for a determinism test, and the trace
    recapture cost is negligible at batch-32.)

    Args:
        make_executor: Zero-arg callable returning a fresh traced executor
            (``run_perf_benchmark`` requires traced). Called once per repeat.
        allocate_kv_cache: Callable(executor) -> fresh zeroed kv_cache bound on that executor.
        page_table: Fixed contiguous page table (shared across repeats).
        prompts: The N (=max_batch_size) prompts to rotate (TTTv1 ci-eval-32 numeric prompts;
            see the module note above re: degenerate-output sensitivity on small models).
        tokenizer: HF tokenizer (for stop / special ids).
        tokenize_fn: Callable(list[str]) -> (tokens, prompt_lens).
        num_decode_tokens: Decode steps per repeat.
        max_batch_size: Padded batch (== len(prompts) for this fixed-32 case).
        sampling_params: None -> host argmax (deterministic, mesh-agnostic default).
        repeat_batches: Number of rotated repeats (TTTv1 uses 3).
        hf_model_id: Optional, to enrich stop ids from generation_config.
        first_repeat_profiler: Optional profiler passed only to the first repeat, allowing a caller
            to emit perf telemetry without changing the three-repeat determinism geometry.
        page_table_mode: ``slot-stable`` preserves TTTv1 acceptance geometry.
            ``prompt-stable`` keeps each prompt on the same physical KV blocks
            as a diagnostic A/B while it rotates through decode slots.
        identical_prompt_index: Diagnostic-only fixed request replicated into
            every logical slot. Slot consistency is checked after the first
            batch, before any repeat-lifecycle comparison.
        active_batch_size: Diagnostic-only number of leading logical slots to
            activate while retaining the model's full lane capacity. This is
            restricted to the identical-request probe so inactive lanes cannot
            complicate prompt-rotation semantics.
    """
    assert (
        len(prompts) == max_batch_size
    ), f"ci-eval-32 expects len(prompts)==max_batch_size; got {len(prompts)} vs {max_batch_size}"
    if active_batch_size is not None:
        if identical_prompt_index is None:
            raise ValueError("active_batch_size requires identical_prompt_index")
        if not 1 <= active_batch_size <= max_batch_size:
            raise ValueError(f"active_batch_size must be in [1, {max_batch_size}]")
    if identical_prompt_index is not None:
        if not 0 <= identical_prompt_index < len(prompts):
            raise ValueError(f"identical_prompt_index must be in [0, {len(prompts) - 1}]")
        prompts = [prompts[identical_prompt_index]] * (active_batch_size or len(prompts))
    stop_ids = hf_stop_ids(tokenizer, hf_model_id)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    # Garbage guard targets only special tokens that are NOT recognized stops: a legitimate
    # stop is removed by truncation, so anything special left in the body is degenerate output.
    garbage_ids = special_ids - stop_ids
    logger.info(
        f"ci-eval-32: repeat_batches={repeat_batches}, N={len(prompts)}, "
        f"stop_ids={sorted(stop_ids)}, |special_ids|={len(special_ids)}, sampling_params={sampling_params}, "
        f"page_table_mode={page_table_mode}, identical_prompt_index={identical_prompt_index}, "
        f"active_batch_size={len(prompts)}"
    )

    per_repeat: list[list[str]] = []
    per_repeat_token_ids: list[list[list[int]]] = []
    per_repeat_prompt_lens: list[list[int]] = []
    per_repeat_argmax_margins: list[list[list[float]] | None] = []
    first_result = None
    for i in range(repeat_batches):
        traced_executor = make_executor()
        try:
            kv_cache = allocate_kv_cache(traced_executor)
            rotated = rotate_prompts(prompts, i)
            tokens, prompt_lens = tokenize_fn(rotated)
            repeat_page_table = eval_page_table_for_repeat(page_table, i, mode=page_table_mode)
            result = run_perf_benchmark(
                traced_executor,
                tokens=tokens,
                kv_cache=kv_cache,
                page_table=repeat_page_table,
                num_decode_tokens=num_decode_tokens,
                max_batch_size=max_batch_size,
                prompt_lens=prompt_lens,
                sampling_params=sampling_params,
                profiler=first_repeat_profiler if i == 0 else None,
                collect_argmax_diagnostics=sampling_params is None,
            )
        finally:
            traced_executor.cleanup()
        truncated = [truncate_at_stop(ids, stop_ids) for ids in result.generated_token_ids]
        for u, ids in enumerate(truncated):
            bad = set(ids) & garbage_ids
            assert not bad, f"ci-eval-32: user {u} produced special token(s) {sorted(bad)} mid-stream"
        decoded = [decode_eval_output(tokenizer, ids, stop_ids) for ids in result.generated_token_ids]
        per_repeat.append(decoded)
        per_repeat_token_ids.append(result.generated_token_ids)
        per_repeat_prompt_lens.append([int(length) for length in prompt_lens])
        per_repeat_argmax_margins.append(getattr(result, "argmax_top2_margins", None))
        if identical_prompt_index is not None:
            assert_within_batch_slot_consistency(
                decoded,
                token_ids=result.generated_token_ids,
                argmax_margins=getattr(result, "argmax_top2_margins", None),
                prompt_index=identical_prompt_index,
            )
        if i == 0:
            first_result = result
        logger.info(
            f"ci-eval-32 repeat {i}: truncated token lengths = {[len(t) for t in truncated]}, "
            f"decoded character lengths = {[len(text) for text in decoded]}"
        )

    if identical_prompt_index is None:
        assert_cross_batch_consistency(
            per_repeat,
            per_repeat_token_ids=per_repeat_token_ids,
            per_repeat_prompt_lens=per_repeat_prompt_lens,
            per_repeat_argmax_margins=per_repeat_argmax_margins,
        )
        logger.info(f"ci-eval-32: all {(repeat_batches - 1) * len(prompts)} cross-batch consistency checks passed")
    else:
        logger.info(
            f"ci-eval-32 identical-request diagnostic: prompt index {identical_prompt_index} "
            f"is invariant across all {len(prompts)} logical slots"
        )
    assert first_result is not None
    return first_result
