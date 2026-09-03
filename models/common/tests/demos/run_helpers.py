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
) -> None:
    """Compile the concrete prefill and decode cases through the public target surface."""
    assert prefill_tokens.dim() == 2, f"prefill_tokens must be [batch_size, seq_len], got {prefill_tokens.dim()}D"
    assert (
        prefill_page_table.dim() == 2
    ), f"prefill_page_table must be [batch_size, max_blocks], got {prefill_page_table.dim()}D"

    batch_size = prefill_tokens.shape[0]
    decode_start_pos = torch.full(
        (batch_size,),
        prefill_tokens.shape[-1],
        dtype=torch.long,
        device=prefill_tokens.device,
    )

    if prefill_sampling_params is _SAME_SAMPLING_PARAMS:
        prefill_sampling_params = sampling_params

    if sampling_params is not None:
        execution_target.compile_decode(
            tokens=torch.zeros(batch_size, dtype=torch.long, device=prefill_tokens.device),
            start_pos=decode_start_pos,
            page_table=prefill_page_table,
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
    decode_tokens = torch.zeros(batch_size, dtype=torch.long, device=prefill_tokens.device)

    execution_target.compile_decode(
        tokens=decode_tokens,
        start_pos=decode_start_pos,
        page_table=prefill_page_table,
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
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        empty_slots=list(range(batch_size)),
        start_pos=start_pos,
    )

    compile_tokens = torch.zeros(max_batch_size, prompt_len, dtype=tokens.dtype)
    compile_tokens[:batch_size] = tokens
    compile_prompt_lens = torch.zeros(max_batch_size, dtype=prompt_lens.dtype)
    compile_prompt_lens[:batch_size] = prompt_lens
    _compile_prefill_and_decode(
        execution_target,
        prefill_tokens=compile_tokens,
        prefill_page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=compile_prompt_lens,
        empty_slots=list(range(batch_size)),
        start_pos=start_pos,
        sampling_params=sampling_params,
        prefill_sampling_params=prefill_sampling_params,
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

    first_token = prefill_output[0] if isinstance(prefill_output, tuple) else torch.argmax(prefill_output, dim=-1)
    first_token = first_token.view(-1)[:batch_size].detach().cpu()
    generated_token_ids = [[int(token)] for token in first_token.tolist()]

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
                    next_token = torch.argmax(output[:, -1, :], dim=-1)
                else:
                    next_token = output
                next_token = next_token.view(-1)[:batch_size].detach().cpu()
                for user_id, token in enumerate(next_token.tolist()):
                    generated_token_ids[user_id].append(int(token))
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


def assert_cross_batch_consistency(per_repeat_outputs: list[list[list[int]]]) -> None:
    """Assert prompt-position invariance across repeats.

    ``per_repeat_outputs[b][u]`` = truncated token-id list for slot ``u`` of repeat ``b``.
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
                    first_failure = (b, offset)
    assert failed == 0, (
        f"ci-eval-32: {failed}/{total} cross-batch consistency checks failed "
        f"(first at repeat {first_failure[0]}->{first_failure[0] + 1}, offset {first_failure[1]})"
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
) -> None:
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
    """
    assert (
        len(prompts) == max_batch_size
    ), f"ci-eval-32 expects len(prompts)==max_batch_size; got {len(prompts)} vs {max_batch_size}"
    stop_ids = hf_stop_ids(tokenizer, hf_model_id)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    # Garbage guard targets only special tokens that are NOT recognized stops: a legitimate
    # stop is removed by truncation, so anything special left in the body is degenerate output.
    garbage_ids = special_ids - stop_ids
    logger.info(
        f"ci-eval-32: repeat_batches={repeat_batches}, N={len(prompts)}, "
        f"stop_ids={sorted(stop_ids)}, |special_ids|={len(special_ids)}, sampling_params={sampling_params}"
    )

    per_repeat: list[list[list[int]]] = []
    for i in range(repeat_batches):
        traced_executor = make_executor()
        try:
            kv_cache = allocate_kv_cache(traced_executor)
            rotated = rotate_prompts(prompts, i)
            tokens, prompt_lens = tokenize_fn(rotated)
            result = run_perf_benchmark(
                traced_executor,
                tokens=tokens,
                kv_cache=kv_cache,
                page_table=page_table,
                num_decode_tokens=num_decode_tokens,
                max_batch_size=max_batch_size,
                prompt_lens=prompt_lens,
                sampling_params=sampling_params,
            )
        finally:
            traced_executor.cleanup()
        truncated = [truncate_at_stop(ids, stop_ids) for ids in result.generated_token_ids]
        for u, ids in enumerate(truncated):
            bad = set(ids) & garbage_ids
            assert not bad, f"ci-eval-32: user {u} produced special token(s) {sorted(bad)} mid-stream"
        per_repeat.append(truncated)
        logger.info(f"ci-eval-32 repeat {i}: truncated lengths = {[len(t) for t in truncated]}")

    assert_cross_batch_consistency(per_repeat)
    logger.info(f"ci-eval-32: all {(repeat_batches - 1) * len(prompts)} cross-batch consistency checks passed")
