# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reusable teacher-forcing and benchmark run helpers for demos and tests."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from loguru import logger

import ttnn


@dataclass
class TeacherForceResult:
    """Result from a teacher-forcing evaluation run."""

    predicted_tokens: list[int]
    predicted_tokens_per_user: list[list[int]]
    reference_top5: torch.Tensor

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


@dataclass
class PerfBenchmarkResult:
    """Result from a performance benchmark run."""

    prefill_time_s: float
    compile_decode_time_s: float
    decode_times_s: list[float]
    batch_size: int
    num_decode_tokens: int
    generated_token_ids: list[list[int]]

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
            sampling_params=sampling_params,
        )
        return

    prefill_output = execution_target.compile_prefill(
        tokens=prefill_tokens,
        page_table=prefill_page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        empty_slots=empty_slots,
        start_pos=start_pos,
        sampling_params=None,
    )
    decode_tokens = torch.zeros(batch_size, dtype=torch.long, device=prefill_tokens.device)
    if isinstance(prefill_output, tuple):
        decode_tokens = prefill_output[0].view(-1)[:batch_size].to(dtype=torch.long, device=prefill_tokens.device)
    elif prefill_output is not None:
        decode_tokens = torch.argmax(prefill_output[:, -1:, :], dim=-1).view(-1)

    execution_target.compile_decode(
        tokens=decode_tokens,
        start_pos=decode_start_pos,
        page_table=prefill_page_table,
        kv_cache=kv_cache,
        sampling_params=None,
    )


def run_teacher_forcing(
    executor,
    *,
    prompt_tokens: torch.Tensor,
    reference_tokens: torch.Tensor,
    top5_tokens: torch.Tensor,
    kv_cache: list,
    page_table: torch.Tensor,
    max_batch_size: int = 1,
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
    prefill_output = execution_target.prefill_forward(
        prompt_tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=prompt_lens,
        empty_slots=empty_slots,
    )
    first_tokens = torch.argmax(prefill_output, dim=-1).view(-1).tolist()
    predicted_tokens_per_user = [[int(token)] for token in first_tokens]

    logger.info(f"Teacher forcing: decoding {num_target - 1} tokens")
    for step in range(1, num_target):
        ground_truth_token = reference_tokens[prompt_len + step - 1]
        decode_token = torch.full((batch_size,), ground_truth_token, dtype=torch.long)
        current_pos = torch.full((batch_size,), prompt_len + step - 1, dtype=torch.long)
        logits, _ = execution_target.decode_forward(
            decode_token,
            current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            read_from_device=True,
        )

        next_tokens = torch.argmax(logits[:, -1, :], dim=-1).view(-1).tolist()
        for user_id, token in enumerate(next_tokens):
            predicted_tokens_per_user[user_id].append(int(token))

    return TeacherForceResult(
        predicted_tokens=predicted_tokens_per_user[0],
        predicted_tokens_per_user=predicted_tokens_per_user,
        reference_top5=top5_tokens[:num_target],
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
    pipeline_readback: bool = False,
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
    )

    start_time = time.perf_counter()
    prefill_output = execution_target.prefill_forward(tokens, **prefill_kwargs, sampling_params=sampling_params)
    _synchronize_target(execution_target)
    prefill_time = time.perf_counter() - start_time

    first_token = prefill_output[0] if isinstance(prefill_output, tuple) else torch.argmax(prefill_output, dim=-1)
    first_token = first_token.view(-1)[:batch_size].detach().cpu()
    generated_token_ids = [[int(token)] for token in first_token.tolist()]

    current_tokens = torch.zeros(max_batch_size, dtype=torch.long)
    current_tokens[:batch_size] = first_token
    current_pos = torch.full((max_batch_size,), -1, dtype=torch.long)
    current_pos[:batch_size] = prompt_lens[:batch_size]

    compile_time = None
    decode_times = []
    sampled_decode_start = None
    pending_host_output = None
    pending_read_events = None

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

        if can_pipeline_readback:
            host_output, read_events = _submit_decode_read(execution_target, decode_output)
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
            pending_host_output = host_output
            pending_read_events = read_events

        if sampling_params is not None and iteration == 0:
            _synchronize_target(execution_target)
        elapsed = time.perf_counter() - start_time

        if iteration == 0:
            compile_time = elapsed
        elif sampling_params is None:
            decode_times.append(elapsed)

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
        if sampled_decode_start is not None:
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
    )
