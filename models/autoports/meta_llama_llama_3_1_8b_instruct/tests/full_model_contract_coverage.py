# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bounded exact-weight coverage for the public full-model generator contract.

The near-context instance first records all public chunk/page planning, then
runs the real one-layer decoder and fills the complete paged cache for a
131071-token non-divisible prompt.  A separate identity-decoder boundary case
proves that a full 131072-token prompt can produce its first token without an
extra cache position.  Short exact-real cases cover the terminal, sampler,
fixed 32 slots, inactive rows, traced decode, reset, and trace reuse.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.generator import _first_device_to_torch, build_generator
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.model import (
    DEFAULT_NUM_BLOCKS,
    DEFAULT_PREFILL_CHUNK_SIZE,
    HF_CONTEXT_LENGTH,
)
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device

MODEL_DIR = Path("models/autoports/meta_llama_llama_3_1_8b_instruct")
LONG_LOGICAL_LENGTH = HF_CONTEXT_LENGTH - 1
PAGE_SIZE = 64
FIXED_SAMPLING_SLOTS = 32


class _RecordingIdentityLayer:
    """Record public prefill planning while bypassing only decoder math/cache."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def prefill_forward(
        self,
        hidden,
        key_cache,
        value_cache,
        *,
        page_table,
        prompt_lens,
        chunk_start_idx,
        chunk_page_table,
    ):
        self.calls.append(
            {
                "chunk_start": int(chunk_start_idx),
                "hidden_shape": list(hidden.shape),
                "prompt_lens": list(prompt_lens),
                "full_page_table_shape": list(page_table.shape),
                "chunk_page_table_shape": None if chunk_page_table is None else list(chunk_page_table.shape),
                "key_cache_is_exact_buffer": key_cache is not None,
                "value_cache_is_exact_buffer": value_cache is not None,
            }
        )
        return hidden


def _tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _sampled_tokens(generator, sampled) -> list[int]:
    return [int(value) for value in generator._sampled_tokens_to_torch(sampled).tolist()]


def _synchronize(mesh_device) -> None:
    ttnn.synchronize_device(mesh_device)


def _mixed_prompt_batch(generator) -> tuple[torch.Tensor, list[int]]:
    lengths = [
        128,
        2,
        3,
        5,
        7,
        8,
        9,
        15,
        16,
        17,
        31,
        32,
        33,
        47,
        48,
        49,
        63,
        64,
        65,
        66,
        79,
        80,
        81,
        95,
        96,
        97,
        111,
        112,
        113,
        127,
        23,
        128,
    ]
    tokens = torch.zeros((FIXED_SAMPLING_SLOTS, max(lengths)), dtype=torch.long)
    bos = int(generator.tokenizer.bos_token_id)
    for row, length in enumerate(lengths):
        tokens[row, 0] = bos
        if length > 1:
            tokens[row, 1:length] = 42 + (row % 17)
    # The same prompt at distant physical sampling slots proves that fixed-slot
    # batch position does not perturb terminal logits or decode tokens.
    tokens[-1] = tokens[0]
    return tokens, lengths


def _collect_long_planning(generator, mesh_device, kv_cache) -> dict:
    original_layer = generator.model.layers[0]
    recording_layer = _RecordingIdentityLayer(original_layer)
    original_prefill_copy = generator._copy_prefill_host_to_device
    original_logits_to_torch = generator._local_logits_to_torch
    page_records: list[torch.Tensor] = []
    host_logit_conversions = 0

    def record_prefill_copy(host: torch.Tensor, device, *, dtype):
        if dtype == ttnn.int32:
            page_records.append(host.clone())
        return original_prefill_copy(host, device, dtype=dtype)

    def reject_host_logits(_logits):
        nonlocal host_logit_conversions
        host_logit_conversions += 1
        raise AssertionError("near-context device prefill attempted host-logit materialization")

    generator.model.layers[0] = recording_layer
    generator._copy_prefill_host_to_device = record_prefill_copy
    generator._local_logits_to_torch = reject_host_logits
    try:
        prompt = torch.full((1, LONG_LOGICAL_LENGTH), 42, dtype=torch.long)
        prompt[0, 0] = int(generator.tokenizer.bos_token_id)
        page_table = generator._make_page_table([LONG_LOGICAL_LENGTH])
        generator.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0, active_batch=1)

        started = time.perf_counter()
        sampled = generator.prefill_forward(
            prompt,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[LONG_LOGICAL_LENGTH],
            sampling_mode="device",
            enable_trace=False,
        )
        _synchronize(mesh_device)
        elapsed = time.perf_counter() - started
        physical_sample_shape = list(sampled.shape)
        samples = _sampled_tokens(generator, sampled)
    finally:
        generator.model.layers[0] = original_layer
        generator._copy_prefill_host_to_device = original_prefill_copy
        generator._local_logits_to_torch = original_logits_to_torch

    expected_starts = list(range(0, HF_CONTEXT_LENGTH, DEFAULT_PREFILL_CHUNK_SIZE))
    observed_starts = [call["chunk_start"] for call in recording_layer.calls]
    assert observed_starts == expected_starts
    assert len(recording_layer.calls) == 64
    assert all(call["hidden_shape"] == [1, 1, 2048, 4096] for call in recording_layer.calls)
    assert all(call["prompt_lens"] == [LONG_LOGICAL_LENGTH] for call in recording_layer.calls)
    assert recording_layer.calls[0]["chunk_page_table_shape"] is None
    assert all(call["chunk_page_table_shape"] == [1, 32] for call in recording_layer.calls[1:])

    assert len(page_records) == 64
    full_page_table = page_records[0]
    assert list(full_page_table.shape) == [1, DEFAULT_NUM_BLOCKS]
    assert torch.equal(full_page_table[0], torch.arange(DEFAULT_NUM_BLOCKS, dtype=torch.int32))
    for chunk_index, chunk_table in enumerate(page_records[1:], start=1):
        first_page = chunk_index * (DEFAULT_PREFILL_CHUNK_SIZE // PAGE_SIZE)
        assert torch.equal(
            chunk_table[0],
            torch.arange(first_page, first_page + DEFAULT_PREFILL_CHUNK_SIZE // PAGE_SIZE, dtype=torch.int32),
        )
    assert host_logit_conversions == 0
    assert physical_sample_shape == [1, 1, 1, FIXED_SAMPLING_SLOTS]
    assert len(samples) == 1

    return {
        "status": "pass",
        "logical_prompt_length": LONG_LOGICAL_LENGTH,
        "advertised_context_length": HF_CONTEXT_LENGTH,
        "non_divisible_by_chunk": LONG_LOGICAL_LENGTH % DEFAULT_PREFILL_CHUNK_SIZE != 0,
        "chunk_size": DEFAULT_PREFILL_CHUNK_SIZE,
        "chunk_count": len(observed_starts),
        "chunk_starts_first": observed_starts[:4],
        "chunk_starts_last": observed_starts[-4:],
        "exact_chunk_start_sequence": observed_starts == expected_starts,
        "full_page_table_shape": list(full_page_table.shape),
        "allocated_physical_pages": int(full_page_table[0].ge(0).sum().item()),
        "chunk_page_table_shape": [1, 32],
        "chunk_page_tables_exact_and_contiguous": True,
        "padded_prompt_lens_exact": True,
        "physical_sampling_tensor_shape": physical_sample_shape,
        "fixed_sampling_slot_count": FIXED_SAMPLING_SLOTS,
        "caller_visible_active_tokens": len(samples),
        "active_sample": samples[0],
        "host_logit_conversions": host_logit_conversions,
        "full_chunk_vocab_host_logits_materialized": False,
        "elapsed_seconds": elapsed,
        "coverage_label": "recording identity decoder; exact embedding/final norm/LM head/device sampler",
        "decoder_math_executed": False,
        "kv_cache_filled": False,
        "limitation": (
            "The 131071-token run proves public generator chunk/page/position planning and the selected-token terminal "
            "boundary, but deliberately bypasses decoder math and cache fill to keep runtime bounded."
        ),
    }


def _cache_page_stats(cache, physical_page: int) -> dict:
    page = ttnn.slice(
        cache,
        [physical_page, 0, 0, 0],
        [physical_page + 1, cache.shape[1], PAGE_SIZE, cache.shape[3]],
        [1, 1, 1, 1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    host = _first_device_to_torch(page).to(torch.float32)
    ttnn.deallocate(page)
    return {
        "shape_first_tp_rank": list(host.shape),
        "nonzero_elements_first_tp_rank": int(torch.count_nonzero(host).item()),
        "sha256_first_tp_rank": _tensor_sha256(host),
        "last_valid_position_nonzero": bool(torch.count_nonzero(host[..., PAGE_SIZE - 2, :]).item()),
        "padding_position_nonzero": bool(torch.count_nonzero(host[..., PAGE_SIZE - 1, :]).item()),
    }


def _collect_real_near_context(generator, mesh_device, kv_cache) -> dict:
    """Fill the real one-layer cache through the public high-level path."""

    generator.reset()
    prompt = [42] * LONG_LOGICAL_LENGTH
    prompt[0] = int(generator.tokenizer.bos_token_id)
    started = time.perf_counter()
    output = generator.generate(
        prompt,
        1,
        enable_trace=False,
        sampling_mode="device",
        top_k=1,
        top_p=0.0,
        temperature=1.0,
    )
    _synchronize(mesh_device)
    elapsed = time.perf_counter() - started
    page_table = generator._page_table_host
    last_physical_page = int(page_table[0, DEFAULT_NUM_BLOCKS - 1].item())
    assert last_physical_page == DEFAULT_NUM_BLOCKS - 1
    key_stats = _cache_page_stats(kv_cache[0][0], last_physical_page)
    value_stats = _cache_page_stats(kv_cache[0][1], last_physical_page)
    assert len(output) == 1
    assert page_table[0].ge(0).sum().item() == DEFAULT_NUM_BLOCKS
    assert key_stats["nonzero_elements_first_tp_rank"] > 0
    assert value_stats["nonzero_elements_first_tp_rank"] > 0
    assert key_stats["last_valid_position_nonzero"]
    assert value_stats["last_valid_position_nonzero"]
    return {
        "status": "pass",
        "coverage_label": "exact weights, real one-layer decoder, real full paged-cache fill, public generate",
        "logical_prompt_length": LONG_LOGICAL_LENGTH,
        "non_divisible_by_chunk": LONG_LOGICAL_LENGTH % DEFAULT_PREFILL_CHUNK_SIZE != 0,
        "chunk_count": HF_CONTEXT_LENGTH // DEFAULT_PREFILL_CHUNK_SIZE,
        "allocated_physical_pages": int(page_table[0].ge(0).sum().item()),
        "last_physical_page": last_physical_page,
        "generated_token": output[0],
        "key_cache_last_page": key_stats,
        "value_cache_last_page": value_stats,
        "padding_slot_disposition": (
            "The final padded slot may be written by the fixed-shape cache-fill kernel. Logical prompt masking "
            "excludes it during prefill, and a subsequent decode at that position overwrites it before attention."
        ),
        "elapsed_seconds": elapsed,
    }


def _collect_exact_max_prompt_boundary(generator, mesh_device, kv_cache) -> dict:
    """Prove max-context prefill can return one token without a decode slot."""

    original_layer = generator.model.layers[0]
    recording_layer = _RecordingIdentityLayer(original_layer)
    generator.model.layers[0] = recording_layer
    try:
        generator.reset()
        prompt = [42] * HF_CONTEXT_LENGTH
        prompt[0] = int(generator.tokenizer.bos_token_id)
        output = generator.generate(
            prompt,
            1,
            enable_trace=False,
            sampling_mode="device",
            top_k=1,
            top_p=0.0,
            temperature=1.0,
        )
        _synchronize(mesh_device)
        page_table = generator._page_table_host
    finally:
        generator.model.layers[0] = original_layer
    assert len(output) == 1
    assert len(recording_layer.calls) == HF_CONTEXT_LENGTH // DEFAULT_PREFILL_CHUNK_SIZE
    assert recording_layer.calls[-1]["prompt_lens"] == [HF_CONTEXT_LENGTH]
    assert page_table[0].ge(0).sum().item() == DEFAULT_NUM_BLOCKS
    return {
        "status": "pass",
        "coverage_label": "public generate maximum-prompt boundary; recording identity decoder",
        "logical_prompt_length": HF_CONTEXT_LENGTH,
        "requested_new_tokens": 1,
        "processed_position_horizon": HF_CONTEXT_LENGTH,
        "chunk_count": len(recording_layer.calls),
        "allocated_physical_pages": int(page_table[0].ge(0).sum().item()),
        "generated_token": output[0],
        "decoder_math_executed": False,
        "reason": "The first sampled token is produced by prefill and requires no position beyond the prompt.",
    }


def _prefill_host_logits(generator, tokens, lengths, page_table, kv_cache, mesh_device):
    started = time.perf_counter()
    logits = generator.prefill_forward(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=lengths,
        sampling_mode="host",
    )
    _synchronize(mesh_device)
    return logits, time.perf_counter() - started


def _prefill_device_tokens(generator, tokens, lengths, page_table, kv_cache, mesh_device, *, enable_trace=True):
    sampled = generator.prefill_forward(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=lengths,
        sampling_mode="device",
        enable_trace=enable_trace,
    )
    _synchronize(mesh_device)
    return _sampled_tokens(generator, sampled)


def _decode_device_tokens(generator, tokens, lengths, page_table, kv_cache, mesh_device, *, enable_trace=True):
    sampled = generator.decode_forward(
        torch.tensor(tokens, dtype=torch.long).reshape(-1, 1),
        torch.tensor(lengths, dtype=torch.long),
        page_table=page_table,
        kv_cache=kv_cache,
        sampling_mode="device",
        enable_trace=enable_trace,
        active_batch=len(lengths),
    )
    _synchronize(mesh_device)
    return _sampled_tokens(generator, sampled)


def _collect_exact_real_coverage(generator, mesh_device, kv_cache) -> dict:
    tokens, lengths = _mixed_prompt_batch(generator)
    # Reserve both the initial decode write at ``length`` and the steady
    # feedback write at ``length + 1``. Supplying only ``length + 1`` tokens
    # leaves page-boundary prompts with a -1 page for the second write.
    page_table = generator._make_page_table([length + 2 for length in lengths])

    generator.reset()
    logits_one, logits_one_s = _prefill_host_logits(generator, tokens, lengths, page_table, kv_cache, mesh_device)
    generator.reset()
    logits_two, logits_two_s = _prefill_host_logits(generator, tokens, lengths, page_table, kv_cache, mesh_device)

    assert list(logits_one.shape) == [32, 1, generator.model.vocab_size]
    assert torch.equal(logits_one, logits_two)
    assert torch.equal(logits_one[0], logits_one[31])
    host_argmax = [int(value) for value in logits_one[:, 0, :].argmax(dim=-1).tolist()]

    generator.reset()
    generator.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0, active_batch=32)

    # Isolate terminal versus sampler behavior before tracing. Capture the
    # exact eager logits object consumed by Sampling1D, then compare its host
    # argmax and the common force-argmax control to the split result.
    captured_logits = {}
    original_split = generator.model.sample_greedy_split

    def capture_split(logits, **kwargs):
        captured_logits["value"] = logits
        return original_split(logits, **kwargs)

    generator.model.sample_greedy_split = capture_split
    try:
        eager_prefill_tokens = _prefill_device_tokens(
            generator,
            tokens,
            lengths,
            page_table,
            kv_cache,
            mesh_device,
            enable_trace=False,
        )
    finally:
        generator.model.sample_greedy_split = original_split
    exact_device_logits = captured_logits["value"]
    eager_force_tokens = _sampled_tokens(generator, generator.model.sample_force_argmax(exact_device_logits))
    eager_logits_host = generator._local_logits_to_torch(exact_device_logits)[0, 0, :32, :]
    eager_logits_argmax = [int(value) for value in eager_logits_host.argmax(dim=-1).tolist()]

    # Prove the exact local-argmax/global-rank-candidate reduction on the same
    # native LM-head tensor consumed by the public device path.
    sampler = generator.model.sampler
    if eager_prefill_tokens != host_argmax:
        print(
            json.dumps(
                {
                    "host_argmax": host_argmax,
                    "eager_split": eager_prefill_tokens,
                    "force_argmax": eager_force_tokens,
                },
                indent=2,
            ),
            flush=True,
        )
    assert eager_logits_argmax == host_argmax
    assert eager_force_tokens == host_argmax
    assert eager_prefill_tokens == host_argmax

    # Capture only the canonical split sampler against the exact persistent
    # native LM-head tensor. Two identical replays must be semantically greedy
    # before combining this graph with model/prefill traces.
    standalone_sample_output = generator._prefill_sampled
    sampler_k, sampler_p, sampler_temp = generator._ensure_sampling_params()
    original_split(
        exact_device_logits,
        k=sampler_k,
        p=sampler_p,
        temp=sampler_temp,
        tt_out_tok=standalone_sample_output,
    )
    _synchronize(mesh_device)
    standalone_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    original_split(
        exact_device_logits,
        k=sampler_k,
        p=sampler_p,
        temp=sampler_temp,
        tt_out_tok=standalone_sample_output,
    )
    ttnn.end_trace_capture(mesh_device, standalone_trace_id, cq_id=0)
    _synchronize(mesh_device)
    standalone_trace_tokens = []
    try:
        for _ in range(2):
            ttnn.execute_trace(mesh_device, standalone_trace_id, cq_id=0, blocking=False)
            _synchronize(mesh_device)
            standalone_trace_tokens.append(_sampled_tokens(generator, standalone_sample_output))
    finally:
        ttnn.release_trace(mesh_device, standalone_trace_id)
    if standalone_trace_tokens[0] != standalone_trace_tokens[1]:
        print(json.dumps({"standalone_sampler_trace_tokens": standalone_trace_tokens}, indent=2), flush=True)
    assert standalone_trace_tokens[0] == host_argmax
    assert standalone_trace_tokens[1] == host_argmax

    # First isolate request reset/cache refill from trace replay. If these
    # eager results are stable but the traced pair below is not, the fault is
    # confined to captured model/sampler buffer lifetime or replay state.
    eager_decode_requests = []
    for request_index in range(2):
        generator.reset()
        eager_request_prefill = _prefill_device_tokens(
            generator,
            tokens,
            lengths,
            page_table,
            kv_cache,
            mesh_device,
            enable_trace=False,
        )
        eager_decode_requests.append(
            _decode_device_tokens(
                generator,
                eager_request_prefill,
                lengths,
                page_table,
                kv_cache,
                mesh_device,
                enable_trace=False,
            )
        )
        if request_index == 0:
            _decode_device_tokens(
                generator,
                eager_decode_requests[-1],
                [length + 1 for length in lengths],
                page_table,
                kv_cache,
                mesh_device,
                enable_trace=False,
            )
    if eager_decode_requests[0] != eager_decode_requests[1]:
        print(json.dumps({"eager_decode_requests": eager_decode_requests}, indent=2), flush=True)
    assert eager_decode_requests[0] == eager_decode_requests[1]

    # Comparison-only control: run the identical public trace lifecycle with
    # the common full-vocabulary force-argmax sampler. This identifies whether
    # prefill/decode trace interleaving is generally unstable or specifically
    # interacts with the split sampler's two candidate all-gathers.
    def force_argmax_control(logits, *, k, p, temp, tt_out_tok=None):
        return generator.model.sample_force_argmax(logits, tt_out_tok=tt_out_tok)

    force_argmax_trace_requests = []
    generator.model.sample_greedy_split = force_argmax_control
    try:
        for request_index in range(2):
            generator.reset()
            force_prefill = _prefill_device_tokens(
                generator,
                tokens,
                lengths,
                page_table,
                kv_cache,
                mesh_device,
            )
            force_argmax_trace_requests.append(
                _decode_device_tokens(
                    generator,
                    force_prefill,
                    lengths,
                    page_table,
                    kv_cache,
                    mesh_device,
                )
            )
            if request_index == 0:
                force_steady = generator.decode_forward(
                    None,
                    None,
                    page_table=None,
                    kv_cache=kv_cache,
                    sampling_mode="device",
                    enable_trace=True,
                    active_batch=32,
                )
                _synchronize(mesh_device)
                _sampled_tokens(generator, force_steady)
    finally:
        generator._release_all_traces()
        generator._decode_warm_key = None
        generator.model.sample_greedy_split = original_split
    if force_argmax_trace_requests[0] != force_argmax_trace_requests[1]:
        print(json.dumps({"force_argmax_trace_requests": force_argmax_trace_requests}, indent=2), flush=True)
    assert force_argmax_trace_requests[0] == force_argmax_trace_requests[1]

    generator.reset()
    prefill_tokens_one = _prefill_device_tokens(generator, tokens, lengths, page_table, kv_cache, mesh_device)
    captures_before = generator.trace_stats["captures"]
    decode_tokens_one = _decode_device_tokens(
        generator,
        prefill_tokens_one,
        lengths,
        page_table,
        kv_cache,
        mesh_device,
    )
    captures_after_first = generator.trace_stats["captures"]
    steady_sample = generator.decode_forward(
        None,
        None,
        page_table=None,
        kv_cache=kv_cache,
        sampling_mode="device",
        enable_trace=True,
        active_batch=32,
    )
    _synchronize(mesh_device)
    steady_tokens = _sampled_tokens(generator, steady_sample)

    generator.reset()
    prefill_tokens_two = _prefill_device_tokens(generator, tokens, lengths, page_table, kv_cache, mesh_device)
    decode_tokens_two = _decode_device_tokens(
        generator,
        prefill_tokens_two,
        lengths,
        page_table,
        kv_cache,
        mesh_device,
    )
    captures_after_second = generator.trace_stats["captures"]

    greedy_mismatch_slots = [
        slot
        for slot, (host_token, device_token) in enumerate(zip(host_argmax, prefill_tokens_one))
        if host_token != device_token
    ]
    if greedy_mismatch_slots:
        print(
            json.dumps(
                {
                    "batch_32_greedy_mismatch_slots": greedy_mismatch_slots,
                    "host_argmax": host_argmax,
                    "device_prefill_tokens": prefill_tokens_one,
                },
                indent=2,
            ),
            flush=True,
        )
    assert prefill_tokens_one == host_argmax
    assert prefill_tokens_one == prefill_tokens_two
    if decode_tokens_one != decode_tokens_two:
        print(
            json.dumps(
                {
                    "decode_tokens_first_request": decode_tokens_one,
                    "decode_tokens_repeat_request": decode_tokens_two,
                    "trace_stats": generator.trace_stats,
                },
                indent=2,
            ),
            flush=True,
        )
    assert decode_tokens_one == decode_tokens_two
    assert prefill_tokens_one[0] == prefill_tokens_one[31]
    assert decode_tokens_one[0] == decode_tokens_one[31]
    assert captures_after_first == captures_before + 1
    assert captures_after_second == captures_after_first

    generator.reset()
    inactive_tokens = tokens[:31]
    inactive_lengths = lengths[:31]
    inactive_page_table = generator._make_page_table([length + 1 for length in inactive_lengths])
    generator.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0, active_batch=31)
    inactive_prefill_one = _prefill_device_tokens(
        generator,
        inactive_tokens,
        inactive_lengths,
        inactive_page_table,
        kv_cache,
        mesh_device,
    )
    inactive_decode_one = _decode_device_tokens(
        generator,
        inactive_prefill_one[:31],
        inactive_lengths,
        inactive_page_table,
        kv_cache,
        mesh_device,
    )
    captures_after_first_inactive = generator.trace_stats["captures"]
    generator.reset()
    inactive_prefill_two = _prefill_device_tokens(
        generator,
        inactive_tokens,
        inactive_lengths,
        inactive_page_table,
        kv_cache,
        mesh_device,
    )
    inactive_decode_two = _decode_device_tokens(
        generator,
        inactive_prefill_two[:31],
        inactive_lengths,
        inactive_page_table,
        kv_cache,
        mesh_device,
    )
    captures_after_inactive = generator.trace_stats["captures"]

    assert len(inactive_prefill_one) == len(inactive_decode_one) == FIXED_SAMPLING_SLOTS
    assert inactive_prefill_one == inactive_prefill_two
    assert inactive_decode_one == inactive_decode_two
    assert inactive_page_table[31].eq(-1).all()
    assert captures_after_first_inactive == captures_after_first + 1
    assert captures_after_inactive == captures_after_first_inactive

    logits_sha = _tensor_sha256(logits_one)
    return {
        "status": "pass",
        "coverage_label": "exact weights, exact real one-layer decoder/cache/terminal/sampler",
        "real_decoder_layers": generator.model.num_layers,
        "kv_cache_identity_reused": kv_cache is generator._ensure_kv_cache(),
        "batch_32": {
            "mixed_prompt_lengths": lengths,
            "non_aligned_prompt_count": sum(length % 128 != 0 for length in lengths),
            "terminal_logits_shape": list(logits_one.shape),
            "terminal_logits_sha256": logits_sha,
            "repeat_terminal_logits_sha256": _tensor_sha256(logits_two),
            "repeat_logits_bit_exact": torch.equal(logits_one, logits_two),
            "distant_duplicate_slots": [0, 31],
            "duplicate_slot_logits_bit_exact": torch.equal(logits_one[0], logits_one[31]),
            "host_argmax_matches_device_prefill_greedy": host_argmax == prefill_tokens_one,
            "same_native_logits_force_argmax_matches_host": eager_force_tokens == host_argmax,
            "same_native_logits_split_greedy_matches_host": eager_prefill_tokens == host_argmax,
            "split_local_argmax_matches_full_force_argmax": eager_prefill_tokens == eager_force_tokens,
            "split_greedy_strategy": "local argmax + one packed FP32 rank-candidate gather + device global argmax",
            "sampler_local_vocab_width": generator.model.local_vocab_size,
            "stochastic_topk_padded_width": int(sampler._local_indices.shape[-1]),
            "sampler_pad_to_power_of_2": sampler.config.pad_to_power_of_2,
            "prefill_tokens": prefill_tokens_one,
            "repeat_prefill_tokens_exact": prefill_tokens_one == prefill_tokens_two,
            "decode_tokens": decode_tokens_one,
            "repeat_decode_tokens_exact": decode_tokens_one == decode_tokens_two,
            "duplicate_slot_prefill_token_exact": prefill_tokens_one[0] == prefill_tokens_one[31],
            "duplicate_slot_decode_token_exact": decode_tokens_one[0] == decode_tokens_one[31],
            "steady_public_replay_tokens": steady_tokens,
            "host_terminal_logits_elements": logits_one.numel(),
            "host_terminal_logits_only": True,
            "first_host_prefill_seconds": logits_one_s,
            "repeat_host_prefill_seconds": logits_two_s,
        },
        "inactive_row": {
            "active_batch": 31,
            "fixed_sampling_slot_count": len(inactive_prefill_one),
            "inactive_slot": 31,
            "inactive_page_table_all_negative": bool(inactive_page_table[31].eq(-1).all()),
            "inactive_prefill_token": inactive_prefill_one[31],
            "inactive_decode_token": inactive_decode_one[31],
            "repeat_prefill_tokens_exact": inactive_prefill_one == inactive_prefill_two,
            "repeat_decode_tokens_exact": inactive_decode_one == inactive_decode_two,
        },
        "trace_reuse": {
            "captures_before_first_decode": captures_before,
            "captures_after_first_decode": captures_after_first,
            "captures_after_repeat_reset_request": captures_after_second,
            "captures_after_first_incompatible_active_batch": captures_after_first_inactive,
            "captures_after_active_batch_change": captures_after_inactive,
            "reset_reused_compatible_trace": captures_after_second == captures_after_first,
            "active_batch_change_recaptured_once": captures_after_first_inactive == captures_after_first + 1,
            "inactive_repeat_reused_compatible_trace": captures_after_inactive == captures_after_first_inactive,
            "public_none_none_steady_replay_executed": True,
        },
        "trace_stats": dict(generator.trace_stats),
    }


def collect(model_dir: Path, model_path: Path, output: Path) -> dict:
    mesh_device = open_readiness_mesh_device("P300", "FABRIC_1D_RING")
    generator = None
    try:
        # Keep the near-context proof at its actual active batch. Expanding one
        # logical request into 31 inactive embedding rows would add no coverage
        # and would move roughly 17 GiB of hidden-state data per chunk sequence.
        generator = build_generator(
            model_dir,
            mesh_device,
            model_path=model_path,
            max_batch_size=1,
            max_context_len=HF_CONTEXT_LENGTH,
            num_blocks=DEFAULT_NUM_BLOCKS,
            prefill_chunk_size=DEFAULT_PREFILL_CHUNK_SIZE,
            override_num_layers=1,
        )
        kv_cache = generator._ensure_kv_cache()
        long_planning = _collect_long_planning(generator, mesh_device, kv_cache)
        long_real_cache = _collect_real_near_context(generator, mesh_device, kv_cache)
        max_prompt_boundary = _collect_exact_max_prompt_boundary(generator, mesh_device, kv_cache)
        generator.teardown()
        del kv_cache
        del generator
        generator = None
        gc.collect()

        # The separate exact-real instance exercises all 32 fixed slots,
        # inactive-row behavior, cache, terminal, sampler, and trace reuse.
        generator = build_generator(
            model_dir,
            mesh_device,
            model_path=model_path,
            max_batch_size=FIXED_SAMPLING_SLOTS,
            max_context_len=256,
            num_blocks=64,
            prefill_chunk_size=128,
            override_num_layers=1,
        )
        kv_cache = generator._ensure_kv_cache()
        result = {
            "status": "pass",
            "mesh": "P300 1x4 FABRIC_1D_RING TP=4",
            "exact_weight_snapshot": str(model_path.resolve()),
            "configured_layers": generator.model.num_layers,
            "long_instance": {
                "batch": 1,
                "context": HF_CONTEXT_LENGTH,
                "cache_blocks": DEFAULT_NUM_BLOCKS,
                "prefill_chunk": DEFAULT_PREFILL_CHUNK_SIZE,
            },
            "batch_instance": {"batch": 32, "context": 256, "cache_blocks": 64, "prefill_chunk": 128},
            "long_public_path_planning": long_planning,
            "long_real_decoder_cache": long_real_cache,
            "maximum_prompt_boundary": max_prompt_boundary,
            "exact_real_contract": _collect_exact_real_coverage(generator, mesh_device, kv_cache),
            "limitations": [
                "Near-context decoder/cache coverage uses one exact decoder layer; the all-32-layer gates remain the "
                "full-stack numerical and performance evidence.",
                "The exact 131072-token maximum-prompt boundary uses the labeled identity decoder because the real "
                "non-divisible 131071-token run already fills and inspects the final physical cache page.",
                "Host logits are read only for 32 selected terminal rows in the explicit compatibility comparison; "
                "the near-context path reads no logits and the measured serving path remains device sampled.",
            ],
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh_device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.model_dir, args.model_path, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
