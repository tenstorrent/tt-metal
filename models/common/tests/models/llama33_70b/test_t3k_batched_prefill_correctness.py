# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Direct W6 correctness gate for production Llama-3.3-70B on T3K.

This module deliberately contains no fake tensors or mocked execution. It is
collection-safe when T3K is not selected; a configured T3K gate strictly
requires model assets and exercises the production executor, compiler
registries, traces, and paged KV allocation.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import ttnn

if os.environ.get("MESH_DEVICE", "").strip() != "T3K":
    pytest.skip("W6 requires MESH_DEVICE=T3K", allow_module_level=True)

from huggingface_hub import snapshot_download

from models.common.sampling import SamplingParams
from models.common.tests.demos.llama33_70b.demo import create_executor, create_model, lazy_weight_cache_dir_for_demo
from models.common.tests.models.llama33_70b.logits_oracle import assert_rowwise_logits_parity

_HF_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
_BLOCK_SIZE = 32
_PROMPT_LEN = 128
_MAX_BATCH_SIZE = 16
_MAX_SEQ_LEN = 4096
_BLOCK_COUNT = _MAX_BATCH_SIZE * (_MAX_SEQ_LEN // _BLOCK_SIZE)
_RESIDENT_SLOT = _MAX_BATCH_SIZE - 1
_RESUME_SLOT = _MAX_BATCH_SIZE - 2
_RESIDENT_BLOCK_START = _RESIDENT_SLOT * (_MAX_SEQ_LEN // _BLOCK_SIZE)
_STALE_BLOCK = 750
_LOGITS_MIN_ROW_PCC = float(os.environ.get("W6_LOGITS_MIN_ROW_PCC", "0.997"))
_LOGITS_MAX_ABS = float(os.environ.get("W6_LOGITS_MAX_ABS", "1.0"))
_LOGITS_TOPK = int(os.environ.get("W6_LOGITS_TOPK", "5"))
_LOGITS_MIN_TOPK_OVERLAP = int(os.environ.get("W6_LOGITS_MIN_TOPK_OVERLAP", "4"))
_LOGITS_MAX_TOP1_MISMATCHES = int(os.environ.get("W6_LOGITS_MAX_TOP1_MISMATCHES", "1"))
_LOGITS_MAX_ISCLOSE_FAILURE_FRACTION = float(os.environ.get("W6_LOGITS_MAX_ISCLOSE_FAILURE_FRACTION", "0.005"))
_LOGITS_ATOL = float(os.environ.get("W6_LOGITS_ATOL", "0.25"))
_DECODE_MIN_ROW_PCC = float(os.environ.get("W6_DECODE_MIN_ROW_PCC", "0.99"))
_DECODE_MAX_ABS = float(os.environ.get("W6_DECODE_MAX_ABS", "1.25"))


def _mesh_parameter() -> dict:
    return {
        "mesh_shape": (1, 8),
        # This gate captures the expanded strict coverage set, whose cumulative
        # size exceeds the model's fixed CI budget. Zero selects TTNN's dynamic
        # runtime allocation instead of coupling correctness to capture order.
        "trace_region_size": 0,
        "num_command_queues": 1,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    }


pytestmark = pytest.mark.parametrize(
    "ttnn_mesh_device",
    [_mesh_parameter()],
    indirect=True,
    scope="module",
    ids=["T3K"],
)


@pytest.fixture(scope="module")
def local_hf_model(model_location_generator) -> str:
    requested = os.environ.get("HF_MODEL", _HF_MODEL)
    located = model_location_generator(requested)
    if Path(str(located)).exists():
        return str(located)
    try:
        return snapshot_download(str(located), local_files_only=True)
    except Exception as error:
        pytest.fail(f"MESH_DEVICE=T3K requires local Llama-3.3-70B model assets: {error}", pytrace=False)


@pytest.fixture(scope="module")
def production_model(local_hf_model, ttnn_mesh_device):
    previous = os.environ.get("HF_MODEL")
    os.environ["HF_MODEL"] = local_hf_model
    cache_dir = lazy_weight_cache_dir_for_demo(ttnn_mesh_device, _HF_MODEL)
    try:
        yield create_model(
            ttnn_mesh_device,
            "accuracy",
            cache_dir,
            max_batch_size=_MAX_BATCH_SIZE,
            max_seq_len=_MAX_SEQ_LEN,
        )
    finally:
        if previous is None:
            os.environ.pop("HF_MODEL", None)
        else:
            os.environ["HF_MODEL"] = previous


def _page_table(*, offset: int = 0, stale_block: int | None = None) -> torch.Tensor:
    width = _MAX_SEQ_LEN // _BLOCK_SIZE
    table = torch.arange(_MAX_BATCH_SIZE * width, dtype=torch.int32).reshape(_MAX_BATCH_SIZE, width)
    # Compact active prefixes make the complete logical KV region one bounded
    # D2H slice while tails retain realistic scheduler-row capacity.
    table[:, : _PROMPT_LEN // _BLOCK_SIZE] = torch.arange(
        _MAX_BATCH_SIZE * (_PROMPT_LEN // _BLOCK_SIZE), dtype=torch.int32
    ).reshape(_MAX_BATCH_SIZE, _PROMPT_LEN // _BLOCK_SIZE)
    if offset:
        table = (table + offset) % _BLOCK_COUNT
    if stale_block is not None:
        table[:, _PROMPT_LEN // _BLOCK_SIZE :] = stale_block
    return table


def _tokens(rows: int, *, salt: int = 0) -> torch.Tensor:
    values = torch.arange(rows * _PROMPT_LEN, dtype=torch.long).reshape(rows, _PROMPT_LEN)
    return (values + 17 + salt) % 32000


def _prepared(
    executor,
    tokens,
    page_table,
    *,
    sampling=None,
    start_pos=None,
    slots=None,
    prompt_lens=None,
):
    return executor.prefill_runtime.prepare(
        tokens=tokens,
        page_table=page_table[: tokens.shape[0]],
        prompt_lens=(
            torch.full((tokens.shape[0],), tokens.shape[1], dtype=torch.long) if prompt_lens is None else prompt_lens
        ),
        start_pos=start_pos,
        empty_slots=list(range(tokens.shape[0])) if slots is None else slots,
        sampling_params=sampling,
    )


def _program_cache_entries(mesh_device) -> int:
    devices = mesh_device.get_devices() if hasattr(mesh_device, "get_devices") else (mesh_device,)
    return sum(device.num_program_cache_entries() for device in devices)


def _cache_slice(mesh_tensor, block_start: int, block_end: int) -> torch.Tensor:
    shards = []
    for shard in ttnn.get_device_tensors(mesh_tensor):
        shape = tuple(int(value) for value in shard.shape)
        sliced = ttnn.slice(shard, (block_start, 0, 0, 0), (block_end, shape[1], shape[2], shape[3]))
        shards.append(ttnn.to_torch(sliced).clone())
    return torch.cat(shards, dim=1)


def _kv_snapshot(kv_cache, *ranges: tuple[int, int]):
    return tuple(
        tuple(tuple(_cache_slice(tensor, start, end) for start, end in ranges) for tensor in layer)
        for layer in kv_cache
    )


def _assert_nested_close(actual, expected, *, atol: float, rtol: float) -> None:
    assert len(actual) == len(expected) > 0
    for actual_layer, expected_layer in zip(actual, expected):
        assert len(actual_layer) == len(expected_layer) > 0
        for actual_tensor, expected_tensor in zip(actual_layer, expected_layer):
            assert len(actual_tensor) == len(expected_tensor) > 0
            for actual_slice, expected_slice in zip(actual_tensor, expected_tensor):
                torch.testing.assert_close(actual_slice, expected_slice, atol=atol, rtol=rtol)


def _decode_logits(output):
    """Unpack the runtime's normalized ``(logits, log_probs)`` contract."""

    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("decode output must be a (logits, log_probs) tuple")
    logits, log_probs = output
    assert log_probs is None
    return logits


def _sampled_tokens(output):
    """Unpack the runtime's normalized ``(tokens, log_probs)`` contract."""

    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError("sampled prefill output must be a (tokens, log_probs) tuple")
    tokens, log_probs = output
    assert log_probs is None
    return tokens


def _run_sequential_oracle(model, tokens, page_table, resident_tokens, resident_table):
    executor = create_executor(model, traced=False, device_sampling_enabled=False)
    try:
        kv_cache = executor.allocate_kv_cache()
        resident_logits = executor.prefill_forward(
            resident_tokens,
            resident_table,
            kv_cache=kv_cache,
            prompt_lens=torch.tensor([_PROMPT_LEN]),
            empty_slots=[_RESIDENT_SLOT],
            execution=executor.eager_execution,
        )
        outputs = []
        for row in range(tokens.shape[0]):
            outputs.append(
                executor.prefill_forward(
                    tokens[row : row + 1],
                    page_table[row : row + 1],
                    kv_cache=kv_cache,
                    prompt_lens=torch.tensor([_PROMPT_LEN]),
                    empty_slots=[row],
                    execution=executor.eager_execution,
                )
            )
        active_logits = torch.cat(outputs, dim=0)
        active_decode_tokens, active_decode_start_pos, active_decode_page_table = _active_decode_inputs(
            page_table, resident_table
        )
        active_decode = _decode_logits(
            executor.decode_forward(
                active_decode_tokens,
                active_decode_start_pos,
                active_decode_page_table,
                kv_cache=kv_cache,
                execution=executor.eager_execution,
            )
        )[: tokens.shape[0]]
        decode_tokens, decode_start_pos, decode_page_table = _resident_decode_inputs(resident_logits, resident_table)
        resident_decode = _decode_logits(
            executor.decode_forward(
                decode_tokens,
                decode_start_pos,
                decode_page_table,
                kv_cache=kv_cache,
                execution=executor.eager_execution,
            )
        )[_RESIDENT_SLOT : _RESIDENT_SLOT + 1]
        return active_logits, active_decode, resident_decode
    finally:
        executor.cleanup()


def _run_batched_eager_oracle(model, tokens, page_table, resident_tokens, resident_table):
    """Run the same padded batch geometry as trace replay on an isolated cache."""

    executor = create_executor(model, traced=False, device_sampling_enabled=False)
    try:
        kv_cache = executor.allocate_kv_cache()
        executor.prefill_forward(
            resident_tokens,
            resident_table,
            kv_cache=kv_cache,
            prompt_lens=torch.tensor([_PROMPT_LEN]),
            empty_slots=[_RESIDENT_SLOT],
            execution=executor.eager_execution,
        )
        active_logits = executor.prefill_forward(
            tokens,
            page_table[: tokens.shape[0]],
            kv_cache=kv_cache,
            prompt_lens=torch.full((tokens.shape[0],), _PROMPT_LEN, dtype=torch.long),
            empty_slots=list(range(tokens.shape[0])),
            execution=executor.eager_execution,
        )
        kv_after_prefill = _kv_snapshot(
            kv_cache,
            (0, 60),
            (_RESIDENT_BLOCK_START, _RESIDENT_BLOCK_START + 4),
        )
        repeated_logits = executor.prefill_forward(
            tokens,
            page_table[: tokens.shape[0]],
            kv_cache=kv_cache,
            prompt_lens=torch.full((tokens.shape[0],), _PROMPT_LEN, dtype=torch.long),
            empty_slots=list(range(tokens.shape[0])),
            execution=executor.eager_execution,
        )
        repeated_kv = _kv_snapshot(
            kv_cache,
            (0, 60),
            (_RESIDENT_BLOCK_START, _RESIDENT_BLOCK_START + 4),
        )
        assert torch.equal(repeated_logits, active_logits)
        _assert_nested_close(repeated_kv, kv_after_prefill, atol=0.0, rtol=0.0)
        active_decode_tokens, active_decode_start_pos, active_decode_page_table = _active_decode_inputs(
            page_table, resident_table
        )
        active_decode = _decode_logits(
            executor.decode_forward(
                active_decode_tokens,
                active_decode_start_pos,
                active_decode_page_table,
                kv_cache=kv_cache,
                execution=executor.eager_execution,
            )
        )[: tokens.shape[0]]
        return active_logits, kv_after_prefill, active_decode
    finally:
        executor.cleanup()


def _active_decode_inputs(page_table, resident_table):
    """Build an identical 16-lane decode consumer for every populated cache."""

    decode_tokens = (torch.arange(_MAX_BATCH_SIZE, dtype=torch.long) + 313) % 32000
    decode_start_pos = torch.full((_MAX_BATCH_SIZE,), _PROMPT_LEN, dtype=torch.long)
    decode_page_table = _page_table()
    decode_page_table[:_RESIDENT_SLOT, :4] = page_table[:_RESIDENT_SLOT, :4]
    # The compact prompt mapping owns physical blocks 0..59. The default row-0
    # fifth block is 4, which aliases row 1's first prompt block; use a fresh
    # bounded region for the decode write at position 128.
    decode_page_table[:_RESIDENT_SLOT, 4] = torch.arange(800, 800 + _RESIDENT_SLOT, dtype=torch.int32)
    decode_page_table[_RESIDENT_SLOT] = resident_table[0]
    return decode_tokens, decode_start_pos, decode_page_table


def _resident_decode_inputs(resident_logits, resident_table):
    """Build the production 16-lane decode shape around the final resident lane."""

    decode_tokens = torch.zeros(_MAX_BATCH_SIZE, dtype=torch.long)
    decode_tokens[_RESIDENT_SLOT] = resident_logits.argmax(dim=-1).reshape(-1)[0]
    decode_start_pos = torch.zeros(_MAX_BATCH_SIZE, dtype=torch.long)
    decode_start_pos[_RESIDENT_SLOT] = _PROMPT_LEN
    decode_page_table = _page_table()
    decode_page_table[_RESIDENT_SLOT] = resident_table[0]
    return decode_tokens, decode_start_pos, decode_page_table


def _compile_registration_order(executor, kv_cache, page_table, capture_order, sampling_order):
    topk = SamplingParams(temperature=0.0, top_k=1, top_p=1.0)
    active = {15: _tokens(15), 16: _tokens(16, salt=3)}
    cases = {
        "logits": None,
        "topk": topk,
    }
    executor.warmup_model_decode(
        kv_cache=kv_cache,
        max_batch_size=_MAX_BATCH_SIZE,
        num_blocks=page_table.shape[-1],
        can_sample_on_device=True,
        enable_trace=False,
    )
    executor.warmup_model_prefill(kv_cache=kv_cache, can_sample_on_device=True, enable_trace=False)
    for active_rows in capture_order:
        for sampling_name in sampling_order:
            executor.compile_prefill(
                tokens=active[active_rows],
                page_table=page_table[:active_rows],
                kv_cache=kv_cache,
                prompt_lens=torch.full((active_rows,), _PROMPT_LEN, dtype=torch.long),
                empty_slots=list(range(active_rows)),
                sampling_params=cases[sampling_name],
                execution=executor.traced_prefill_execution,
            )
    # Cached/resumed and long fixed-chunk signatures are intentionally not
    # registered here: the production coordinator's configured 128/2048/4096
    # coverage below must own them, or their later strict replays must fail.
    executor.warmup_model_prefill(kv_cache=kv_cache, can_sample_on_device=True, enable_trace=True)
    executor.warmup_model_decode(
        kv_cache=kv_cache,
        max_batch_size=_MAX_BATCH_SIZE,
        num_blocks=page_table.shape[-1],
        can_sample_on_device=True,
        enable_trace=True,
    )
    return topk


@pytest.mark.parametrize("capture_order", [(16, 15), (15, 16)], ids=["16-15", "15-16"])
@pytest.mark.parametrize(
    "sampling_order",
    [("logits", "topk"), ("topk", "logits")],
    ids=["logits-topk", "topk-logits"],
)
def test_w6_active15_padded16_trace_correctness(
    production_model,
    ttnn_mesh_device,
    capture_order,
    sampling_order,
):
    stale_block = _STALE_BLOCK
    page_table = _page_table(stale_block=stale_block)
    resident_table = _page_table()[_RESIDENT_SLOT : _RESIDENT_SLOT + 1]
    resident_table[:, :4] = torch.arange(_RESIDENT_BLOCK_START, _RESIDENT_BLOCK_START + 4, dtype=torch.int32)
    tokens = _tokens(15)
    resident_tokens = _tokens(1, salt=101)
    expected_logits, expected_active_decode, expected_resident_decode = _run_sequential_oracle(
        production_model, tokens, page_table, resident_tokens, resident_table
    )
    batched_eager_logits, batched_eager_kv, batched_eager_active_decode = _run_batched_eager_oracle(
        production_model, tokens, page_table, resident_tokens, resident_table
    )

    executor = create_executor(production_model, traced=True, device_sampling_enabled=True, trace_mode="all")
    try:
        assert executor.config.trace.mode == "all"
        # Production Llama33 disables force-argmax, so argmax->top-k is not an
        # executable registration order for this candidate.
        assert not production_model.sampling.config.allow_force_argmax
        assert executor.prefill_runtime.config.device_sampling_enabled
        assert not executor.prefill_runtime.config.disable_batched_prefill
        kv_cache = executor.allocate_kv_cache()
        # Compile the read-only KV evidence slices before trace activation so
        # the later program-cache invariant measures runtime work, not test
        # instrumentation first use.
        _kv_snapshot(
            kv_cache,
            (0, 60),
            (_STALE_BLOCK, _STALE_BLOCK + 1),
            (_RESIDENT_BLOCK_START, _RESIDENT_BLOCK_START + 4),
        )
        topk = _compile_registration_order(executor, kv_cache, page_table, capture_order, sampling_order)
        assert executor.trace_compiler.trace_active
        baseline_registry = len(executor.program_compiler.compiled_programs)
        baseline_program_cache = _program_cache_entries(ttnn_mesh_device)
        baseline_summary = executor.traced_executor.runtime_summary()

        prepared = _prepared(executor, tokens, page_table)
        assert len(prepared) == 1
        item = prepared[0]
        assert item.request.kind == "batched"
        assert item.request.source_rows == tuple(range(15))
        assert item.request.padded_batch_size == 16
        assert item.program_signatures[0].operation_variant == "regular-batched"
        assert item.sampling_path == "logits"
        assert item.trace_signature is not None
        assert torch.all(item.request.tokens[15] == 0)
        assert torch.all(item.request.page_table[15] == -1)
        assert torch.all(item.request.page_table[:15, 4:] == -1)
        program_key = executor.program_compiler.key_for(item.program_signatures[0])
        trace_key = executor.trace_compiler.trace_key_for_program(program_key)
        assert trace_key is not None
        assert executor.trace_compiler.get(trace_key).artifact is not None

        stale_before = _kv_snapshot(kv_cache, (_STALE_BLOCK, _STALE_BLOCK + 1))
        resident_logits = executor.prefill_forward(
            resident_tokens,
            resident_table,
            kv_cache=kv_cache,
            prompt_lens=torch.tensor([_PROMPT_LEN]),
            empty_slots=[_RESIDENT_SLOT],
            execution=executor.traced_prefill_execution,
        )
        resident_before = _kv_snapshot(kv_cache, (_RESIDENT_BLOCK_START, _RESIDENT_BLOCK_START + 4))
        actual_logits = executor.prefill_forward(
            tokens,
            page_table[:15],
            kv_cache=kv_cache,
            prompt_lens=torch.full((15,), _PROMPT_LEN, dtype=torch.long),
            empty_slots=list(range(15)),
            execution=executor.traced_prefill_execution,
        )
        actual_kv = _kv_snapshot(
            kv_cache,
            (0, 60),
            (_RESIDENT_BLOCK_START, _RESIDENT_BLOCK_START + 4),
        )
        resident_after = _kv_snapshot(kv_cache, (_RESIDENT_BLOCK_START, _RESIDENT_BLOCK_START + 4))
        assert_rowwise_logits_parity(
            batched_eager_logits,
            expected_logits,
            min_row_pcc=_LOGITS_MIN_ROW_PCC,
            max_abs=_LOGITS_MAX_ABS,
            require_exact_top1=False,
            max_top1_mismatches=_LOGITS_MAX_TOP1_MISMATCHES,
            expected_top1_in_actual_topk=_LOGITS_TOPK,
            min_topk_overlap=_LOGITS_MIN_TOPK_OVERLAP,
            isclose_atol=_LOGITS_ATOL,
            isclose_rtol=0.05,
            max_isclose_failure_fraction=_LOGITS_MAX_ISCLOSE_FAILURE_FRACTION,
        )
        assert torch.equal(actual_logits, batched_eager_logits)
        _assert_nested_close(actual_kv, batched_eager_kv, atol=0.0, rtol=0.0)
        repeated_logits = executor.prefill_forward(
            tokens,
            page_table[:15],
            kv_cache=kv_cache,
            prompt_lens=torch.full((15,), _PROMPT_LEN, dtype=torch.long),
            empty_slots=list(range(15)),
            execution=executor.traced_prefill_execution,
        )
        repeated_kv = _kv_snapshot(
            kv_cache,
            (0, 60),
            (_RESIDENT_BLOCK_START, _RESIDENT_BLOCK_START + 4),
        )
        assert torch.equal(repeated_logits, actual_logits)
        _assert_nested_close(repeated_kv, actual_kv, atol=0.0, rtol=0.0)

        active_decode_tokens, active_decode_start_pos, active_decode_page_table = _active_decode_inputs(
            page_table, resident_table
        )
        active_decode = _decode_logits(
            executor.decode_forward(
                active_decode_tokens,
                active_decode_start_pos,
                active_decode_page_table,
                kv_cache=kv_cache,
                execution=executor.traced_decode_execution,
            )
        )[:15]
        assert torch.equal(active_decode, batched_eager_active_decode)
        assert_rowwise_logits_parity(
            batched_eager_active_decode,
            expected_active_decode,
            min_row_pcc=_DECODE_MIN_ROW_PCC,
            max_abs=_DECODE_MAX_ABS,
            require_exact_top1=False,
            max_top1_mismatches=_LOGITS_MAX_TOP1_MISMATCHES,
            expected_top1_in_actual_topk=_LOGITS_TOPK,
            min_topk_overlap=_LOGITS_MIN_TOPK_OVERLAP,
        )
        _assert_nested_close(resident_after, resident_before, atol=0.0, rtol=0.0)
        _assert_nested_close(
            _kv_snapshot(kv_cache, (_STALE_BLOCK, _STALE_BLOCK + 1)),
            stale_before,
            atol=0.0,
            rtol=0.0,
        )

        decode_tokens, decode_start_pos, decode_page_table = _resident_decode_inputs(resident_logits, resident_table)
        resident_decode = _decode_logits(
            executor.decode_forward(
                decode_tokens,
                decode_start_pos,
                decode_page_table,
                kv_cache=kv_cache,
                execution=executor.traced_decode_execution,
            )
        )[_RESIDENT_SLOT : _RESIDENT_SLOT + 1]
        torch.testing.assert_close(
            resident_decode,
            expected_resident_decode,
            atol=_LOGITS_ATOL,
            rtol=0.05,
        )

        # Keep the sampled oracle physically separate from the logits/KV oracle
        # so sampled replay cannot pass by reusing its active cache blocks.
        logits_kv_before_sample = _kv_snapshot(kv_cache, (0, 60))
        sampled_table = _page_table(offset=256)
        sampled_logits = executor.prefill_forward(
            tokens,
            sampled_table[:15],
            kv_cache=kv_cache,
            prompt_lens=torch.full((15,), _PROMPT_LEN, dtype=torch.long),
            empty_slots=list(range(15)),
            execution=executor.traced_prefill_execution,
        )
        sampled_prepared = _prepared(executor, tokens, sampled_table, sampling=topk)[0]
        assert sampled_prepared.sampling_path == "topk"
        assert sampled_prepared.program_signatures[0].operation_variant == "regular-batched"
        sampled = _sampled_tokens(
            executor.prefill_forward(
                tokens,
                sampled_table[:15],
                kv_cache=kv_cache,
                prompt_lens=torch.full((15,), _PROMPT_LEN, dtype=torch.long),
                empty_slots=list(range(15)),
                sampling_params=topk,
                execution=executor.traced_prefill_execution,
            )
        )
        assert sampled.shape == (15,)
        assert sampled_logits.shape[:2] == (15, 1)
        assert torch.equal(sampled, sampled_logits.argmax(dim=-1).reshape(-1))
        _assert_nested_close(
            _kv_snapshot(kv_cache, (0, 60)),
            logits_kv_before_sample,
            atol=0.0,
            rtol=0.0,
        )

        # Direct execution has no scheduler/preemption object; the public
        # resume contract is the full token row plus block-aligned start and
        # refreshed page table supplied after a cache hit/preemption. Keep this
        # traffic after the cache-isolated oracle: the long request writes
        # blocks 0..127 and otherwise changes the measured path's history.
        resumed_tokens = _tokens(1, salt=29).repeat(1, 2)
        resumed_table = _page_table()[:1]
        resumed_table[:, :5] = torch.arange(700, 705, dtype=torch.int32)
        resumed = _prepared(
            executor,
            resumed_tokens,
            resumed_table,
            start_pos=torch.tensor([32]),
            slots=[_RESUME_SLOT],
            prompt_lens=torch.tensor([160]),
        )[0]
        assert resumed.request.uses_chunked_prefill
        assert resumed.trace_signature is not None
        executor.prefill_forward(
            resumed_tokens,
            resumed_table,
            kv_cache=kv_cache,
            prompt_lens=torch.tensor([160]),
            start_pos=torch.tensor([32]),
            empty_slots=[_RESUME_SLOT],
            execution=executor.traced_prefill_execution,
        )

        long_tokens = torch.arange(_MAX_SEQ_LEN, dtype=torch.long).reshape(1, _MAX_SEQ_LEN) % 32000
        long_prepared = _prepared(executor, long_tokens, _page_table(), slots=[_RESUME_SLOT])[0]
        assert long_prepared.request.uses_chunked_prefill
        assert len(long_prepared.request.chunks) == 2
        assert long_prepared.trace_signature is not None
        assert long_prepared.program_signatures[0].operation_variant == "chunked"
        executor.prefill_forward(
            long_tokens,
            _page_table()[:1],
            kv_cache=kv_cache,
            prompt_lens=torch.tensor([_MAX_SEQ_LEN]),
            empty_slots=[_RESUME_SLOT],
            execution=executor.traced_prefill_execution,
        )

        # This completes the initial 15 -> 16 -> 15 cycle with refreshed token,
        # page-table, and sampling tensors. Nonzero start_pos is not supported
        # by production regular batching: cached rows deliberately take the
        # single/chunked path, covered by the resumed request above.
        refresh_cases = (
            (16, 211, 512, SamplingParams(temperature=0.5, top_k=1, top_p=0.75, seed=211)),
            (15, 419, 1024, SamplingParams(temperature=0.8, top_k=1, top_p=0.90, seed=419)),
        )
        for rows, salt, offset, refreshed_sampling in refresh_cases:
            refreshed_tokens = _tokens(rows, salt=salt)
            refreshed_table = _page_table(offset=offset)
            refreshed_logits = executor.prefill_forward(
                refreshed_tokens,
                refreshed_table[:rows],
                kv_cache=kv_cache,
                prompt_lens=torch.full((rows,), _PROMPT_LEN, dtype=torch.long),
                empty_slots=list(range(rows)),
                execution=executor.traced_prefill_execution,
            )
            refreshed_sample = _sampled_tokens(
                executor.prefill_forward(
                    refreshed_tokens,
                    refreshed_table[:rows],
                    kv_cache=kv_cache,
                    prompt_lens=torch.full((rows,), _PROMPT_LEN, dtype=torch.long),
                    empty_slots=list(range(rows)),
                    sampling_params=refreshed_sampling,
                    execution=executor.traced_prefill_execution,
                )
            )
            assert refreshed_sample.shape == (rows,)
            assert refreshed_logits.shape[:2] == (rows, 1)
            assert torch.equal(refreshed_sample, refreshed_logits.argmax(dim=-1).reshape(-1))

        assert len(executor.program_compiler.compiled_programs) == baseline_registry
        assert _program_cache_entries(ttnn_mesh_device) == baseline_program_cache
        summary = executor.traced_executor.runtime_summary()
        assert summary["eager_prefill_executions"] == baseline_summary["eager_prefill_executions"]
        assert summary["successful_trace_replays"] > baseline_summary["successful_trace_replays"]
        assert summary["strict_coverage_misses"] == 0
        assert summary["rejected_post_activation_compile_attempts"] == 0
        evidence = executor.traced_executor.recent_prefill_replay_evidence
        assert len(evidence) == 1
        assert evidence[0].operation == "prefill"
        assert evidence[0].variant == "regular-batched"
        assert evidence[0].sampling_path == "topk"
        assert evidence[0].execution == "trace_replay"
        assert (evidence[0].active_batch_size, evidence[0].padded_batch_size) == (15, 16)
    finally:
        executor.cleanup()
