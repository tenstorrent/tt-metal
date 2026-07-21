# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import pytest
import torch

import ttnn
from models.common.llm_runtime.vllm_adapter import VLLMAdapter


@dataclass(frozen=True)
class _TraceConfig:
    prefill_enabled: bool
    decode_enabled: bool


@dataclass(frozen=True)
class _PagedKVCacheConfig:
    block_size: int
    max_num_blocks: int
    dtype: object
    memory_config: object = "dram"
    num_blocks: int | None = None


def _adapter(*, trace=None, paged_config=None, model_dtype=ttnn.bfloat8_b):
    return VLLMAdapter(
        trace_config=trace or _TraceConfig(prefill_enabled=True, decode_enabled=True),
        paged_kv_cache_config=paged_config
        or _PagedKVCacheConfig(block_size=32, max_num_blocks=128, dtype=ttnn.bfloat8_b),
        expected_num_layers=32,
        expected_kv_heads_per_device=8,
        expected_head_dim=128,
        model_kv_cache_dtype=model_dtype,
    )


def test_normalize_prefill_positional_call_without_mutating_caller_kwargs():
    adapter = _adapter()
    kwargs = {
        "prompt_lens": [4, 3],
        "start_pos": [0, 1],
        "enable_trace": True,
        "page_tables_per_layer": object(),
        "sampling_params": "sampling",
    }

    normalized = adapter.normalize_prefill(
        ([[1, 2, 3, 4], [5, 6, 0, 0]], [[0, 1], [2, 3]]),
        kwargs,
    )

    assert normalized["tokens"].dtype == torch.long
    assert normalized["page_table"].dtype == torch.int32
    assert normalized["prompt_lens"].dtype == torch.long
    assert normalized["start_pos"].dtype == torch.long
    assert normalized["sampling_params"] == "sampling"
    assert normalized["enable_trace"] is True
    assert "page_tables_per_layer" not in normalized
    assert kwargs["enable_trace"] is True
    assert "page_tables_per_layer" in kwargs


def test_normalize_decode_converts_existing_tensors_and_flattens_column_tokens():
    adapter = _adapter(trace=_TraceConfig(prefill_enabled=False, decode_enabled=True))

    normalized = adapter.normalize_decode(
        (
            torch.tensor([[1], [2]], dtype=torch.int32),
            torch.tensor([3, 4], dtype=torch.int32),
            torch.tensor([[0], [1]], dtype=torch.int64),
        ),
        {"enable_trace": True, "slot_remap": [0, 1]},
    )

    assert normalized["tokens"].shape == (2,)
    assert normalized["tokens"].dtype == torch.long
    assert normalized["start_pos"].dtype == torch.long
    assert normalized["page_table"].dtype == torch.int32
    assert normalized["enable_trace"] is True
    assert "slot_remap" not in normalized


@pytest.mark.parametrize(
    ("method_name", "args", "trace", "hint"),
    [
        (
            "normalize_prefill",
            (torch.zeros((1, 1)), torch.zeros((1, 1))),
            _TraceConfig(prefill_enabled=False, decode_enabled=True),
            True,
        ),
        (
            "normalize_decode",
            (torch.zeros(1), torch.zeros(1), torch.zeros((1, 1))),
            _TraceConfig(prefill_enabled=True, decode_enabled=False),
            True,
        ),
    ],
)
def test_normalize_rejects_trace_hint_that_disagrees_with_static_policy(method_name, args, trace, hint, expect_error):
    adapter = _adapter(trace=trace)

    with expect_error(ValueError, "enable_trace"):
        getattr(adapter, method_name)(args, {"enable_trace": hint})


def test_eager_compile_trace_hint_is_allowed_with_static_trace_enabled():
    adapter = _adapter(trace=_TraceConfig(prefill_enabled=True, decode_enabled=True))

    normalized = adapter.normalize_decode(
        (torch.zeros(1), torch.zeros(1), torch.zeros((1, 1))),
        {"enable_trace": False},
    )

    assert normalized["enable_trace"] is False


def test_normalize_rejects_duplicate_positional_and_keyword_argument(expect_error):
    adapter = _adapter()

    with expect_error(TypeError, "tokens"):
        adapter.normalize_prefill(
            (torch.zeros((1, 1)), torch.zeros((1, 1))),
            {"tokens": torch.zeros((1, 1)), "enable_trace": True},
        )


def test_resolve_legacy_kv_cache_returns_new_immutable_config():
    base = _PagedKVCacheConfig(block_size=32, max_num_blocks=128, dtype=ttnn.bfloat8_b)
    adapter = _adapter(paged_config=base)

    resolved = adapter.resolve_legacy_kv_cache_config(
        (64, 8, 32, 128),
        torch.bfloat16,
        32,
    )

    assert resolved is not base
    assert base.num_blocks is None
    assert resolved.num_blocks == 64
    assert resolved.block_size == base.block_size
    assert resolved.max_num_blocks == base.max_num_blocks
    assert resolved.dtype == base.dtype
    assert resolved.memory_config == base.memory_config


@pytest.mark.parametrize(
    ("shape", "dtype", "num_layers", "message"),
    [
        ((64, 8, 16, 128), torch.bfloat16, 32, "block size"),
        ((64, 4, 32, 128), torch.bfloat16, 32, "KV heads"),
        ((64, 8, 32, 64), torch.bfloat16, 32, "head dimension"),
        ((129, 8, 32, 128), torch.bfloat16, 32, "max_num_blocks"),
        ((64, 8, 32, 128), torch.float32, 32, "dtype"),
        ((64, 8, 32, 128), torch.bfloat16, 31, "layer count"),
    ],
)
def test_resolve_legacy_kv_cache_rejects_mismatched_vllm_spec(shape, dtype, num_layers, message, expect_error):
    adapter = _adapter()

    with expect_error((TypeError, ValueError), message):
        adapter.resolve_legacy_kv_cache_config(shape, dtype, num_layers)


def test_adapter_rejects_static_dtype_that_disagrees_with_model_owned_dtype(expect_error):
    with expect_error(ValueError, "model-owned"):
        _adapter(model_dtype=ttnn.bfloat16)


def test_adapter_requires_explicit_model_owned_dtype_metadata(expect_error):
    with expect_error(TypeError, "must be supplied from model metadata"):
        _adapter(model_dtype=None)


def test_bfloat4_model_dtype_uses_shared_bfloat16_torch_surrogate():
    config = _PagedKVCacheConfig(
        block_size=32,
        max_num_blocks=128,
        dtype=ttnn.bfloat4_b,
    )
    adapter = _adapter(paged_config=config, model_dtype=ttnn.bfloat4_b)

    resolved = adapter.resolve_legacy_kv_cache_config(
        (64, 8, 32, 128),
        torch.bfloat16,
        32,
    )

    assert resolved.dtype == ttnn.bfloat4_b
    assert resolved.num_blocks == 64


def test_resolve_legacy_kv_cache_rejects_replacing_resolved_capacity(expect_error):
    adapter = _adapter(
        paged_config=_PagedKVCacheConfig(
            block_size=32,
            max_num_blocks=128,
            dtype=ttnn.bfloat8_b,
            num_blocks=32,
        )
    )

    with expect_error(ValueError, "already resolved"):
        adapter.resolve_legacy_kv_cache_config((64, 8, 32, 128), torch.bfloat16, 32)
