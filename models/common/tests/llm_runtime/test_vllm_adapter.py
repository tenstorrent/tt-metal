# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import inspect

import pytest
import torch

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig
from models.common.llm_runtime.vllm_adapter import VLLMAdapter, VLLMAdapterConfig


def _adapter(*, trace=None, paged_config=None, model_dtype=ttnn.bfloat8_b):
    return VLLMAdapter(
        VLLMAdapterConfig.resolve(
            trace=trace or TraceConfig(mode="all"),
            paged_kv_cache=paged_config or PagedKVCacheConfig(block_size=32, max_num_blocks=128, dtype=ttnn.bfloat8_b),
            expected_num_layers=32,
            expected_kv_heads_per_device=8,
            expected_head_dim=128,
            model_kv_cache_dtype=model_dtype,
        )
    )


def test_config_resolves_canonical_static_policy_and_is_frozen(expect_error):
    trace = TraceConfig(mode="all")
    paged_kv_cache = PagedKVCacheConfig(block_size=32, max_num_blocks=128, dtype=ttnn.bfloat8_b)

    config = VLLMAdapterConfig.resolve(
        trace=trace,
        paged_kv_cache=paged_kv_cache,
        expected_num_layers=32.0,
        expected_kv_heads_per_device=8.0,
        expected_head_dim=128.0,
        model_kv_cache_dtype=[ttnn.bfloat8_b] * 32,
    )

    assert config.trace is trace
    assert config.paged_kv_cache is paged_kv_cache
    assert config.expected_num_layers == 32
    assert isinstance(config.expected_num_layers, int)
    assert config.expected_kv_heads_per_device == 8
    assert isinstance(config.expected_kv_heads_per_device, int)
    assert config.expected_head_dim == 128
    assert isinstance(config.expected_head_dim, int)
    assert config.model_kv_cache_dtypes == (ttnn.bfloat8_b,) * 32
    with expect_error(dataclasses.FrozenInstanceError, "cannot assign to field"):
        config.expected_num_layers = 1
    with expect_error(ValueError, "expected_num_layers"):
        VLLMAdapterConfig(
            trace=trace,
            paged_kv_cache=paged_kv_cache,
            expected_num_layers=0,
            expected_kv_heads_per_device=8,
            expected_head_dim=128,
            model_kv_cache_dtypes=(ttnn.bfloat8_b,),
        )
    with expect_error(TypeError, "must be a tuple"):
        VLLMAdapterConfig(
            trace=trace,
            paged_kv_cache=paged_kv_cache,
            expected_num_layers=32,
            expected_kv_heads_per_device=8,
            expected_head_dim=128,
            model_kv_cache_dtypes=[ttnn.bfloat8_b],
        )


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"trace": object()}, TypeError, "TraceConfig"),
        ({"paged_kv_cache": object()}, TypeError, "PagedKVCacheConfig"),
        ({"expected_num_layers": 0}, ValueError, "positive integer"),
        ({"expected_num_layers": True}, ValueError, "positive integer"),
        ({"expected_kv_heads_per_device": 0}, ValueError, "positive integer"),
        ({"expected_kv_heads_per_device": -1}, ValueError, "positive integer"),
        ({"expected_kv_heads_per_device": True}, ValueError, "positive integer"),
        ({"expected_kv_heads_per_device": 8.5}, ValueError, "positive integer"),
        ({"expected_head_dim": -1}, ValueError, "positive integer"),
        ({"expected_head_dim": 0}, ValueError, "positive integer"),
        ({"expected_head_dim": True}, ValueError, "positive integer"),
        ({"expected_head_dim": 128.5}, ValueError, "positive integer"),
        ({"model_kv_cache_dtype": ()}, ValueError, "cannot be empty"),
        ({"model_kv_cache_dtype": (ttnn.bfloat8_b,) * 2}, ValueError, "one dtype per model layer"),
        ({"model_kv_cache_dtype": None}, TypeError, "model metadata"),
    ],
)
def test_config_rejects_inconsistent_static_inputs(overrides, error_type, message, expect_error):
    arguments = {
        "trace": TraceConfig(mode="all"),
        "paged_kv_cache": PagedKVCacheConfig(block_size=32, max_num_blocks=128, dtype=ttnn.bfloat8_b),
        "expected_num_layers": 32,
        "expected_kv_heads_per_device": 8,
        "expected_head_dim": 128,
        "model_kv_cache_dtype": ttnn.bfloat8_b,
    }

    with expect_error(error_type, message):
        VLLMAdapterConfig.resolve(**(arguments | overrides))


def test_config_requires_exact_static_policy_types(expect_error):
    class TraceConfigSubclass(TraceConfig):
        pass

    class PagedKVCacheConfigSubclass(PagedKVCacheConfig):
        pass

    arguments = {
        "trace": TraceConfig(mode="all"),
        "paged_kv_cache": PagedKVCacheConfig(block_size=32, max_num_blocks=128, dtype=ttnn.bfloat8_b),
        "expected_num_layers": 32,
        "model_kv_cache_dtype": ttnn.bfloat8_b,
    }

    with expect_error(TypeError, "TraceConfig"):
        VLLMAdapterConfig.resolve(**(arguments | {"trace": TraceConfigSubclass(mode="all")}))
    with expect_error(TypeError, "PagedKVCacheConfig"):
        VLLMAdapterConfig.resolve(
            **(
                arguments
                | {
                    "paged_kv_cache": PagedKVCacheConfigSubclass(
                        block_size=32,
                        max_num_blocks=128,
                        dtype=ttnn.bfloat8_b,
                    )
                }
            )
        )


def test_adapter_is_plain_orchestration_with_one_config_surface(expect_error):
    adapter = _adapter()

    assert tuple(inspect.signature(VLLMAdapter).parameters) == ("config",)
    assert vars(adapter) == {"config": adapter.config}
    with expect_error(TypeError, "VLLMAdapterConfig"):
        VLLMAdapter(config=None)


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
    adapter = _adapter(trace=TraceConfig(mode="decode_only"))

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
            TraceConfig(mode="decode_only"),
            True,
        ),
        (
            "normalize_decode",
            (torch.zeros(1), torch.zeros(1), torch.zeros((1, 1))),
            TraceConfig(mode="none"),
            True,
        ),
    ],
)
def test_normalize_rejects_trace_hint_that_disagrees_with_static_policy(method_name, args, trace, hint, expect_error):
    adapter = _adapter(trace=trace)

    with expect_error(ValueError, "enable_trace"):
        getattr(adapter, method_name)(args, {"enable_trace": hint})


def test_eager_compile_trace_hint_is_allowed_with_static_trace_enabled():
    adapter = _adapter(trace=TraceConfig(mode="all"))

    normalized = adapter.normalize_decode(
        (torch.zeros(1), torch.zeros(1), torch.zeros((1, 1))),
        {"enable_trace": False},
    )

    assert normalized["enable_trace"] is False


@pytest.mark.parametrize("hint", [None, "true", 1])
def test_normalize_requires_an_explicit_boolean_trace_selection(hint, expect_error):
    adapter = _adapter()
    kwargs = {} if hint is None else {"enable_trace": hint}

    with expect_error(TypeError, "enable_trace"):
        adapter.normalize_prefill(
            (torch.zeros((1, 1)), torch.zeros((1, 1))),
            kwargs,
        )


def test_normalize_rejects_duplicate_positional_and_keyword_argument(expect_error):
    adapter = _adapter()

    with expect_error(TypeError, "tokens"):
        adapter.normalize_prefill(
            (torch.zeros((1, 1)), torch.zeros((1, 1))),
            {"tokens": torch.zeros((1, 1)), "enable_trace": True},
        )


def test_resolve_legacy_kv_cache_returns_new_immutable_config():
    base = PagedKVCacheConfig(block_size=32, max_num_blocks=128, dtype=ttnn.bfloat8_b)
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
    config = PagedKVCacheConfig(
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
        paged_config=PagedKVCacheConfig(
            block_size=32,
            max_num_blocks=128,
            dtype=ttnn.bfloat8_b,
            num_blocks=32,
        )
    )

    with expect_error(ValueError, "already resolved"):
        adapter.resolve_legacy_kv_cache_config((64, 8, 32, 128), torch.bfloat16, 32)
