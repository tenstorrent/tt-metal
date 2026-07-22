# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError, fields

import pytest

import ttnn
from models.common.llm_runtime import config as runtime_config
from models.common.llm_runtime.config import PagedKVCacheConfig, PageTableLayout, TraceConfig, WarmupConfig
from models.common.models.llama3_8b.executor import Llama3ExecutorConfig


class _TraceConfigSubclass(TraceConfig):
    pass


def _paged_config(**overrides):
    kwargs = {
        "block_size": 32,
        "max_num_blocks": 1024,
        "dtype": ttnn.bfloat8_b,
    }
    kwargs.update(overrides)
    return PagedKVCacheConfig(**kwargs)


def test_executor_config_has_exact_static_policy_owners_and_is_frozen(expect_error):
    config = Llama3ExecutorConfig(
        trace=TraceConfig(mode="all"),
        warmup=WarmupConfig(),
        paged_kv_cache=_paged_config(),
        device_sampling_enabled=True,
    )

    assert [field.name for field in fields(config)] == [
        "trace",
        "warmup",
        "paged_kv_cache",
        "device_sampling_enabled",
    ]
    forbidden = {
        "model",
        "mesh_device",
        "hf_model",
        "tokenizer",
        "dtype",
        "n_layers",
        "sampling_config",
        "sampling_output_dtype",
    }
    assert forbidden.isdisjoint(field.name for field in fields(config))
    assert not hasattr(runtime_config, "LLMGraphCompilerConfig")
    assert not hasattr(runtime_config, "LLMExecutorConfig")
    assert not hasattr(runtime_config, "Sampling1DConfig")
    with expect_error(FrozenInstanceError, ""):
        config.device_sampling_enabled = False


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("trace", WarmupConfig()),
        ("trace", _TraceConfigSubclass()),
        ("warmup", TraceConfig()),
        ("paged_kv_cache", WarmupConfig()),
    ],
)
def test_executor_config_rejects_non_exact_nested_config_types(field_name, invalid_value, expect_error):
    values = {
        "trace": TraceConfig(),
        "warmup": WarmupConfig(),
        "paged_kv_cache": _paged_config(),
        "device_sampling_enabled": False,
    }
    values[field_name] = invalid_value

    with expect_error(TypeError, rf"{field_name} must be exactly"):
        Llama3ExecutorConfig(**values)


@pytest.mark.parametrize(
    ("mode", "prefill", "decode"),
    [("none", False, False), ("decode_only", False, True), ("all", True, True)],
)
def test_trace_config_selects_static_coverage(mode, prefill, decode, expect_error):
    config = TraceConfig(mode=mode)

    assert config.prefill_enabled is prefill
    assert config.decode_enabled is decode
    with expect_error(FrozenInstanceError, ""):
        config.mode = "none"


def test_trace_config_rejects_unknown_mode(expect_error):
    with expect_error(ValueError, "Unsupported trace mode"):
        TraceConfig(mode="prefill_only")


def test_warmup_config_keeps_model_derived_defaults_and_is_deeply_immutable(expect_error):
    config = WarmupConfig()

    assert config.prefill_seq_lens is None
    assert config.prefill_batch_sizes == (1, 2, 4, 8, 16, 32)
    assert config.include_decode_top_k is False
    with expect_error(TypeError, "must be a tuple"):
        WarmupConfig(prefill_batch_sizes=[1, 2])


def test_paged_kv_config_has_plan_fields_and_resolved_capacity(expect_error):
    unresolved = _paged_config()
    resolved = _paged_config(num_blocks=512)

    assert [field.name for field in fields(unresolved)] == [
        "block_size",
        "max_num_blocks",
        "dtype",
        "memory_config",
        "num_blocks",
    ]
    assert unresolved.memory_config == ttnn.DRAM_MEMORY_CONFIG
    assert not unresolved.is_resolved()
    assert unresolved.capacity_tokens is None
    assert unresolved.max_capacity_tokens == 32 * 1024
    assert resolved.is_resolved()
    assert resolved.capacity_tokens == 32 * 512
    with expect_error(FrozenInstanceError, ""):
        resolved.num_blocks = 256


def test_paged_kv_config_rejects_invalid_capacity(expect_error):
    with expect_error(ValueError, "exceeds max_num_blocks"):
        _paged_config(num_blocks=1025)
    with expect_error(ValueError, "block_size"):
        _paged_config(block_size=0)


def test_page_table_layout_is_resolved_without_warmup_policy():
    layout = PageTableLayout.resolve(
        block_size=32,
        model_max_sequence_length=4096,
        physical_num_blocks=100,
        max_prefill_chunk_size=2048,
    )

    assert layout.raw_capacity_width == 100
    assert layout.decode_width == 104
    assert layout.prefill_width == 168
