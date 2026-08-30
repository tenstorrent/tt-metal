# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import inspect
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig
from models.common.llm_runtime.vllm_adapter import (
    NormalizedDecodeKwargs,
    NormalizedPrefillKwargs,
    VLLMAdapter,
    VLLMAdapterConfig,
)
from models.common.models.llama3_8b.executor import Llama3Executor
from models.common.models.llama3_8b.generator import Llama3Generator
from models.common.models.qwen25_72b.generator import Qwen25_72BGenerator
from models.common.models.qwen25_7b.generator import Qwen25Generator
from models.common.models.qwen25_coder_32b.generator import Qwen25Coder32BGenerator
from models.common.models.qwen2_7b.generator import Qwen2Generator


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


@pytest.mark.parametrize(
    ("method_name", "expected"),
    [
        (
            "normalize_prefill",
            [
                ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("tokens", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("page_table", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("enable_trace", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
                ("prompt_lens", inspect.Parameter.KEYWORD_ONLY, None),
                ("start_pos", inspect.Parameter.KEYWORD_ONLY, None),
                ("empty_slots", inspect.Parameter.KEYWORD_ONLY, None),
                ("kv_cache", inspect.Parameter.KEYWORD_ONLY, None),
                ("sampling_params", inspect.Parameter.KEYWORD_ONLY, None),
                ("compatibility_kwargs", inspect.Parameter.KEYWORD_ONLY, None),
            ],
        ),
        (
            "normalize_decode",
            [
                ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("tokens", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("start_pos", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("page_table", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("enable_trace", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
                ("kv_cache", inspect.Parameter.KEYWORD_ONLY, None),
                ("sampling_params", inspect.Parameter.KEYWORD_ONLY, None),
                ("reset_batch", inspect.Parameter.KEYWORD_ONLY, False),
                ("compatibility_kwargs", inspect.Parameter.KEYWORD_ONLY, None),
            ],
        ),
    ],
)
def test_normalizer_signatures_are_explicit(method_name, expected):
    parameters = inspect.signature(getattr(VLLMAdapter, method_name)).parameters

    assert [(name, parameter.kind, parameter.default) for name, parameter in parameters.items()] == expected


def test_normalized_typed_dicts_have_stable_key_order_and_optional_request_state():
    assert tuple(NormalizedPrefillKwargs.__annotations__) == (
        "tokens",
        "page_table",
        "prompt_lens",
        "start_pos",
        "empty_slots",
        "kv_cache",
        "sampling_params",
        "prompt_tokens",
        "output_tokens",
        "slot_remap",
    )
    assert NormalizedPrefillKwargs.__required_keys__ == frozenset(
        {"tokens", "page_table", "prompt_lens", "start_pos", "empty_slots", "kv_cache", "sampling_params"}
    )
    assert NormalizedPrefillKwargs.__optional_keys__ == frozenset({"prompt_tokens", "output_tokens", "slot_remap"})
    assert tuple(NormalizedDecodeKwargs.__annotations__) == (
        "tokens",
        "start_pos",
        "page_table",
        "kv_cache",
        "sampling_params",
        "reset_batch",
        "prompt_tokens",
        "output_tokens",
        "slot_remap",
    )
    assert NormalizedDecodeKwargs.__required_keys__ == frozenset(
        {"tokens", "start_pos", "page_table", "kv_cache", "sampling_params", "reset_batch"}
    )
    assert NormalizedDecodeKwargs.__optional_keys__ == frozenset({"prompt_tokens", "output_tokens", "slot_remap"})


def test_normalize_prefill_positional_call_without_mutating_caller_kwargs():
    adapter = _adapter()
    compatibility_kwargs = {
        "page_tables_per_layer": object(),
        "prompt_tokens": object(),
        "output_tokens": object(),
        "slot_remap": object(),
        "rope_deltas_all_users": object(),
    }

    normalized, enable_trace = adapter.normalize_prefill(
        [[1, 2, 3, 4], [5, 6, 0, 0]],
        [[0, 1], [2, 3]],
        enable_trace=True,
        prompt_lens=[4, 3],
        start_pos=[0, 1],
        sampling_params="sampling",
        compatibility_kwargs=compatibility_kwargs,
    )

    assert tuple(normalized) == tuple(NormalizedPrefillKwargs.__annotations__)
    assert normalized["tokens"].dtype == torch.long
    assert normalized["page_table"].dtype == torch.int32
    assert normalized["prompt_lens"].dtype == torch.long
    assert normalized["start_pos"].dtype == torch.long
    assert normalized["empty_slots"] is None
    assert normalized["kv_cache"] is None
    assert normalized["sampling_params"] == "sampling"
    assert normalized["prompt_tokens"] is compatibility_kwargs["prompt_tokens"]
    assert normalized["output_tokens"] is compatibility_kwargs["output_tokens"]
    assert normalized["slot_remap"] is compatibility_kwargs["slot_remap"]
    assert enable_trace is True
    assert tuple(compatibility_kwargs) == (
        "page_tables_per_layer",
        "prompt_tokens",
        "output_tokens",
        "slot_remap",
        "rope_deltas_all_users",
    )


def test_normalize_decode_converts_existing_tensors_and_flattens_column_tokens():
    adapter = _adapter(trace=TraceConfig(mode="decode_only"))

    normalized, enable_trace = adapter.normalize_decode(
        torch.tensor([[1], [2]], dtype=torch.int32),
        torch.tensor([3, 4], dtype=torch.int32),
        torch.tensor([[0], [1]], dtype=torch.int64),
        enable_trace=True,
        compatibility_kwargs={"slot_remap": [0, 1]},
    )

    assert tuple(normalized) == (
        "tokens",
        "start_pos",
        "page_table",
        "kv_cache",
        "sampling_params",
        "reset_batch",
        "slot_remap",
    )
    assert normalized["tokens"].shape == (2,)
    assert normalized["tokens"].dtype == torch.long
    assert normalized["start_pos"].dtype == torch.long
    assert normalized["page_table"].dtype == torch.int32
    assert normalized["kv_cache"] is None
    assert normalized["sampling_params"] is None
    assert "prompt_tokens" not in normalized
    assert "output_tokens" not in normalized
    assert normalized["slot_remap"] == [0, 1]
    assert normalized["reset_batch"] is False
    assert enable_trace is True


@pytest.mark.parametrize("method_name", ["normalize_prefill", "normalize_decode"])
@pytest.mark.parametrize(
    "compatibility_kwargs",
    (None, {"prompt_tokens": None, "output_tokens": None, "slot_remap": None}),
)
def test_normalize_omits_unsupplied_or_none_request_state(method_name, compatibility_kwargs):
    adapter = _adapter(trace=TraceConfig(mode="all"))
    args = (
        (torch.zeros((1, 1)), torch.zeros((1, 1)))
        if method_name == "normalize_prefill"
        else (torch.zeros(1), torch.zeros(1), torch.zeros((1, 1)))
    )

    normalized, _ = getattr(adapter, method_name)(
        *args,
        enable_trace=True,
        compatibility_kwargs=compatibility_kwargs,
    )

    assert not ({"prompt_tokens", "output_tokens", "slot_remap"} & normalized.keys())


class _NarrowQwenRequestTarget:
    def __init__(self):
        self.calls = []

    def prefill_forward(
        self,
        tokens,
        page_table,
        *,
        prompt_lens=None,
        start_pos=None,
        empty_slots=None,
        kv_cache=None,
        sampling_params=None,
        execution=None,
    ):
        self.calls.append(
            (
                "prefill",
                {
                    "tokens": tokens,
                    "page_table": page_table,
                    "prompt_lens": prompt_lens,
                    "start_pos": start_pos,
                    "empty_slots": empty_slots,
                    "kv_cache": kv_cache,
                    "sampling_params": sampling_params,
                    "execution": execution,
                },
            )
        )
        return "prefill"

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        *,
        kv_cache=None,
        sampling_params=None,
        reset_batch=False,
        read_from_device=True,
        execution=None,
    ):
        self.calls.append(
            (
                "decode",
                {
                    "tokens": tokens,
                    "start_pos": start_pos,
                    "page_table": page_table,
                    "kv_cache": kv_cache,
                    "sampling_params": sampling_params,
                    "reset_batch": reset_batch,
                    "read_from_device": read_from_device,
                    "execution": execution,
                },
            )
        )
        return "decode"


@pytest.mark.parametrize(
    "generator_class",
    (Qwen2Generator, Qwen25Generator, Qwen25_72BGenerator, Qwen25Coder32BGenerator),
)
def test_qwen_generator_dispatch_omits_absent_state_for_narrow_request_surface(generator_class):
    generator = object.__new__(generator_class)
    generator._adapter = _adapter()
    generator.target = _NarrowQwenRequestTarget()
    prefill_execution = object()
    decode_execution = object()
    generator._select_prefill_execution = lambda normalized, requested: prefill_execution
    generator._select_execution = lambda operation, requested: decode_execution

    assert generator.prefill_forward([[1]], [[0]], enable_trace=False) == "prefill"
    assert generator.decode_forward([1], [0], [[0]], enable_trace=False) == "decode"

    assert [name for name, _ in generator.target.calls] == ["prefill", "decode"]
    assert generator.target.calls[0][1]["execution"] is prefill_execution
    assert generator.target.calls[1][1]["execution"] is decode_execution


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
        getattr(adapter, method_name)(*args, enable_trace=hint)


def test_normalize_accepts_unselected_operation_with_available_trace_target():
    adapter = _adapter(trace=TraceConfig(mode="all"))

    _, enable_trace = adapter.normalize_prefill(
        torch.zeros((1, 1)),
        torch.zeros((1, 1)),
        enable_trace=False,
    )

    assert enable_trace is False


@pytest.mark.parametrize(
    ("mode", "prefill_enabled", "decode_enabled"),
    [
        ("none", False, False),
        ("decode_only", False, True),
        ("all", True, True),
    ],
)
def test_normalize_accepts_operation_specific_static_trace_policy(mode, prefill_enabled, decode_enabled):
    adapter = _adapter(trace=TraceConfig(mode=mode))

    _, normalized_prefill_trace = adapter.normalize_prefill(
        torch.zeros((1, 1)),
        torch.zeros((1, 1)),
        enable_trace=prefill_enabled,
    )
    _, normalized_decode_trace = adapter.normalize_decode(
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros((1, 1)),
        enable_trace=decode_enabled,
    )

    assert normalized_prefill_trace is prefill_enabled
    assert normalized_decode_trace is decode_enabled


@pytest.mark.parametrize("hint", [None, "true", 1])
def test_normalize_requires_an_explicit_boolean_trace_selection(hint, expect_error):
    adapter = _adapter()

    with expect_error(TypeError, "enable_trace"):
        if hint is None:
            adapter.normalize_prefill(torch.zeros((1, 1)), torch.zeros((1, 1)))
        else:
            adapter.normalize_prefill(torch.zeros((1, 1)), torch.zeros((1, 1)), enable_trace=hint)


def test_normalize_rejects_duplicate_positional_and_keyword_argument(expect_error):
    adapter = _adapter()

    with expect_error(TypeError, "tokens"):
        adapter.normalize_prefill(
            torch.zeros((1, 1)),
            torch.zeros((1, 1)),
            tokens=torch.zeros((1, 1)),
            enable_trace=True,
        )


@pytest.mark.parametrize("method_name", ["normalize_prefill", "normalize_decode"])
def test_normalize_rejects_unknown_compatibility_keys(method_name, expect_error):
    adapter = _adapter()
    args = (
        (torch.zeros((1, 1)), torch.zeros((1, 1)))
        if method_name == "normalize_prefill"
        else (torch.zeros(1), torch.zeros(1), torch.zeros((1, 1)))
    )

    with expect_error(TypeError, "unexpected keyword argument 'unknown_plugin_field'"):
        getattr(adapter, method_name)(
            *args,
            enable_trace=True,
            compatibility_kwargs={"unknown_plugin_field": object()},
        )


def _signature_entries(method):
    return [
        (name, parameter.kind, parameter.default) for name, parameter in inspect.signature(method).parameters.items()
    ]


@pytest.mark.parametrize(
    ("method_name", "expected"),
    [
        (
            "compile_prefill",
            [
                ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("tokens", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("page_table", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("enable_trace", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
                ("prompt_lens", inspect.Parameter.KEYWORD_ONLY, None),
                ("start_pos", inspect.Parameter.KEYWORD_ONLY, None),
                ("empty_slots", inspect.Parameter.KEYWORD_ONLY, None),
                ("kv_cache", inspect.Parameter.KEYWORD_ONLY, None),
                ("sampling_params", inspect.Parameter.KEYWORD_ONLY, None),
            ],
        ),
        (
            "compile_decode",
            [
                ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("tokens", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("start_pos", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("page_table", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("enable_trace", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
                ("kv_cache", inspect.Parameter.KEYWORD_ONLY, None),
                ("sampling_params", inspect.Parameter.KEYWORD_ONLY, None),
                ("reset_batch", inspect.Parameter.KEYWORD_ONLY, False),
            ],
        ),
        (
            "prefill_forward",
            [
                ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("tokens", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("page_table", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("enable_trace", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
                ("prompt_lens", inspect.Parameter.KEYWORD_ONLY, None),
                ("start_pos", inspect.Parameter.KEYWORD_ONLY, None),
                ("empty_slots", inspect.Parameter.KEYWORD_ONLY, None),
                ("kv_cache", inspect.Parameter.KEYWORD_ONLY, None),
                ("sampling_params", inspect.Parameter.KEYWORD_ONLY, None),
                ("compatibility_kwargs", inspect.Parameter.VAR_KEYWORD, inspect.Parameter.empty),
            ],
        ),
        (
            "decode_forward",
            [
                ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("tokens", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("start_pos", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("page_table", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("enable_trace", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
                ("kv_cache", inspect.Parameter.KEYWORD_ONLY, None),
                ("sampling_params", inspect.Parameter.KEYWORD_ONLY, None),
                ("reset_batch", inspect.Parameter.KEYWORD_ONLY, False),
                ("read_from_device", inspect.Parameter.KEYWORD_ONLY, True),
                ("compatibility_kwargs", inspect.Parameter.VAR_KEYWORD, inspect.Parameter.empty),
            ],
        ),
        (
            "read_decode_output",
            [
                ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("tt_out", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("async_read", inspect.Parameter.KEYWORD_ONLY, False),
            ],
        ),
        (
            "process_decode_output_host",
            [
                ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("tt_out", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("is_tokens", inspect.Parameter.KEYWORD_ONLY, False),
            ],
        ),
        (
            "warmup_model_prefill",
            [
                ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("kv_cache", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
                ("can_sample_on_device", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
                ("enable_trace", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
            ],
        ),
        (
            "warmup_model_decode",
            [
                ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
                ("kv_cache", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
                ("max_batch_size", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
                ("num_blocks", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
                ("can_sample_on_device", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
                ("enable_trace", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
            ],
        ),
    ],
)
def test_registered_generator_signatures_are_exact(method_name, expected):
    assert _signature_entries(getattr(Llama3Generator, method_name)) == expected


def test_registered_generator_sizing_signature_has_no_compatibility_bag():
    assert _signature_entries(Llama3Generator.get_max_tokens_all_users) == [
        ("model_name", inspect.Parameter.POSITIONAL_OR_KEYWORD, ""),
        ("num_devices", inspect.Parameter.POSITIONAL_OR_KEYWORD, 1),
        ("tt_data_parallel", inspect.Parameter.POSITIONAL_OR_KEYWORD, 1),
        ("max_model_len", inspect.Parameter.POSITIONAL_OR_KEYWORD, 0),
        ("max_num_seqs", inspect.Parameter.POSITIONAL_OR_KEYWORD, 1),
    ]


def test_registered_generator_compatibility_bags_exist_only_on_forward_methods():
    variadic_keyword_methods = {
        method_name
        for method_name, method in vars(Llama3Generator).items()
        if inspect.isfunction(method)
        and any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in inspect.signature(method).parameters.values()
        )
    }

    assert variadic_keyword_methods == {"prefill_forward", "decode_forward"}


class _ExplicitGeneratorTarget:
    model = SimpleNamespace()
    model_args = object()
    mesh_device = object()
    cache_path = "cache"
    already_warmed_up_prefill = False
    eager_execution = object()
    traced_prefill_execution = object()
    traced_decode_execution = object()

    def __init__(self):
        self.calls = []

    def _record(self, method_name, arguments):
        self.calls.append(
            (
                method_name,
                {name: value for name, value in arguments.items() if name != "self"},
            )
        )

    def can_trace_prefill(
        self,
        *,
        tokens,  # ↓ Core request
        prompt_lens=None,  # ↓ Sequence metadata
        start_pos=None,
        empty_slots=None,  # ↓ Lane routing
    ):
        self._record("can_trace_prefill", locals())
        return True

    def compile_prefill(
        self,
        tokens,
        page_table,
        *,
        prompt_lens=None,  # ↓ Sequence metadata
        start_pos=None,
        empty_slots=None,  # ↓ Lane routing
        kv_cache=None,  # ↓ Borrowed resources
        sampling_params=None,  # ↓ Sampling
        prompt_tokens=None,  # ↓ Request-owned sampling state
        output_tokens=None,
        slot_remap=None,
        execution=None,  # ↓ Internal dispatch
    ):
        self._record("compile_prefill", locals())

    def compile_decode(
        self,
        tokens,
        start_pos,
        page_table,
        *,
        kv_cache=None,  # ↓ Borrowed resources
        sampling_params=None,  # ↓ Sampling
        prompt_tokens=None,  # ↓ Request-owned sampling state
        output_tokens=None,
        slot_remap=None,
        reset_batch=False,  # ↓ State transition
        execution=None,  # ↓ Internal dispatch
    ):
        self._record("compile_decode", locals())

    def prefill_forward(
        self,
        tokens,
        page_table,
        *,
        prompt_lens=None,  # ↓ Sequence metadata
        start_pos=None,
        empty_slots=None,  # ↓ Lane routing
        kv_cache=None,  # ↓ Borrowed resources
        sampling_params=None,  # ↓ Sampling
        prompt_tokens=None,  # ↓ Request-owned sampling state
        output_tokens=None,
        slot_remap=None,
        execution=None,  # ↓ Internal dispatch
    ):
        self._record("prefill_forward", locals())
        return "prefill"

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        *,
        kv_cache=None,  # ↓ Borrowed resources
        sampling_params=None,  # ↓ Sampling
        prompt_tokens=None,  # ↓ Request-owned sampling state
        output_tokens=None,
        slot_remap=None,
        reset_batch=False,  # ↓ State transition
        read_from_device=True,  # ↓ Output policy
        execution=None,  # ↓ Internal dispatch
    ):
        self._record("decode_forward", locals())
        return "decode"


def test_registered_generator_compile_methods_normalize_and_select_execution():
    target = _ExplicitGeneratorTarget()
    generator = Llama3Generator(target, _adapter())
    tokens = torch.tensor([[1, 2]])
    page_table = torch.tensor([[0]])
    prompt_lens = torch.tensor([2])
    start_pos = torch.tensor([0])
    kv_cache = object()
    sampling_params = object()

    generator.compile_prefill(
        tokens,
        page_table,
        enable_trace=True,
        prompt_lens=prompt_lens,
        start_pos=start_pos,
        empty_slots=[0],
        kv_cache=kv_cache,
        sampling_params=sampling_params,
    )
    assert target.calls[0][0] == "compile_prefill"
    assert target.calls[0][1]["execution"] is target.traced_prefill_execution
    assert set(target.calls[0][1]) == set(NormalizedPrefillKwargs.__annotations__) | {"execution"}

    generator.compile_decode(
        tokens[:, 0],
        start_pos,
        page_table,
        enable_trace=True,
        kv_cache=kv_cache,
        sampling_params=sampling_params,
        reset_batch=True,
    )
    assert target.calls[1][0] == "compile_decode"
    assert target.calls[1][1]["execution"] is target.traced_decode_execution
    assert set(target.calls[1][1]) == set(NormalizedDecodeKwargs.__annotations__) | {"execution"}


def test_registered_generator_discards_allowlisted_compatibility_and_limits_trace_classification():
    target = _ExplicitGeneratorTarget()
    generator = Llama3Generator(target, _adapter())
    tokens = torch.tensor([[1, 2]])
    page_table = torch.tensor([[0]])
    prompt_lens = torch.tensor([2])
    start_pos = torch.tensor([0])
    kv_cache = object()
    sampling_params = object()

    assert (
        generator.prefill_forward(
            tokens,
            page_table,
            enable_trace=True,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=[0],
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            page_tables_per_layer=object(),
        )
        == "prefill"
    )
    assert target.calls[0][0] == "prefill_forward"
    assert target.calls[0][1]["execution"] is target.traced_prefill_execution
    assert set(target.calls[0][1]) == set(NormalizedPrefillKwargs.__annotations__) | {"execution"}

    assert (
        generator.decode_forward(
            tokens[:, 0],
            start_pos,
            page_table,
            enable_trace=True,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            reset_batch=True,
            read_from_device=False,
            slot_remap=[0],
        )
        == "decode"
    )
    assert target.calls[1][0] == "decode_forward"
    assert target.calls[1][1]["execution"] is target.traced_decode_execution
    assert target.calls[1][1]["read_from_device"] is False
    assert set(target.calls[1][1]) == set(NormalizedDecodeKwargs.__annotations__) | {
        "read_from_device",
        "execution",
    }


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        ("prefill_forward", (torch.zeros((1, 1)), torch.zeros((1, 1)))),
        ("decode_forward", (torch.zeros(1), torch.zeros(1), torch.zeros((1, 1)))),
    ],
)
def test_registered_generator_rejects_unknown_compatibility_before_target_selection(
    method_name,
    args,
    expect_error,
):
    target = _ExplicitGeneratorTarget()
    generator = Llama3Generator(target, _adapter())

    with expect_error(TypeError, "unexpected keyword argument 'unknown_plugin_field'"):
        getattr(generator, method_name)(
            *args,
            enable_trace=True,
            unknown_plugin_field=object(),
        )

    assert target.calls == []


def test_registered_generator_rejects_unknown_nonforward_keywords(expect_error):
    with expect_error(TypeError, "unexpected keyword argument"):
        Llama3Generator.get_max_tokens_all_users(unknown_plugin_field=True)


def test_registered_generator_forwards_output_and_warmup_arguments_by_name():
    pending_output = object()
    read_events = [object()]
    target = create_autospec(Llama3Executor, instance=True)
    target.read_decode_output.return_value = pending_output, read_events
    target.process_decode_output_host.return_value = "tokens", "log-probs"
    generator = Llama3Generator(target, adapter=object())
    tt_out = object()
    kv_cache = object()

    assert generator.read_decode_output(tt_out, async_read=True) == (pending_output, read_events)
    target.read_decode_output.assert_called_once_with(tt_out=tt_out, async_read=True)
    assert generator.process_decode_output_host(pending_output, is_tokens=True) == ("tokens", "log-probs")
    target.process_decode_output_host.assert_called_once_with(tt_out=pending_output, is_tokens=True)

    generator.warmup_model_prefill(
        kv_cache=kv_cache,
        can_sample_on_device=True,
        enable_trace=False,
    )
    target.warmup_model_prefill.assert_called_once_with(
        kv_cache=kv_cache,
        can_sample_on_device=True,
        enable_trace=False,
    )
    generator.warmup_model_decode(
        kv_cache=kv_cache,
        max_batch_size=8,
        num_blocks=128,
        can_sample_on_device=False,
        enable_trace=True,
    )
    target.warmup_model_decode.assert_called_once_with(
        kv_cache=kv_cache,
        max_batch_size=8,
        num_blocks=128,
        can_sample_on_device=False,
        enable_trace=True,
    )


def test_resolve_legacy_kv_cache_returns_new_immutable_config():
    base = PagedKVCacheConfig(block_size=32, max_num_blocks=128, dtype=ttnn.bfloat8_b)
    adapter = _adapter(paged_config=base)

    resolved = adapter.resolve_legacy_kv_cache_config(
        (129, 8, 64, 128),
        torch.bfloat16,
        32,
    )

    assert resolved is not base
    assert base.num_blocks is None
    assert base.block_size == 32
    assert base.max_num_blocks == 128
    assert resolved.num_blocks == 129
    assert resolved.block_size == 64
    assert resolved.max_num_blocks == 129
    assert resolved.dtype == base.dtype
    assert resolved.memory_config == base.memory_config


@pytest.mark.parametrize(
    ("shape", "dtype", "num_layers", "message"),
    [
        ((64, 8, 0, 128), torch.bfloat16, 32, "block_size"),
        ((64, 4, 32, 128), torch.bfloat16, 32, "KV heads"),
        ((64, 8, 32, 64), torch.bfloat16, 32, "head dimension"),
        ((0, 8, 32, 128), torch.bfloat16, 32, "num_blocks"),
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
