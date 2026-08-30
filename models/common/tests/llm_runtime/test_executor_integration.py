# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Generic host-only executor/generator integration for migrated siblings."""

import inspect
from dataclasses import replace
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import MagicMock, create_autospec

import pytest
import torch

import ttnn
from models.common.llm_runtime.config import PageTableLayout, PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.llm_runtime.decode import DecodeTraceSignature
from models.common.llm_runtime.execution import EagerExecutor
from models.common.llm_runtime.lane_group import LaneGroupExecutor
from models.common.llm_runtime.prefill.signatures import PrefillProgramSignature
from models.common.llm_runtime.program_compiler import ProgramKey
from models.common.llm_runtime.warmup import _build_plan
from models.common.models import executor as shared_model_executor
from models.common.models import llama3_executor as llama3_family_executor
from models.common.models import qwen2_executor as qwen2_family_executor
from models.common.models.deepseek_r1_distill_qwen_14b import executor as deepseek_executor
from models.common.models.deepseek_r1_distill_qwen_14b import generator as deepseek_generator
from models.common.models.llama32_1b import executor as llama32_executor
from models.common.models.llama32_1b import generator as llama32_generator
from models.common.models.llama32_3b import executor as llama32_3b_executor
from models.common.models.llama32_3b import generator as llama32_3b_generator
from models.common.models.llama33_70b import executor as llama33_70b_executor
from models.common.models.llama33_70b import generator as llama33_70b_generator
from models.common.models.mistral_7b import executor as mistral_executor
from models.common.models.mistral_7b import generator as mistral_generator
from models.common.models.phi4 import executor as phi4_executor
from models.common.models.phi4 import generator as phi4_generator
from models.common.models.qwen2_7b import executor as qwen2_executor
from models.common.models.qwen2_7b import generator as qwen2_generator
from models.common.models.qwen3_32b import executor as qwen3_32b_executor
from models.common.models.qwen3_32b import generator as qwen3_32b_generator
from models.common.models.qwen25_7b import executor as qwen25_executor
from models.common.models.qwen25_7b import generator as qwen25_generator
from models.common.models.qwen25_72b import executor as qwen25_72b_executor
from models.common.models.qwen25_72b import generator as qwen25_72b_generator
from models.common.models.qwen25_coder_32b import executor as qwen25_coder_32b_executor
from models.common.models.qwen25_coder_32b import generator as qwen25_coder_32b_generator

EXECUTOR_BINDINGS = {
    "llama32_1b": SimpleNamespace(
        executor_module=llama32_executor,
        executor_class=llama32_executor.Llama32_1BExecutor,
        executor_config_class=llama32_executor.Llama32_1BExecutorConfig,
        generator_module=llama32_generator,
        generator_class=llama32_generator.Llama32_1BGenerator,
        generator_config_class=llama32_generator.Llama32_1BGeneratorConfig,
        build_generator_name="build_llama32_1b_generator",
        build_executor_name="build_llama32_1b_executor",
        make_model=lambda **kwargs: _make_llama32_model(**kwargs),
        make_runtime_config=lambda: _make_llama32_runtime_config(),
        make_executor_config=lambda mode="none": _make_llama32_executor_config(mode),
        make_recording_target=lambda **kwargs: _RecordingTarget(_make_llama32_model(), **kwargs),
        make_product=lambda mesh_device, max_batch_size: _make_llama32_product(mesh_device, max_batch_size),
        make_lane=lambda llm, config: _FakeLane(llm, config),
        hf_model="meta-llama/Llama-3.2-1B-Instruct",
    ),
    "llama32_3b": SimpleNamespace(
        executor_module=llama32_3b_executor,
        executor_class=llama32_3b_executor.Llama32_3BExecutor,
        executor_config_class=llama32_3b_executor.Llama32_3BExecutorConfig,
        generator_module=llama32_3b_generator,
        generator_class=llama32_3b_generator.Llama32_3BGenerator,
        generator_config_class=llama32_3b_generator.Llama32_3BGeneratorConfig,
        build_generator_name="build_llama32_3b_generator",
        build_executor_name="build_llama32_3b_executor",
        make_model=lambda **kwargs: _make_llama32_model(**kwargs),
        make_runtime_config=lambda: _make_llama32_runtime_config(),
        make_executor_config=lambda mode="none": _make_llama32_executor_config(mode, module=llama32_3b_executor),
        make_recording_target=lambda **kwargs: _RecordingTarget(_make_llama32_model(), **kwargs),
        make_product=lambda mesh_device, max_batch_size: _make_llama32_product(mesh_device, max_batch_size),
        make_lane=lambda llm, config: _FakeLane(llm, config),
        hf_model="meta-llama/Llama-3.2-3B-Instruct",
    ),
    "llama33_70b": SimpleNamespace(
        executor_module=llama33_70b_executor,
        executor_class=llama33_70b_executor.Llama33_70BExecutor,
        executor_config_class=llama33_70b_executor.Llama33_70BExecutorConfig,
        generator_module=llama33_70b_generator,
        generator_class=llama33_70b_generator.Llama33_70BGenerator,
        generator_config_class=llama33_70b_generator.Llama33_70BGeneratorConfig,
        build_generator_name="build_llama33_70b_generator",
        build_executor_name="build_llama33_70b_executor",
        make_model=lambda **kwargs: _make_llama32_model(**kwargs),
        make_runtime_config=lambda: _make_llama32_runtime_config(),
        make_executor_config=lambda mode="none": _make_llama32_executor_config(mode, module=llama33_70b_executor),
        make_recording_target=lambda **kwargs: _RecordingTarget(
            _make_llama32_model(),
            request_state_fields=llama33_70b_executor.Llama33_70BExecutor.request_state_fields,
            **kwargs,
        ),
        make_product=lambda mesh_device, max_batch_size: _make_llama32_product(mesh_device, max_batch_size),
        make_lane=lambda llm, config: _FakeLane(
            llm, config, request_state_fields=llama33_70b_executor.Llama33_70BExecutor.request_state_fields
        ),
        hf_model="meta-llama/Llama-3.3-70B-Instruct",
    ),
    "qwen2_7b": SimpleNamespace(
        executor_module=qwen2_executor,
        executor_class=qwen2_executor.Qwen2Executor,
        executor_config_class=qwen2_executor.Qwen2ExecutorConfig,
        generator_module=qwen2_generator,
        generator_class=qwen2_generator.Qwen2Generator,
        generator_config_class=qwen2_generator.Qwen2GeneratorConfig,
        build_generator_name="build_qwen2_7b_generator",
        build_executor_name="build_qwen2_7b_executor",
        make_model=lambda **kwargs: _make_qwen2_model(**kwargs),
        make_runtime_config=lambda: _make_qwen2_runtime_config(),
        make_executor_config=lambda mode="none": _make_qwen2_executor_config(mode),
        make_recording_target=lambda **kwargs: _RecordingTarget(_make_qwen2_model(), **kwargs),
        make_product=lambda mesh_device, max_batch_size: _make_qwen2_product(mesh_device, max_batch_size),
        make_lane=lambda llm, config: _FakeLane(llm, config),
        hf_model="Qwen/Qwen2-7B-Instruct",
    ),
    "qwen25_7b": SimpleNamespace(
        executor_module=qwen25_executor,
        executor_class=qwen25_executor.Qwen25Executor,
        executor_config_class=qwen25_executor.Qwen25ExecutorConfig,
        generator_module=qwen25_generator,
        generator_class=qwen25_generator.Qwen25Generator,
        generator_config_class=qwen25_generator.Qwen25GeneratorConfig,
        build_generator_name="build_qwen25_7b_generator",
        build_executor_name="build_qwen25_7b_executor",
        make_model=lambda **kwargs: _make_qwen2_model(**kwargs),
        make_runtime_config=lambda: _make_qwen2_runtime_config(max_prefill_batch_size=8),
        make_executor_config=lambda mode="none": _make_qwen2_executor_config(mode, module=qwen25_executor),
        make_recording_target=lambda **kwargs: _RecordingTarget(_make_qwen2_model(), **kwargs),
        make_product=lambda mesh_device, max_batch_size: _make_qwen2_product(
            mesh_device, max_batch_size, max_prefill_batch_size=8
        ),
        make_lane=lambda llm, config: _FakeLane(llm, config),
        hf_model="Qwen/Qwen2.5-7B-Instruct",
    ),
    "qwen25_72b": SimpleNamespace(
        executor_module=qwen25_72b_executor,
        executor_class=qwen25_72b_executor.Qwen25_72BExecutor,
        executor_config_class=qwen25_72b_executor.Qwen25_72BExecutorConfig,
        generator_module=qwen25_72b_generator,
        generator_class=qwen25_72b_generator.Qwen25_72BGenerator,
        generator_config_class=qwen25_72b_generator.Qwen25_72BGeneratorConfig,
        build_generator_name="build_qwen25_72b_generator",
        build_executor_name="build_qwen25_72b_executor",
        make_model=lambda **kwargs: _make_qwen25_72b_model(**kwargs),
        make_runtime_config=lambda: _make_qwen25_72b_runtime_config(),
        make_executor_config=lambda mode="none": _make_qwen2_executor_config(mode, module=qwen25_72b_executor),
        make_recording_target=lambda **kwargs: _RecordingTarget(_make_qwen25_72b_model(), **kwargs),
        make_product=lambda mesh_device, max_batch_size: _make_qwen25_72b_product(mesh_device, max_batch_size),
        make_lane=lambda llm, config: _FakeLane(llm, config),
        hf_model="Qwen/Qwen2.5-72B-Instruct",
    ),
    "qwen25_coder_32b": SimpleNamespace(
        executor_module=qwen25_coder_32b_executor,
        executor_class=qwen25_coder_32b_executor.Qwen25Coder32BExecutor,
        executor_config_class=qwen25_coder_32b_executor.Qwen25Coder32BExecutorConfig,
        generator_module=qwen25_coder_32b_generator,
        generator_class=qwen25_coder_32b_generator.Qwen25Coder32BGenerator,
        generator_config_class=qwen25_coder_32b_generator.Qwen25Coder32BGeneratorConfig,
        build_generator_name="build_qwen25_coder_32b_generator",
        build_executor_name="build_qwen25_coder_32b_executor",
        make_model=lambda **kwargs: _make_qwen25_coder_32b_model(**kwargs),
        make_runtime_config=lambda: _make_qwen25_coder_32b_runtime_config(),
        make_executor_config=lambda mode="none": _make_qwen2_executor_config(mode, module=qwen25_coder_32b_executor),
        make_recording_target=lambda **kwargs: _RecordingTarget(_make_qwen25_coder_32b_model(), **kwargs),
        make_product=lambda mesh_device, max_batch_size: _make_qwen25_coder_32b_product(mesh_device, max_batch_size),
        make_lane=lambda llm, config: _FakeLane(llm, config),
        hf_model="Qwen/Qwen2.5-Coder-32B-Instruct",
    ),
    "qwen3_32b": SimpleNamespace(
        executor_module=qwen3_32b_executor,
        executor_class=qwen3_32b_executor.Qwen3_32BExecutor,
        executor_config_class=qwen3_32b_executor.Qwen3_32BExecutorConfig,
        generator_module=qwen3_32b_generator,
        generator_class=qwen3_32b_generator.Qwen3_32BGenerator,
        generator_config_class=qwen3_32b_generator.Qwen3_32BGeneratorConfig,
        build_generator_name="build_qwen3_32b_generator",
        build_executor_name="build_qwen3_32b_executor",
        make_model=lambda **kwargs: _make_qwen3_32b_model(**kwargs),
        make_runtime_config=lambda: _make_qwen3_32b_runtime_config(),
        make_executor_config=lambda mode="none": _make_qwen2_executor_config(mode, module=qwen3_32b_executor),
        make_recording_target=lambda **kwargs: _RecordingTarget(
            _make_qwen3_32b_model(),
            request_state_fields=qwen3_32b_executor.Qwen3_32BExecutor.request_state_fields,
            **kwargs,
        ),
        make_product=lambda mesh_device, max_batch_size: _make_qwen3_32b_product(mesh_device, max_batch_size),
        make_lane=lambda llm, config: _FakeLane(
            llm, config, request_state_fields=qwen3_32b_executor.Qwen3_32BExecutor.request_state_fields
        ),
        hf_model="Qwen/Qwen3-32B",
    ),
    "deepseek_r1_distill_qwen_14b": SimpleNamespace(
        executor_module=deepseek_executor,
        executor_class=deepseek_executor.DeepSeekR1Qwen14BExecutor,
        executor_config_class=deepseek_executor.DeepSeekR1Qwen14BExecutorConfig,
        generator_module=deepseek_generator,
        generator_class=deepseek_generator.DeepSeekR1Qwen14BGenerator,
        generator_config_class=deepseek_generator.DeepSeekR1Qwen14BGeneratorConfig,
        build_generator_name="build_deepseek_r1_distill_qwen_14b_generator",
        build_executor_name="build_deepseek_r1_distill_qwen_14b_executor",
        make_model=lambda **kwargs: _make_qwen2_model(**kwargs),
        make_runtime_config=lambda: _make_qwen2_runtime_config(max_prefill_batch_size=32),
        make_executor_config=lambda mode="none": _make_qwen2_executor_config(mode, module=deepseek_executor),
        make_recording_target=lambda **kwargs: _RecordingTarget(_make_qwen2_model(), **kwargs),
        make_product=lambda mesh_device, max_batch_size: _make_qwen2_product(
            mesh_device, max_batch_size, max_prefill_batch_size=32
        ),
        make_lane=lambda llm, config: _FakeLane(llm, config),
        hf_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    ),
    "mistral_7b": SimpleNamespace(
        executor_module=mistral_executor,
        executor_class=mistral_executor.Mistral7BExecutor,
        executor_config_class=mistral_executor.Mistral7BExecutorConfig,
        generator_module=mistral_generator,
        generator_class=mistral_generator.Mistral7BGenerator,
        generator_config_class=mistral_generator.Mistral7BGeneratorConfig,
        build_generator_name="build_mistral_7b_generator",
        build_executor_name="build_mistral_7b_executor",
        make_model=lambda **kwargs: _make_qwen2_model(**kwargs),
        make_runtime_config=lambda: _make_qwen2_runtime_config(max_prefill_batch_size=8),
        make_executor_config=lambda mode="none": _make_qwen2_executor_config(mode, module=mistral_executor),
        make_recording_target=lambda **kwargs: _RecordingTarget(_make_qwen2_model(), **kwargs),
        make_product=lambda mesh_device, max_batch_size: _make_qwen2_product(
            mesh_device, max_batch_size, max_prefill_batch_size=8
        ),
        make_lane=lambda llm, config: _FakeLane(llm, config),
        hf_model="mistralai/Mistral-7B-Instruct-v0.3",
    ),
    "phi4": SimpleNamespace(
        executor_module=phi4_executor,
        executor_class=phi4_executor.Phi4Executor,
        executor_config_class=phi4_executor.Phi4ExecutorConfig,
        generator_module=phi4_generator,
        generator_class=phi4_generator.Phi4Generator,
        generator_config_class=phi4_generator.Phi4GeneratorConfig,
        build_generator_name="build_phi4_generator",
        build_executor_name="build_phi4_executor",
        make_model=lambda **kwargs: _make_qwen2_model(**kwargs),
        make_runtime_config=lambda: _make_qwen2_runtime_config(max_prefill_batch_size=8),
        make_executor_config=lambda mode="none": _make_qwen2_executor_config(mode, module=phi4_executor),
        make_recording_target=lambda **kwargs: _RecordingTarget(_make_qwen2_model(), **kwargs),
        make_product=lambda mesh_device, max_batch_size: _make_qwen2_product(
            mesh_device, max_batch_size, max_prefill_batch_size=8
        ),
        make_lane=lambda llm, config: _FakeLane(llm, config),
        hf_model="microsoft/phi-4",
    ),
}

GENERATOR_PATHS = {
    "llama32_1b": "models.common.models.llama32_1b.generator:Llama32_1BGenerator",
    "llama32_3b": "models.common.models.llama32_3b.generator:Llama32_3BGenerator",
    "llama33_70b": "models.common.models.llama33_70b.generator:Llama33_70BGenerator",
    "mistral_7b": "models.common.models.mistral_7b.generator:Mistral7BGenerator",
    "phi4": "models.common.models.phi4.generator:Phi4Generator",
    "qwen2_7b": "models.common.models.qwen2_7b.generator:Qwen2Generator",
    "qwen25_7b": "models.common.models.qwen25_7b.generator:Qwen25Generator",
    "qwen25_72b": "models.common.models.qwen25_72b.generator:Qwen25_72BGenerator",
    "qwen25_coder_32b": "models.common.models.qwen25_coder_32b.generator:Qwen25Coder32BGenerator",
    "qwen3_32b": "models.common.models.qwen3_32b.generator:Qwen3_32BGenerator",
    "deepseek_r1_distill_qwen_14b": (
        "models.common.models.deepseek_r1_distill_qwen_14b.generator:DeepSeekR1Qwen14BGenerator"
    ),
}


@pytest.fixture(params=EXECUTOR_BINDINGS.items(), ids=lambda item: item[0])
def binding(request):
    return request.param[1]


_LLAMA_FAMILY_EXECUTOR_MODULES = (llama32_executor, llama32_3b_executor, llama33_70b_executor)
_QWEN2_FAMILY_EXECUTOR_MODULES = (
    qwen2_executor,
    qwen25_executor,
    qwen25_72b_executor,
    qwen25_coder_32b_executor,
)
_SHARED_MODEL_EXECUTOR_MODULES = (
    *_LLAMA_FAMILY_EXECUTOR_MODULES,
    *_QWEN2_FAMILY_EXECUTOR_MODULES,
    qwen3_32b_executor,
)


def _composition_module(binding):
    if binding.executor_module in _SHARED_MODEL_EXECUTOR_MODULES:
        return shared_model_executor
    return binding.executor_module


def _sampling_policy_module(binding):
    if binding.executor_module in _LLAMA_FAMILY_EXECUTOR_MODULES:
        return llama3_family_executor
    if binding.executor_module in _QWEN2_FAMILY_EXECUTOR_MODULES:
        return qwen2_family_executor
    return binding.executor_module


class _Mesh:
    shape = (1, 1)

    @staticmethod
    def get_num_devices():
        return 1


class _Mesh2:
    shape = (1, 2)

    @staticmethod
    def get_num_devices():
        return 2


class _Mesh8:
    shape = (1, 8)

    @staticmethod
    def get_num_devices():
        return 8


def _make_llama32_model(max_batch_size=4):
    paged = SimpleNamespace(block_size=32, max_num_blocks=132)
    attention = SimpleNamespace(
        n_kv_heads=8,
        head_dim=64,
        kv_cache_dtype=ttnn.bfloat8_b,
        paged_attention_config=paged,
        use_vllm_paged_kv_cache=True,
        kv_cache=None,
    )
    live = SimpleNamespace(config=attention, kv_cache=None)
    model = SimpleNamespace(
        config=SimpleNamespace(
            mesh_device=_Mesh(),
            max_batch_size=max_batch_size,
            max_seq_len=4096,
            n_layers=1,
            num_devices=1,
            block_configs=(SimpleNamespace(attention_config=attention),),
        ),
        layers=(SimpleNamespace(attention=live),),
        iter_executor_named_modules=lambda: (),
        vocab_size=128256,
        num_devices=1,
    )

    def configure_paged_attention(*, block_size, max_num_blocks):
        assert attention.kv_cache is None
        assert live.kv_cache is None
        attention.paged_attention_config = SimpleNamespace(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

    model.configure_paged_attention = configure_paged_attention
    return model


def _make_llama32_runtime_config():
    return SimpleNamespace(
        model_cache_path="cache",
        max_prefill_chunk_size=2048,
        trace_prefill_supported_seq_lens=(128,),
        can_enable_trace=lambda length, num_cached_tokens=0: length == 128,
        supports_batched_prefill=True,
        disable_batched_prefill=False,
        max_prefill_batch_size=32,
        batched_prefill_batched_extract=True,
    )


def _make_llama32_executor_config(mode="none", *, module=llama32_executor):
    config_class = next(
        getattr(module, name)
        for name in (
            "Llama32_1BExecutorConfig",
            "Llama32_3BExecutorConfig",
            "Llama33_70BExecutorConfig",
        )
        if hasattr(module, name)
    )
    return config_class(
        trace=TraceConfig(mode),
        warmup=WarmupConfig(prefill_seq_lens=(128,), prefill_batch_sizes=(1,)),
        paged_kv_cache=PagedKVCacheConfig(block_size=32, max_num_blocks=132, dtype=ttnn.bfloat8_b),
        device_sampling_enabled=False,
    )


def _make_llama32_product(mesh_device, max_batch_size):
    model = _make_llama32_model(max_batch_size=max_batch_size)
    model.config.mesh_device = mesh_device
    return SimpleNamespace(model=model, runtime_config=_make_llama32_runtime_config())


def _make_qwen2_model(max_batch_size=4):
    model = _make_llama32_model(max_batch_size=max_batch_size)
    model.config.mesh_device = _Mesh2()
    model.config.num_devices = 2
    model.num_devices = 2
    attention = model.layers[0].attention.config
    attention.n_kv_heads = 4
    attention.head_dim = 128
    return model


def _make_qwen2_runtime_config(*, max_prefill_batch_size=32):
    runtime = _make_llama32_runtime_config()
    runtime.trace_prefill_supported_seq_lens = (128, 1024)
    runtime.can_enable_trace = lambda length, num_cached_tokens=0: length in (128, 1024)
    runtime.max_prefill_batch_size = max_prefill_batch_size
    return runtime


def _make_qwen2_executor_config(mode="none", *, module=qwen2_executor):
    config_class = (
        getattr(module, "Qwen2ExecutorConfig", None)
        or getattr(module, "Qwen25ExecutorConfig", None)
        or getattr(module, "Qwen25_72BExecutorConfig", None)
        or getattr(module, "Qwen25Coder32BExecutorConfig", None)
        or getattr(module, "Qwen3_32BExecutorConfig", None)
        or getattr(module, "DeepSeekR1Qwen14BExecutorConfig", None)
        or getattr(module, "Mistral7BExecutorConfig", None)
        or module.Phi4ExecutorConfig
    )
    return config_class(
        trace=TraceConfig(mode),
        warmup=WarmupConfig(prefill_seq_lens=(128, 1024), prefill_batch_sizes=(1,)),
        paged_kv_cache=PagedKVCacheConfig(block_size=32, max_num_blocks=132, dtype=ttnn.bfloat8_b),
        device_sampling_enabled=False,
    )


def _make_qwen2_product(mesh_device, max_batch_size, *, max_prefill_batch_size=32):
    model = _make_qwen2_model(max_batch_size=max_batch_size)
    model.config.mesh_device = mesh_device
    return SimpleNamespace(
        model=model,
        runtime_config=_make_qwen2_runtime_config(max_prefill_batch_size=max_prefill_batch_size),
    )


def _make_qwen25_72b_model(max_batch_size=4):
    model = _make_llama32_model(max_batch_size=max_batch_size)
    model.config.mesh_device = _Mesh8()
    model.config.num_devices = 8
    model.num_devices = 8
    attention = model.layers[0].attention.config
    attention.n_kv_heads = 8
    attention.head_dim = 128
    return model


def _make_qwen25_72b_runtime_config():
    return _make_qwen2_runtime_config(max_prefill_batch_size=32)


def _make_qwen25_72b_product(mesh_device, max_batch_size):
    model = _make_qwen25_72b_model(max_batch_size=max_batch_size)
    model.config.mesh_device = mesh_device
    return SimpleNamespace(model=model, runtime_config=_make_qwen25_72b_runtime_config())


def _make_qwen25_coder_32b_model(max_batch_size=4):
    model = _make_qwen25_72b_model(max_batch_size=max_batch_size)
    model.config.dim = 5120
    model.config.n_heads = 40
    model.config.hidden_dim = 27648
    model.config.hf_model_id = "Qwen/Qwen2.5-Coder-32B-Instruct"
    return model


def _make_qwen25_coder_32b_runtime_config():
    runtime = _make_qwen25_72b_runtime_config()
    runtime.max_prefill_chunk_size = 4096
    runtime.can_enable_trace = lambda length, num_cached_tokens=0: num_cached_tokens == 0 and length in (128, 1024)
    return runtime


def _make_qwen25_coder_32b_product(mesh_device, max_batch_size):
    model = _make_qwen25_coder_32b_model(max_batch_size=max_batch_size)
    model.config.mesh_device = mesh_device
    return SimpleNamespace(model=model, runtime_config=_make_qwen25_coder_32b_runtime_config())


def _make_qwen3_32b_model(max_batch_size=4):
    model = _make_qwen25_72b_model(max_batch_size=max_batch_size)
    model.config.dim = 5120
    model.config.n_heads = 64
    model.config.hidden_dim = 27648
    model.config.vocab_size = 151936
    model.config.hf_model_id = "Qwen/Qwen3-32B"
    model.padded_vocab_size = 152064
    return model


def _make_qwen3_32b_runtime_config():
    runtime = _make_qwen25_coder_32b_runtime_config()
    runtime.max_prefill_chunk_size = 4096
    return runtime


def _make_qwen3_32b_product(mesh_device, max_batch_size):
    model = _make_qwen3_32b_model(max_batch_size=max_batch_size)
    model.config.mesh_device = mesh_device
    return SimpleNamespace(model=model, runtime_config=_make_qwen3_32b_runtime_config())


def test_qwen2_binding_preserves_tp2_runtime_and_sampling_defaults():
    model = _make_qwen2_model()
    _, num_layers, kv_heads_per_device, head_dim = qwen2_generator._model_kv_metadata(model)
    runtime = _make_qwen2_runtime_config()
    config = qwen2_generator.Qwen2GeneratorConfig(
        hf_model="Qwen/Qwen2-7B-Instruct",
        hf_revision="test-revision",
        mesh_device=model.config.mesh_device,
        max_batch_size=32,
        max_seq_len=4096,
    )

    assert model.config.mesh_device.shape == (1, 2)
    assert (num_layers, kv_heads_per_device, head_dim) == (1, 2, 128)
    assert runtime.trace_prefill_supported_seq_lens == (128, 1024)
    assert runtime.max_prefill_chunk_size == 2048
    assert runtime.max_prefill_batch_size == 32
    assert config.hf_revision == "test-revision"
    assert config.device_sampling_enabled is False


def test_qwen25_72b_binding_preserves_tp8_runtime_and_sampling_defaults():
    model = _make_qwen25_72b_model()
    _, num_layers, kv_heads_per_device, head_dim = qwen25_72b_generator._model_kv_metadata(model)
    runtime = _make_qwen25_72b_runtime_config()
    config = qwen25_72b_generator.Qwen25_72BGeneratorConfig(
        hf_model="Qwen/Qwen2.5-72B-Instruct",
        mesh_device=model.config.mesh_device,
        max_batch_size=32,
        max_seq_len=4096,
    )

    assert model.config.mesh_device.shape == (1, 8)
    assert (num_layers, kv_heads_per_device, head_dim) == (1, 1, 128)
    assert runtime.trace_prefill_supported_seq_lens == (128, 1024)
    assert runtime.max_prefill_chunk_size == 2048
    assert runtime.max_prefill_batch_size == 32
    assert config.hf_revision == qwen25_72b_generator.DEFAULT_HF_REVISION
    assert config.device_sampling_enabled is False


def test_qwen25_coder_32b_binding_preserves_tp8_runtime_and_sampling_defaults():
    model = _make_qwen25_coder_32b_model()
    _, num_layers, kv_heads_per_device, head_dim = qwen25_coder_32b_generator._model_kv_metadata(model)
    runtime = _make_qwen25_coder_32b_runtime_config()
    config = qwen25_coder_32b_generator.Qwen25Coder32BGeneratorConfig(
        hf_model="Qwen/Qwen2.5-Coder-32B-Instruct",
        mesh_device=model.config.mesh_device,
        max_batch_size=32,
        max_seq_len=4096,
    )

    assert model.config.mesh_device.shape == (1, 8)
    assert (num_layers, kv_heads_per_device, head_dim) == (1, 1, 128)
    assert runtime.trace_prefill_supported_seq_lens == (128, 1024)
    assert runtime.max_prefill_chunk_size == 4096
    assert runtime.can_enable_trace(128, 0) is True
    assert runtime.can_enable_trace(128, 1) is False
    assert runtime.max_prefill_batch_size == 32
    assert config.hf_revision == qwen25_coder_32b_generator.DEFAULT_HF_REVISION
    assert config.device_sampling_enabled is False


@pytest.mark.parametrize(
    "product_binding",
    (EXECUTOR_BINDINGS["qwen25_72b"], EXECUTOR_BINDINGS["qwen25_coder_32b"]),
    ids=("qwen25_72b", "qwen25_coder_32b"),
)
@pytest.mark.parametrize("device_sampling_enabled", (False, True), ids=("sampling-off", "sampling-on"))
def test_large_qwen_builder_threads_exact_decode_sampling_coverage(
    monkeypatch,
    product_binding,
    device_sampling_enabled,
):
    mesh_device = _Mesh()
    product = product_binding.make_product(mesh_device, 4)
    executor_configs = []

    monkeypatch.setattr(product_binding.generator_module, "from_pretrained", lambda **kwargs: product)
    monkeypatch.setattr(
        product_binding.generator_module,
        "_model_kv_metadata",
        lambda model: ((ttnn.bfloat8_b,), 1, 1, 128),
    )

    def build_executor(llm, config):
        executor_configs.append(config)
        return product_binding.make_lane(llm, config)

    monkeypatch.setattr(product_binding.generator_module, product_binding.build_executor_name, build_executor)
    generator = getattr(product_binding.generator_module, product_binding.build_generator_name)(
        product_binding.generator_config_class(
            hf_model=product_binding.hf_model,
            mesh_device=mesh_device,
            max_batch_size=4,
            max_seq_len=1024,
            n_layers=1,
            device_sampling_enabled=device_sampling_enabled,
        )
    )

    try:
        assert len(executor_configs) == 1
        warmup = executor_configs[0].warmup
        assert warmup.include_decode_top_k is device_sampling_enabled
        plan = _build_plan(
            warmup=warmup,
            layout=PageTableLayout(block_size=32, raw_capacity_width=32, prefill_width=64, decode_width=32),
            prefill_sequence_lengths=(128,),
            lane_batch_size=4,
            allow_force_argmax=True,
            can_sample_on_device=device_sampling_enabled,
        )
        assert [case.sampling_path for case in plan.decode] == (
            ["logits", "argmax", "topk"] if device_sampling_enabled else ["logits"]
        )
    finally:
        generator.cleanup()


def test_qwen3_32b_binding_preserves_tp8_runtime_and_padded_vocab_defaults():
    model = _make_qwen3_32b_model()
    _, num_layers, kv_heads_per_device, head_dim = qwen3_32b_generator._model_kv_metadata(model)
    runtime = _make_qwen3_32b_runtime_config()
    config = qwen3_32b_generator.Qwen3_32BGeneratorConfig(
        hf_model="Qwen/Qwen3-32B",
        mesh_device=model.config.mesh_device,
        max_batch_size=32,
        max_seq_len=4096,
    )

    assert model.config.mesh_device.shape == (1, 8)
    assert (num_layers, kv_heads_per_device, head_dim) == (1, 1, 128)
    assert model.config.vocab_size == 151936
    assert model.padded_vocab_size == 152064
    assert runtime.trace_prefill_supported_seq_lens == (128, 1024)
    assert runtime.max_prefill_chunk_size == 4096
    assert runtime.can_enable_trace(128, 0) is True
    assert runtime.can_enable_trace(128, 1) is False
    assert config.hf_revision == qwen3_32b_generator.DEFAULT_HF_REVISION
    assert config.device_sampling_enabled is False


@pytest.mark.parametrize(
    ("cluster_shape", "disable_batched_prefill", "advertised_lengths", "expected"),
    [
        ([1, 8], False, (128, 1024), (128, 1024)),
        ([1, 8], True, (128, 1024), (128, 1024)),
        ([1, 4], False, (128, 1024), ()),
        ([1, 8], False, (128,), (128,)),
    ],
    ids=("t3k-batched", "t3k-sequential", "bh-batched", "t3k-low-ceiling"),
)
def test_qwen3_prefill_capture_primes_are_t3k_product_owned(
    cluster_shape,
    disable_batched_prefill,
    advertised_lengths,
    expected,
):
    runtime = SimpleNamespace(
        cluster_shape=cluster_shape,
        disable_batched_prefill=disable_batched_prefill,
        trace_prefill_supported_seq_lens=advertised_lengths,
        can_enable_trace=lambda length, cached: length in advertised_lengths and cached == 0,
    )

    num_devices = int(cluster_shape[0]) * int(cluster_shape[1])
    assert (
        qwen3_32b_executor._resolve_trace_capture_prime_sequence_lengths(
            runtime,
            num_devices=num_devices,
        )
        == expected
    )


def test_phi4_binding_preserves_cap8_trace_buckets_and_pinned_revision():
    runtime = _make_qwen2_runtime_config(max_prefill_batch_size=8)
    config = phi4_generator.Phi4GeneratorConfig(
        hf_model="microsoft/phi-4",
        mesh_device=_Mesh2(),
        max_batch_size=32,
        max_seq_len=4096,
    )

    assert runtime.trace_prefill_supported_seq_lens == (128, 1024)
    assert runtime.max_prefill_chunk_size == 2048
    assert runtime.max_prefill_batch_size == 8
    assert config.hf_revision == phi4_generator.DEFAULT_HF_REVISION
    assert config.device_sampling_enabled is False


@pytest.mark.parametrize("mode", ["none", "decode_only", "all"])
def test_model_owned_executor_has_exact_composition_and_owner_counts(binding, mode, monkeypatch):
    owner_names = (
        "PagedKVCacheManager",
        "OutputReader",
        "PrefillRuntime",
        "DecodeRuntime",
        "ProgramCompiler",
        "EagerExecutor",
        "TraceCompiler",
        "TracedExecutor",
        "WarmupCoordinator",
    )
    composition_module = _composition_module(binding)
    owner_factories = {}
    for name in owner_names:
        factory = MagicMock(wraps=getattr(composition_module, name))
        monkeypatch.setattr(composition_module, name, factory)
        owner_factories[name] = factory

    executor = binding.executor_class(
        binding.make_model(),
        binding.make_runtime_config(),
        binding.make_executor_config(mode),
    )
    expected_counts = {name: 1 for name in owner_names}
    if mode == "none":
        expected_counts["TraceCompiler"] = 0
        expected_counts["TracedExecutor"] = 0
    assert {name: factory.call_count for name, factory in owner_factories.items()} == expected_counts
    assert executor.eager_executor.program_compiler is executor.program_compiler
    assert executor.eager_executor.prefill is executor.prefill_runtime
    assert executor.eager_executor.decode is executor.decode_runtime
    assert executor.warmup.eager is executor.eager_executor
    assert executor.warmup.trace_compiler is executor.trace_compiler
    assert executor.eager_execution is executor.eager_executor
    assert executor.prefill_runtime.config.trace_capture_prime_sequence_lengths == (
        (128, 1024) if binding.executor_module is qwen3_32b_executor else ()
    )
    if mode == "none":
        assert executor.warmup.execution is executor.eager_executor
        assert executor.trace_compiler is None
        assert executor.traced_executor is None
        assert executor.traced_prefill_execution is None
        assert executor.traced_decode_execution is None
    else:
        assert executor.warmup.execution is executor.traced_executor
        assert executor.traced_executor.eager_executor is executor.eager_executor
        assert executor.traced_executor.trace_compiler is executor.trace_compiler
        assert executor.trace_compiler.program_compiler is executor.program_compiler
        assert (executor.traced_prefill_execution is not None) is (mode == "all")
        assert executor.traced_decode_execution is executor.traced_executor


def _device_sampling_executor(binding, monkeypatch, *, runtime_disable: bool):
    class FakeSampling1D:
        config = SimpleNamespace(
            is_resolved=lambda: True,
            allow_force_argmax=False,
            max_batch_size=32,
            max_top_k=32,
        )

        def decode_forward(self):
            raise AssertionError("construction-policy test must not execute sampling")

    sampling_policy_module = _sampling_policy_module(binding)
    monkeypatch.setattr(sampling_policy_module, "Sampling1D", FakeSampling1D)
    model = binding.make_model()
    model.sampling = FakeSampling1D()
    if binding.executor_module in (llama33_70b_executor, qwen3_32b_executor):

        class FakeSamplingState1D:
            def __init__(self, sampling):
                self.sampling = sampling
                self.seed_manager = SimpleNamespace()

            def create_state(self):
                return SimpleNamespace(seed_state=SimpleNamespace(capacity=32))

            def admit(self, *args, **kwargs):
                return None

            def decode_forward(self, *args, **kwargs):
                return None

            def release(self, *args, **kwargs):
                return None

        monkeypatch.setattr(sampling_policy_module, "SamplingState1D", FakeSamplingState1D)
    runtime_config = binding.make_runtime_config()
    runtime_config.disable_batched_prefill = runtime_disable
    config = replace(
        binding.make_executor_config("none"),
        device_sampling_enabled=True,
    )
    return binding.executor_class(model, runtime_config, config)


@pytest.mark.parametrize(
    ("runtime_disable", "environment_disable", "expected_kinds"),
    [
        (False, False, ("batched",)),
        (True, False, ("single", "single")),
        (False, True, ("single", "single")),
    ],
    ids=("device-sampled-batched", "runtime-disabled", "environment-disabled"),
)
def test_device_sampling_prefill_batch_policy_is_model_owned(
    binding,
    monkeypatch,
    runtime_disable,
    environment_disable,
    expected_kinds,
):
    if environment_disable:
        monkeypatch.setenv("DISABLE_BATCHED_PREFILL", "1")
    else:
        monkeypatch.delenv("DISABLE_BATCHED_PREFILL", raising=False)
    executor = _device_sampling_executor(
        binding,
        monkeypatch,
        runtime_disable=runtime_disable,
    )

    prepared = executor.prefill_runtime.prepare(
        tokens=torch.ones((2, 128), dtype=torch.long),
        page_table=torch.arange(8, dtype=torch.int32).reshape(2, 4),
        prompt_lens=torch.full((2,), 128, dtype=torch.long),
        empty_slots=[0, 1],
    )

    if binding.executor_module in (llama33_70b_executor, qwen3_32b_executor):
        expected_kinds = ("single", "single")
    assert tuple(item.request.kind for item in prepared) == expected_kinds
    if expected_kinds == ("batched",):
        assert prepared[0].request.source_rows == (0, 1)
        assert not prepared[0].request.uses_chunked_prefill


def test_llama32_1b_warms_every_q128_topk_tile_start_once_per_execution_mode():
    executor = object.__new__(llama32_executor.Llama32_1BExecutor)
    executor._q128_topk_tile_ends_warmed = set()
    executor.eager_executor = object()
    executor.traced_executor = object()
    executor.page_table_layout = SimpleNamespace(block_size=32)
    executor.prefill_runtime = SimpleNamespace(config=SimpleNamespace(static_q128_topk_supported=True))
    executor.warmup = SimpleNamespace(
        config=SimpleNamespace(
            prefill_sequence_lengths=(128,),
            prime_q128_tile_ends=False,
        )
    )
    executor.compile_prefill = MagicMock()
    kv_cache = object()

    for enable_trace in (False, False, True, True):
        executor._warmup_q128_topk_tile_ends(
            kv_cache=kv_cache,
            can_sample_on_device=True,
            enable_trace=enable_trace,
        )

    assert executor.compile_prefill.call_count == 6
    calls = executor.compile_prefill.call_args_list
    assert [call.kwargs["tokens"].shape[1] for call in calls] == [32, 64, 96, 32, 64, 96]
    assert [call.kwargs["page_table"].shape[1] for call in calls] == [1, 2, 3, 1, 2, 3]
    assert all(call.kwargs["kv_cache"] is kv_cache for call in calls)
    assert all(call.kwargs["execution"] is executor.eager_executor for call in calls[:3])
    assert all(call.kwargs["execution"] is executor.traced_executor for call in calls[3:])
    assert executor._q128_topk_tile_ends_warmed == {False, True}


@pytest.mark.parametrize(
    ("enable_trace", "expected_order"),
    [
        (False, ("default", 96, 0, 32, 64)),
        (True, (0, 32, 64, "default", 96)),
    ],
    ids=("eager", "traced"),
)
def test_qwen3_lane4_warms_every_runtime_q128_topk_signature_before_activation(enable_trace, expected_order):
    executor = SimpleNamespace(
        _q128_topk_tile_ends_warmed=set(),
        eager_executor=object(),
        traced_executor=object(),
        page_table_layout=SimpleNamespace(block_size=32),
        prefill_runtime=SimpleNamespace(config=SimpleNamespace(static_q128_topk_supported=True)),
        warmup=SimpleNamespace(
            config=SimpleNamespace(
                prefill_sequence_lengths=(128, 1024),
                prime_q128_tile_ends=False,
            )
        ),
    )
    compiled = []
    order = []
    activation = []

    def record_signature(prompt_length):
        assert not activation
        order.append(((prompt_length - 1) // 32) * 32)
        compiled.append(
            PrefillProgramSignature(
                operation_variant="regular-single",
                padded_batch_size=1,
                invocation_sequence_length=128,
                page_table_width=64,
                chunk_page_table_width=None,
                sampling_path="topk",
                penalties_enabled=False,
                logprobs_enabled=False,
                last_token_tile_start=((prompt_length - 1) // 32) * 32,
            )
        )

    def compile_prefill(**kwargs):
        expected_execution = executor.traced_executor if enable_trace else executor.eager_executor
        assert kwargs["execution"] is expected_execution
        assert kwargs["kv_cache"] == "cache"
        assert kwargs["sampling_params"].top_k.tolist() == [32]
        record_signature(int(kwargs["prompt_lens"][0]))

    executor.compile_prefill = compile_prefill

    default_warmed = False

    def default_warmup():
        nonlocal default_warmed
        if default_warmed:
            return
        default_warmed = True
        order.append("default")
        # The coordinator's ordinary Q128 top-k case covers the final tile.
        record_signature(128)

    for _ in range(2):
        qwen3_32b_executor._warmup_q128_around_prefill(
            executor,
            default_warmup,
            kv_cache="cache",
            can_sample_on_device=True,
            enable_trace=enable_trace,
        )
    activation.append(True)

    assert tuple(order) == expected_order
    assert {signature.last_token_tile_start for signature in compiled} == {0, 32, 64, 96}
    compiled_keys = {ProgramKey.from_signature(signature) for signature in compiled}
    runtime_signatures = {replace(compiled[0], last_token_tile_start=tile_start) for tile_start in (0, 32, 64, 96)}
    assert {ProgramKey.from_signature(signature) for signature in runtime_signatures} == compiled_keys
    assert executor._q128_topk_tile_ends_warmed == {enable_trace}


def test_qwen3_lane4_executor_installs_model_owned_q128_warmup(monkeypatch):
    executor = _device_sampling_executor(EXECUTOR_BINDINGS["qwen3_32b"], monkeypatch, runtime_disable=False)

    assert executor._prefill_warmup is qwen3_32b_executor._warmup_q128_around_prefill
    assert executor._q128_topk_tile_ends_warmed == set()


@pytest.mark.parametrize("device_sampling_enabled", (False, True), ids=("disabled", "enabled"))
def test_qwen3_generator_sampling_policy_controls_decode_topk_warmup(monkeypatch, device_sampling_enabled):
    mesh_device = _Mesh8()
    product = _make_qwen3_32b_product(mesh_device, max_batch_size=4)
    executor_configs = []

    monkeypatch.setattr(qwen3_32b_generator, "from_pretrained", lambda **kwargs: product)
    monkeypatch.setattr(
        qwen3_32b_generator,
        "_model_kv_metadata",
        lambda model: ((ttnn.bfloat8_b,), 1, 8, 64),
    )

    def build_executor(llm, config):
        executor_configs.append(config)
        return _FakeLane(
            llm,
            config,
            request_state_fields=qwen3_32b_executor.Qwen3_32BExecutor.request_state_fields,
        )

    monkeypatch.setattr(qwen3_32b_generator, "build_qwen3_32b_executor", build_executor)
    generator = qwen3_32b_generator.build_qwen3_32b_generator(
        qwen3_32b_generator.Qwen3_32BGeneratorConfig(
            hf_model="Qwen/Qwen3-32B",
            mesh_device=mesh_device,
            max_batch_size=4,
            max_seq_len=1024,
            trace_mode="decode_only",
            device_sampling_enabled=device_sampling_enabled,
        )
    )

    assert len(executor_configs) == 1
    assert executor_configs[0].warmup.include_decode_top_k is device_sampling_enabled
    if device_sampling_enabled:
        observed_missing_signature = DecodeTraceSignature(
            batch_size=4,
            page_table_width=32,
            sampling_path="topk",
            device_feedback=True,
        )
        assert (
            ProgramKey.from_signature(observed_missing_signature).digest
            == "6f8351f51a0c90eaea5fca6700b3887e380a015dad7ee6f6a9e8be971dfebbd5"
        )
    generator.cleanup()


@pytest.mark.parametrize(
    "method,positional,keyword_only",
    [
        (
            "compile_prefill",
            ["self"],
            [
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
                "execution",
            ],
        ),
        (
            "compile_decode",
            ["self"],
            [
                "tokens",
                "start_pos",
                "page_table",
                "kv_cache",
                "sampling_params",
                "prompt_tokens",
                "output_tokens",
                "slot_remap",
                "reset_batch",
                "execution",
            ],
        ),
        (
            "prefill_forward",
            ["self", "tokens", "page_table"],
            [
                "prompt_lens",
                "start_pos",
                "empty_slots",
                "kv_cache",
                "sampling_params",
                "prompt_tokens",
                "output_tokens",
                "slot_remap",
                "execution",
            ],
        ),
        (
            "decode_forward",
            ["self", "tokens", "start_pos", "page_table"],
            [
                "kv_cache",
                "sampling_params",
                "prompt_tokens",
                "output_tokens",
                "slot_remap",
                "reset_batch",
                "read_from_device",
                "execution",
            ],
        ),
        ("read_decode_output", ["self", "tt_out"], ["async_read"]),
        ("process_decode_output_host", ["self", "tt_out"], ["is_tokens"]),
        ("can_trace_prefill", ["self"], ["tokens", "prompt_lens", "start_pos", "empty_slots"]),
        ("warmup_model_prefill", ["self"], ["kv_cache", "can_sample_on_device", "enable_trace"]),
        (
            "warmup_model_decode",
            ["self"],
            ["kv_cache", "max_batch_size", "num_blocks", "can_sample_on_device", "enable_trace"],
        ),
    ],
)
def test_executor_call_contract(binding, method, positional, keyword_only):
    if binding.executor_module not in (llama33_70b_executor, qwen3_32b_executor):
        keyword_only = [name for name in keyword_only if name not in {"prompt_tokens", "output_tokens", "slot_remap"}]
    signature = inspect.signature(getattr(binding.executor_class, method))
    parameters = signature.parameters
    required = {
        "compile_prefill": {"tokens", "page_table"},
        "compile_decode": {"tokens", "start_pos", "page_table"},
        "prefill_forward": {"tokens", "page_table"},
        "decode_forward": {"tokens", "start_pos", "page_table"},
        "read_decode_output": {"tt_out"},
        "process_decode_output_host": {"tt_out"},
        "can_trace_prefill": {"tokens"},
        "warmup_model_prefill": {"kv_cache", "can_sample_on_device", "enable_trace"},
        "warmup_model_decode": {
            "kv_cache",
            "max_batch_size",
            "num_blocks",
            "can_sample_on_device",
            "enable_trace",
        },
    }[method]
    non_none_defaults = {"reset_batch": False, "read_from_device": True, "async_read": False, "is_tokens": False}

    assert list(parameters) == positional + keyword_only
    assert all(parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD for name in positional)
    assert all(parameters[name].kind is inspect.Parameter.KEYWORD_ONLY for name in keyword_only)
    for name, parameter in tuple(parameters.items())[1:]:
        expected_default = inspect.Parameter.empty if name in required else non_none_defaults.get(name)
        assert parameter.default == expected_default
        assert parameter.annotation is not inspect.Parameter.empty
    assert signature.return_annotation is not inspect.Signature.empty


@pytest.mark.parametrize(
    "executor_class",
    [
        qwen3_32b_executor.Qwen3_32BExecutor,
    ],
)
def test_qwen3_delegates_resolved_sampling_warmup_and_activation_to_coordinator(executor_class):
    warmup = SimpleNamespace(warmup_prefill=MagicMock(), warmup_decode=MagicMock())
    executor = SimpleNamespace(_ensure_active=MagicMock(), warmup=warmup)

    executor_class.warmup_model_prefill(
        executor,
        kv_cache="cache",
        can_sample_on_device=True,
        enable_trace=True,
    )
    executor_class.warmup_model_decode(
        executor,
        kv_cache="cache",
        max_batch_size=32,
        num_blocks=64,
        can_sample_on_device=True,
        enable_trace=True,
    )

    warmup.warmup_prefill.assert_called_once_with(
        kv_cache="cache",
        can_sample_on_device=True,
        enable_trace=True,
    )
    warmup.warmup_decode.assert_called_once_with(
        kv_cache="cache",
        max_batch_size=32,
        num_blocks=64,
        can_sample_on_device=True,
        enable_trace=True,
    )
    assert "capture_all" not in inspect.getsource(executor_class.warmup_model_decode)


class _RecordingTarget:
    model_args = object()
    mesh_device = object()
    cache_path = "cache"
    already_warmed_up_prefill = False
    eager_execution = object()
    traced_prefill_execution = object()
    traced_decode_execution = object()

    def __init__(self, model, traceable=True, request_state_fields=()):
        self.model = model
        self.traceable = traceable
        self._request_state_fields = tuple(request_state_fields)
        self.calls = []

    def can_trace_prefill(self, **kwargs):
        self.calls.append(("can_trace_prefill", kwargs))
        return self.traceable

    def prefill_forward(self, **kwargs):
        self.calls.append(("prefill_forward", kwargs))
        return kwargs["execution"]

    def decode_forward(self, **kwargs):
        self.calls.append(("decode_forward", kwargs))
        return kwargs["execution"]

    def cleanup(self):
        self.calls.append(("cleanup", {}))


def test_generator_preserves_required_trace_intent_for_ineligible_prefill(binding):
    target = binding.make_recording_target(traceable=False)
    target.config = binding.make_executor_config("all")
    generator = binding.generator_class(target, binding.generator_module._build_vllm_adapter(target))
    tokens = __import__("torch").tensor([[1]])
    page_table = __import__("torch").tensor([[0]], dtype=__import__("torch").int32)
    assert generator.prefill_forward(tokens, page_table, enable_trace=True) is target.traced_prefill_execution
    assert [name for name, _ in target.calls] == ["prefill_forward"]


def test_generator_routes_external_decode_only_policy_with_all_trace_targets(binding):
    target = binding.make_recording_target()
    target.config = binding.make_executor_config("all")
    generator = binding.generator_class(target, binding.generator_module._build_vllm_adapter(target))
    torch = __import__("torch")
    tokens = torch.tensor([[1]])
    page_table = torch.tensor([[0]], dtype=torch.int32)
    start_pos = torch.tensor([0])

    assert generator.prefill_forward(tokens, page_table, enable_trace=False) is target.eager_execution
    assert (
        generator.decode_forward(tokens[:, 0], start_pos, page_table, enable_trace=True)
        is target.traced_decode_execution
    )
    assert [name for name, _ in target.calls] == ["prefill_forward", "decode_forward"]


def test_executor_validates_borrowed_cache_then_omits_it_from_execution(binding):
    execution = create_autospec(EagerExecutor, instance=True)
    events = []
    execution.compile_prefill.side_effect = lambda **kwargs: events.append("dispatch_compile_prefill")
    execution.compile_decode.side_effect = lambda **kwargs: events.append("dispatch_compile_decode")
    execution.prefill_forward.side_effect = lambda **kwargs: events.append("dispatch_prefill") or "prefill"
    execution.decode_forward.side_effect = lambda **kwargs: events.append("dispatch_decode") or "decode"
    executor = object.__new__(binding.executor_class)
    executor._prefill_execution = execution
    executor._decode_execution = execution
    executor._ensure_active = lambda: None
    executor._validate_bound_cache = lambda cache: events.append(("validate_cache", cache))
    executor._ensure_sampling_for = lambda params: events.append(("validate_sampling", params))

    tokens = torch.zeros((1, 4), dtype=torch.long)
    start_pos = torch.zeros((1,), dtype=torch.long)
    page_table = torch.zeros((1, 1), dtype=torch.int32)
    prompt_lens = torch.full((1,), 4, dtype=torch.long)
    empty_slots = [0]
    kv_cache = object()
    sampling_params = object()

    executor.compile_prefill(
        tokens=tokens,
        page_table=page_table,
        prompt_lens=prompt_lens,
        start_pos=start_pos,
        empty_slots=empty_slots,
        kv_cache=kv_cache,
        sampling_params=sampling_params,
    )
    executor.compile_decode(
        tokens=tokens,
        start_pos=start_pos,
        page_table=page_table,
        kv_cache=kv_cache,
        sampling_params=sampling_params,
        reset_batch=True,
    )
    assert (
        executor.prefill_forward(
            tokens,
            page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )
        == "prefill"
    )
    assert (
        executor.decode_forward(
            tokens,
            start_pos,
            page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            reset_batch=True,
            read_from_device=False,
        )
        == "decode"
    )

    expected_validation = [("validate_cache", kv_cache), ("validate_sampling", sampling_params)]
    assert events == [
        *expected_validation,
        "dispatch_compile_prefill",
        *expected_validation,
        "dispatch_compile_decode",
        *expected_validation,
        "dispatch_prefill",
        *expected_validation,
        "dispatch_decode",
    ]
    request_state_names = (
        ("prompt_tokens", "output_tokens", "slot_remap")
        if "prompt_tokens" in inspect.signature(binding.executor_class.compile_prefill).parameters
        else ()
    )
    for target, expected_names in (
        (
            execution.compile_prefill,
            (
                "tokens",
                "page_table",
                "prompt_lens",
                "start_pos",
                "empty_slots",
                "sampling_params",
                *request_state_names,
            ),
        ),
        (
            execution.compile_decode,
            ("tokens", "start_pos", "page_table", "sampling_params", *request_state_names, "reset_batch"),
        ),
        (
            execution.prefill_forward,
            (
                "tokens",
                "page_table",
                "prompt_lens",
                "start_pos",
                "empty_slots",
                "sampling_params",
                *request_state_names,
            ),
        ),
        (
            execution.decode_forward,
            (
                "tokens",
                "start_pos",
                "page_table",
                "sampling_params",
                *request_state_names,
                "reset_batch",
                "read_from_device",
            ),
        ),
    ):
        assert target.call_count == 1
        assert tuple(target.call_args.kwargs) == expected_names
        assert "kv_cache" not in target.call_args.kwargs


def test_late_capacity_reconfigures_existing_owners_before_allocation(binding, monkeypatch):
    executor = binding.executor_class(
        binding.make_model(), binding.make_runtime_config(), binding.make_executor_config()
    )
    owner_ids = tuple(
        id(owner)
        for owner in (executor.prefill_runtime, executor.decode_runtime, executor.warmup, executor.program_compiler)
    )
    assert executor.page_table_layout.raw_capacity_width == 128

    executor.configure_paged_kv_cache(
        PagedKVCacheConfig(
            block_size=16,
            max_num_blocks=200,
            dtype=ttnn.bfloat8_b,
            num_blocks=200,
        )
    )

    assert (
        tuple(
            id(owner)
            for owner in (executor.prefill_runtime, executor.decode_runtime, executor.warmup, executor.program_compiler)
        )
        == owner_ids
    )
    assert executor.config.paged_kv_cache is executor.kv_cache_manager.config
    assert executor.kv_cache_manager.config.block_size == 16
    assert executor.kv_cache_manager.config.max_num_blocks == executor.kv_cache_manager.config.num_blocks == 200
    assert executor.model.layers[0].attention.config.paged_attention_config.block_size == 16
    assert executor.model.layers[0].attention.config.paged_attention_config.max_num_blocks == 200
    assert executor.page_table_layout.block_size == 16
    assert executor.page_table_layout.raw_capacity_width == 200
    assert executor.prefill_runtime.config.page_table_layout is executor.page_table_layout
    assert executor.decode_runtime.config.page_table_layout is executor.page_table_layout
    assert executor.warmup.config.page_table_layout is executor.page_table_layout
    assert executor.prefill_runtime.config.trace_capture_prime_sequence_lengths == (
        (128, 1024) if binding.executor_module is qwen3_32b_executor else ()
    )

    def fake_allocate():
        assert executor._runtime_configuration_sealed
        assert executor.warmup._configuration_sealed
        return ["allocated"]

    monkeypatch.setattr(executor.kv_cache_manager, "allocate", fake_allocate)
    assert executor.allocate_kv_cache() == ["allocated"]


def test_late_capacity_failure_is_atomic(binding, expect_error):
    executor = binding.executor_class(
        binding.make_model(), binding.make_runtime_config(), binding.make_executor_config()
    )
    executor._seal_runtime_configuration()
    unresolved = executor.kv_cache_manager.config
    original_layout = executor.page_table_layout
    original_model_paged = executor.model.layers[0].attention.config.paged_attention_config

    with expect_error(RuntimeError, "runtime configuration is sealed"):
        executor.configure_paged_kv_cache(
            PagedKVCacheConfig(
                block_size=32,
                max_num_blocks=132,
                dtype=ttnn.bfloat8_b,
                num_blocks=64,
            )
        )

    assert executor.kv_cache_manager.config is unresolved
    assert not unresolved.is_resolved()
    assert executor.page_table_layout is original_layout
    assert executor.model.layers[0].attention.config.paged_attention_config is original_model_paged


def test_generator_resolves_configures_then_allocates_vllm_kv_shape(binding):
    events = []
    resolved = object()
    cache = object()
    shape = (129, 8, 64, 128)
    dtype = object()
    target = SimpleNamespace(
        configure_paged_kv_cache=lambda config: events.append(("configure", config)),
        allocate_kv_cache=lambda: events.append(("allocate",)) or cache,
    )
    adapter = SimpleNamespace(
        resolve_legacy_kv_cache_config=lambda *args: events.append(("resolve", args)) or resolved,
    )
    generator = binding.generator_class(target, adapter)

    assert generator.allocate_kv_cache(shape, dtype, 32) is cache
    assert events == [
        ("resolve", (shape, dtype, 32)),
        ("configure", resolved),
        ("allocate",),
    ]


def test_generator_allocates_model_owned_kv_without_reconfiguration(binding):
    events = []
    cache = object()
    target = SimpleNamespace(
        configure_paged_kv_cache=lambda config: events.append(("configure", config)),
        allocate_kv_cache=lambda: events.append(("allocate",)) or cache,
    )
    adapter = SimpleNamespace(
        resolve_legacy_kv_cache_config=lambda *args: events.append(("resolve", args)),
    )
    generator = binding.generator_class(target, adapter)

    assert generator.allocate_kv_cache() is cache
    assert events == [("allocate",)]


@pytest.mark.parametrize(
    "arguments",
    (
        ((64, 8, 32, 128), None, None),
        (None, object(), None),
        (None, None, 32),
        ((64, 8, 32, 128), object(), None),
    ),
)
def test_generator_rejects_partial_vllm_kv_shape_atomically(binding, arguments, expect_error):
    events = []
    target = SimpleNamespace(
        configure_paged_kv_cache=lambda config: events.append(("configure", config)),
        allocate_kv_cache=lambda: events.append(("allocate",)),
    )
    adapter = SimpleNamespace(
        resolve_legacy_kv_cache_config=lambda *args: events.append(("resolve", args)),
    )
    generator = binding.generator_class(target, adapter)

    with expect_error(TypeError, "must be supplied together"):
        generator.allocate_kv_cache(*arguments)

    assert events == []


def test_generator_does_not_configure_or_allocate_after_vllm_kv_resolution_failure(binding, expect_error):
    events = []

    def fail_resolution(*args):
        events.append(("resolve", args))
        raise ValueError("invalid vLLM KV geometry")

    target = SimpleNamespace(
        configure_paged_kv_cache=lambda config: events.append(("configure", config)),
        allocate_kv_cache=lambda: events.append(("allocate",)),
    )
    generator = binding.generator_class(
        target,
        SimpleNamespace(resolve_legacy_kv_cache_config=fail_resolution),
    )

    with expect_error(ValueError, "invalid vLLM KV geometry"):
        generator.allocate_kv_cache((129, 8, 64, 128), object(), 32)

    assert tuple(name for name, *_ in events) == ("resolve",)


def test_generator_reports_unmultiplied_per_submesh_token_capacity(binding):
    assert (
        binding.generator_class.get_max_tokens_all_users(
            model_name="ignored",
            num_devices=8,
            tt_data_parallel=4,
            max_model_len=32768,
            max_num_seqs=64,
        )
        == 32768
    )


def test_generator_rejects_unavailable_traced_execution(binding, expect_error):
    target = binding.make_recording_target()
    target.traced_decode_execution = None
    target.config = binding.make_executor_config("none")
    generator = binding.generator_class(target, binding.generator_module._build_vllm_adapter(target))

    with expect_error(RuntimeError, "unavailable traced decode execution"):
        generator._select_execution("decode", True)


def test_initialize_vllm_model_threads_policy(binding, monkeypatch):
    captured = []
    sentinel = object()
    mesh_device = object()
    monkeypatch.setattr(
        binding.generator_module,
        binding.build_generator_name,
        lambda config: captured.append(config) or sentinel,
    )

    result = binding.generator_class.initialize_vllm_model(
        SimpleNamespace(_name_or_path=binding.hf_model),
        mesh_device,
        8,
        4096,
        n_layers=3,
        tt_data_parallel=2,
        optimizations="accuracy",
        trace_mode="decode_only",
        device_sampling_enabled=True,
    )

    assert result is sentinel
    config = captured[0]
    assert isinstance(config, binding.generator_config_class)
    assert config.hf_model == binding.hf_model
    assert config.mesh_device is mesh_device
    assert config.max_batch_size == 8
    assert config.max_seq_len == 4096
    assert config.n_layers == 3
    assert config.tt_data_parallel == 2
    assert config.optimizations == "accuracy"
    assert config.trace_mode == "decode_only"
    assert config.device_sampling_enabled is True


@pytest.mark.parametrize("model_id,generator_path", GENERATOR_PATHS.items(), ids=GENERATOR_PATHS)
def test_vllm_generator_path_and_construction_defaults(model_id, generator_path):
    module_name, class_name = generator_path.split(":", maxsplit=1)
    generator_class = getattr(import_module(module_name), class_name)

    assert generator_class is EXECUTOR_BINDINGS[model_id].generator_class
    assert callable(getattr(generator_class, "initialize_vllm_model", None))

    parameters = inspect.signature(generator_class.initialize_vllm_model).parameters
    assert parameters["trace_mode"].default == "all"
    assert parameters["device_sampling_enabled"].default is True


class _FakeLane:
    requires_prefill_trace_warmup = True

    def __init__(self, llm, config, request_state_fields=()):
        self.model = llm.model
        self.model_args = llm.runtime_config
        self.mesh_device = llm.model.config.mesh_device
        self.cache_path = llm.runtime_config.model_cache_path
        self.config = config
        self._request_state_fields = tuple(request_state_fields)
        self.paged_kv_cache_config = config.paged_kv_cache
        self.already_warmed_up_prefill = False
        self.eager_execution = object()
        self.traced_prefill_execution = object()
        self.traced_decode_execution = object()
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1


def test_generator_constructs_data_parallel_lane_group(binding, monkeypatch):
    executor_calls = []
    built_lanes = []
    pretrained_calls = []
    parent_mesh = object()
    submeshes = [_Mesh(), _Mesh()]
    create_submeshes = MagicMock(return_value=submeshes)
    monkeypatch.setattr(binding.generator_module, "_create_submeshes", create_submeshes)

    def fake_from_pretrained(mesh_device, **kwargs):
        pretrained_calls.append((mesh_device, kwargs))
        return binding.make_product(mesh_device, kwargs["max_batch_size"])

    def fake_build_executor(llm, config):
        executor_calls.append((llm, config))
        lane = binding.make_lane(llm, config)
        built_lanes.append(lane)
        return lane

    monkeypatch.setattr(binding.generator_module, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(binding.generator_module, binding.build_executor_name, fake_build_executor)
    monkeypatch.setattr(
        binding.generator_module,
        "_model_kv_metadata",
        lambda model: ((ttnn.bfloat8_b,), 1, 8, 64),
    )

    generator = getattr(binding.generator_module, binding.build_generator_name)(
        binding.generator_config_class(
            hf_model=binding.hf_model,
            mesh_device=parent_mesh,
            max_batch_size=4,
            max_seq_len=4096,
            n_layers=1,
            tt_data_parallel=2,
            trace_mode="all",
            device_sampling_enabled=True,
        )
    )

    try:
        create_submeshes.assert_called_once_with(parent_mesh, 2)
        assert [mesh for mesh, _ in pretrained_calls] == submeshes
        assert all(call[1]["max_batch_size"] == 2 for call in pretrained_calls)
        assert all(call[1]["max_seq_len"] == 4096 for call in pretrained_calls)
        assert all(call[1]["n_layers"] == 1 for call in pretrained_calls)
        assert isinstance(generator.target, LaneGroupExecutor)
        assert generator.target.mesh_device is parent_mesh
        assert generator.target.tt_data_parallel == 2
        assert len(executor_calls) == 2
        assert executor_calls[0][0] is not executor_calls[1][0]
        assert generator.target.lanes == built_lanes
        assert [lane.model for lane in generator.target.lanes] == [llm.model for llm, _ in executor_calls]
        assert [lane.mesh_device for lane in generator.target.lanes] == submeshes
        assert len({id(lane) for lane in generator.target.lanes}) == 2
        assert all(isinstance(config, binding.executor_config_class) for _, config in executor_calls)
        assert all(llm.model.config.max_batch_size == 2 for llm, _ in executor_calls)
        assert generator._adapter.config.trace.mode == "all"
        assert generator._adapter.config.expected_num_layers == 1
        assert generator._adapter.config.expected_kv_heads_per_device == 8
        assert generator._adapter.config.expected_head_dim == 64
    finally:
        generator.cleanup()


def test_executor_cleanup_is_ordered_retryable_and_idempotent(binding, expect_error):
    calls = []
    failures = {"reader", "trace"}

    class _Owner:
        def __init__(self, name):
            self.name = name

        def cleanup(self, *args):
            calls.append(self.name)
            if self.name in failures:
                raise RuntimeError(self.name)

        drain = cleanup
        drain_external_outputs = cleanup
        cleanup_transients = cleanup
        release = cleanup

    executor = object.__new__(binding.executor_class)
    executor._terminal = False
    executor._cleaned_up = False
    executor.decode_runtime = _Owner("decode-external")
    executor.output_reader = _Owner("reader")
    executor.prefill_runtime = _Owner("prefill")
    executor.trace_compiler = _Owner("trace")
    executor.program_compiler = _Owner("program")
    executor.config = SimpleNamespace(device_sampling_enabled=True)
    executor.model = SimpleNamespace(sampling=_Owner("sampling"))
    if binding.executor_module in (llama33_70b_executor, qwen3_32b_executor):
        executor.sampling_state_controller = _Owner("sampling-state")
        executor.sampling_state = object()
    else:
        executor.sampling_state_controller = None
        executor.sampling_state = None
    executor.kv_cache_manager = _Owner("kv")

    with expect_error(RuntimeError, "reader") as raised:
        executor.cleanup()

    expected_order = [
        "decode-external",
        "reader",
        "prefill",
        "decode-external",
        "trace",
        "program",
    ]
    if binding.executor_module in (llama33_70b_executor, qwen3_32b_executor):
        expected_order.append("sampling-state")
    expected_order.extend(["sampling", "kv"])
    assert calls == expected_order
    assert tuple(error.args[0] for error in raised.value.cleanup_failures) == ("trace",)
    assert executor.terminal
    assert not executor._cleaned_up

    failures.clear()
    executor.cleanup()
    assert calls == expected_order * 2
    assert executor._cleaned_up

    executor.cleanup()
    assert calls == expected_order * 2


def test_llama33_generator_emits_runtime_summary_before_owned_cleanup():
    events = []
    traced = SimpleNamespace(log_runtime_summary=lambda **kwargs: events.append(("summary", kwargs)))
    target = SimpleNamespace(
        traced_executor=traced,
        cleanup=lambda: events.append("cleanup"),
    )
    generator = llama33_70b_generator.Llama33_70BGenerator(target, SimpleNamespace())

    generator.cleanup()

    assert events == [("summary", {"phase": "shutdown"}), "cleanup"]


def test_llama33_generator_emits_serving_ready_and_idempotent_shutdown_summaries():
    phases = []
    trace_compiler = SimpleNamespace(trace_active=False)
    traced = SimpleNamespace(
        trace_compiler=trace_compiler,
        log_runtime_summary=lambda **kwargs: phases.append(kwargs["phase"]),
    )
    target = SimpleNamespace(
        traced_executor=traced,
        warmup_model_prefill=lambda **kwargs: None,
        warmup_model_decode=lambda **kwargs: None,
        cleanup=lambda: None,
    )
    generator = llama33_70b_generator.Llama33_70BGenerator(target, SimpleNamespace())

    generator.warmup_model_prefill(kv_cache="cache", can_sample_on_device=True, enable_trace=True)
    trace_compiler.trace_active = True
    generator.warmup_model_decode(
        kv_cache="cache",
        max_batch_size=16,
        num_blocks=128,
        can_sample_on_device=True,
        enable_trace=True,
    )
    generator.warmup_model_prefill(kv_cache="cache", can_sample_on_device=True, enable_trace=True)
    generator._shutdown_summary_callback()
    generator.cleanup()

    assert phases == ["serving_ready", "shutdown"]
