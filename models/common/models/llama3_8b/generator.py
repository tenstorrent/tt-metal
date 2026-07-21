# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""vLLM construction and compatibility delegation for Llama 3.1-8B."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ttnn
from models.common.llm_runtime.config import LLMExecutorConfig, PagedKVCacheConfig, TraceConfig, TraceMode, WarmupConfig
from models.common.llm_runtime.lane_group import LaneGroupExecutor
from models.common.llm_runtime.vllm_adapter import VLLMAdapter
from models.common.models.llama3_8b.executor import build_llama3_executor
from models.common.models.llama3_8b.hf_adaptor import from_pretrained
from models.common.models.llama3_8b.model import Llama31_8BPagedAttentionConfig

_VLLM_BLOCK_SIZE = 32


@dataclass(frozen=True)
class Llama3GeneratorConfig:
    """Validated construction inputs for one vLLM-facing Llama generator."""

    hf_model: str
    mesh_device: Any
    max_batch_size: int
    max_seq_len: int
    n_layers: int | None = None
    tt_data_parallel: int = 1
    optimizations: Any = "performance"
    trace_mode: TraceMode = "all"
    device_sampling_enabled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.hf_model, str) or not self.hf_model:
            raise ValueError("hf_model must be a non-empty string")
        if self.mesh_device is None:
            raise ValueError("mesh_device is required")
        _validate_positive_int("max_batch_size", self.max_batch_size)
        _validate_positive_int("max_seq_len", self.max_seq_len)
        _validate_positive_int("tt_data_parallel", self.tt_data_parallel)
        if self.n_layers is not None:
            _validate_positive_int("n_layers", self.n_layers)
        if self.max_batch_size % self.tt_data_parallel != 0:
            raise ValueError(
                f"max_batch_size={self.max_batch_size} must be divisible by "
                f"tt_data_parallel={self.tt_data_parallel}"
            )
        if not isinstance(self.device_sampling_enabled, bool):
            raise TypeError("device_sampling_enabled must be bool")
        TraceConfig(mode=self.trace_mode)


class Llama3Generator:
    """Normalize vLLM calls and delegate to one duck-typed execution target."""

    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
        "required_block_size": _VLLM_BLOCK_SIZE,
    }
    requires_prefill_trace_warmup = True

    def __init__(self, target: Any, adapter: VLLMAdapter):
        self.target = target
        self._adapter = adapter

    @property
    def model(self):
        return self.target.model

    @property
    def model_args(self):
        return self.target.model_args

    @property
    def mesh_device(self):
        return self.target.mesh_device

    @property
    def cache_path(self):
        return self.target.cache_path

    @property
    def already_warmed_up_prefill(self):
        return self.target.already_warmed_up_prefill

    @already_warmed_up_prefill.setter
    def already_warmed_up_prefill(self, value):
        self.target.already_warmed_up_prefill = value

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int = 0,
        max_num_seqs: int = 1,
        **kwargs,
    ) -> int:
        """Return the unpadded per-submesh KV token budget for vLLM sizing."""

        return int(max_model_len)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations="performance",
        trace_mode: TraceMode = "all",
        device_sampling_enabled: bool = True,
    ):
        """Build the configured single-lane or data-parallel Llama target."""

        hf_model = getattr(hf_config, "_name_or_path", None)
        if not hf_model:
            raise ValueError("hf_config must provide a non-empty _name_or_path")
        return build_llama3_generator(
            Llama3GeneratorConfig(
                hf_model=str(hf_model),
                mesh_device=mesh_device,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                n_layers=n_layers,
                tt_data_parallel=tt_data_parallel,
                optimizations=optimizations,
                trace_mode=trace_mode,
                device_sampling_enabled=device_sampling_enabled,
            )
        )

    def allocate_kv_cache(self, kv_cache_shape=None, dtype=None, num_layers=None):
        """Resolve the late vLLM capacity, then allocate a borrowed cache handle."""

        supplied = (kv_cache_shape is not None, dtype is not None, num_layers is not None)
        if not any(supplied):
            return self.target.allocate_kv_cache()
        if not all(supplied):
            raise TypeError("kv_cache_shape, dtype, and num_layers must be supplied together")

        resolved = self._adapter.resolve_legacy_kv_cache_config(kv_cache_shape, dtype, num_layers)
        self.target.configure_paged_kv_cache(resolved)
        return self.target.allocate_kv_cache()

    def compile_prefill(self, *args, **kwargs):
        return self.target.compile_prefill(**self._adapter.normalize_prefill(args, kwargs))

    def compile_decode(self, *args, **kwargs):
        return self.target.compile_decode(**self._adapter.normalize_decode(args, kwargs))

    def prefill_forward(self, *args, **kwargs):
        return self.target.prefill_forward(**self._adapter.normalize_prefill(args, kwargs))

    def decode_forward(self, *args, **kwargs):
        return self.target.decode_forward(**self._adapter.normalize_decode(args, kwargs))

    def read_decode_output(self, *args, **kwargs):
        return self.target.read_decode_output(*args, **kwargs)

    def process_decode_output_host(self, *args, **kwargs):
        return self.target.process_decode_output_host(*args, **kwargs)

    def warmup_model_prefill(self, *args, **kwargs):
        return self.target.warmup_model_prefill(*args, **kwargs)

    def warmup_model_decode(self, *args, **kwargs):
        return self.target.warmup_model_decode(*args, **kwargs)

    def cleanup(self):
        return self.target.cleanup()


def build_llama3_generator(config: Llama3GeneratorConfig) -> Llama3Generator:
    """Construct lane-local models/executors and compose their shared target surface."""

    per_lane_max_batch_size = config.max_batch_size // config.tt_data_parallel
    submeshes = (
        [config.mesh_device]
        if config.tt_data_parallel == 1
        else list(_create_submeshes(config.mesh_device, config.tt_data_parallel))
    )
    if len(submeshes) != config.tt_data_parallel:
        raise ValueError(f"Expected {config.tt_data_parallel} submeshes, got {len(submeshes)}")

    max_num_blocks = (config.max_seq_len + _VLLM_BLOCK_SIZE - 1) // _VLLM_BLOCK_SIZE + per_lane_max_batch_size
    lanes = []
    try:
        for submesh in submeshes:
            paged_attention_config = Llama31_8BPagedAttentionConfig(
                block_size=_VLLM_BLOCK_SIZE,
                max_num_blocks=max_num_blocks,
            )
            llm = from_pretrained(
                mesh_device=submesh,
                hf_model=config.hf_model,
                instruct="Instruct" in config.hf_model,
                max_batch_size=per_lane_max_batch_size,
                max_seq_len=config.max_seq_len,
                optimizations=config.optimizations,
                n_layers=config.n_layers,
                dtype=ttnn.bfloat8_b,
                paged_attention_config=paged_attention_config,
            )
            model_kv_cache_dtypes, _, _, _ = _model_kv_metadata(llm.model)
            executor_config = LLMExecutorConfig(
                trace=TraceConfig(mode=config.trace_mode),
                warmup=WarmupConfig(),
                paged_kv_cache=PagedKVCacheConfig(
                    block_size=_VLLM_BLOCK_SIZE,
                    max_num_blocks=max_num_blocks,
                    dtype=model_kv_cache_dtypes[0],
                ),
                device_sampling_enabled=config.device_sampling_enabled,
            )
            lanes.append(build_llama3_executor(llm, executor_config))

        adapter = _build_vllm_adapter(lanes[0])
    except BaseException as primary:
        _cleanup_after_construction_failure(lanes, primary)
        raise

    target = lanes[0] if config.tt_data_parallel == 1 else LaneGroupExecutor(lanes, mesh_device=config.mesh_device)
    return Llama3Generator(target, adapter)


def _build_vllm_adapter(lane) -> VLLMAdapter:
    model_kv_cache_dtypes, num_layers, kv_heads_per_device, head_dim = _model_kv_metadata(lane.model)
    return VLLMAdapter(
        trace_config=lane.config.trace,
        paged_kv_cache_config=lane.config.paged_kv_cache,
        expected_num_layers=num_layers,
        expected_kv_heads_per_device=kv_heads_per_device,
        expected_head_dim=head_dim,
        model_kv_cache_dtype=model_kv_cache_dtypes,
    )


def _model_kv_metadata(model) -> tuple[tuple[Any, ...], int, int, int]:
    layers = tuple(getattr(model, "layers", ()))
    if not layers:
        raise ValueError("Llama model must contain at least one attention layer")

    attention_configs = tuple(layer.attention.config for layer in layers)
    model_config = model.config
    num_layers = int(model_config.n_layers)
    if len(attention_configs) != num_layers:
        raise ValueError(f"Model config declares {num_layers} layers but exposes {len(attention_configs)}")

    num_devices = int(model_config.num_devices)
    n_kv_heads = int(attention_configs[0].n_kv_heads)
    if n_kv_heads % num_devices != 0:
        raise ValueError(f"n_kv_heads={n_kv_heads} must be divisible by num_devices={num_devices}")

    head_dim = int(attention_configs[0].head_dim)
    if any(
        int(attention_config.n_kv_heads) != n_kv_heads or int(attention_config.head_dim) != head_dim
        for attention_config in attention_configs
    ):
        raise ValueError("Every Llama layer must expose the same KV head shape")

    return (
        tuple(attention_config.kv_cache_dtype for attention_config in attention_configs),
        num_layers,
        n_kv_heads // num_devices,
        head_dim,
    )


def _create_submeshes(mesh_device, tt_data_parallel):
    from models.tt_transformers.tt.generator import create_submeshes

    return create_submeshes(mesh_device, tt_data_parallel)


def _cleanup_after_construction_failure(lanes, primary):
    failures = []
    for lane in lanes:
        try:
            lane.cleanup()
        except BaseException as error:
            failures.append(error)
    if failures:
        setattr(primary, "cleanup_failures", failures)


def _validate_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
