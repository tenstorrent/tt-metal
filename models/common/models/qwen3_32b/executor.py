# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Model-owned Qwen3-32B policy over the shared model executor.

Qwen3-32B is currently the only Qwen3 product in ``models/common/models``.
This module therefore retains its concrete sampling, trace-prime, and
compatibility policy while delegating the family-neutral lifecycle to
``ModelExecutor``. The public class is a composition facade, not a subclass.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

import torch

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.models.executor import ModelExecutor, ModelExecutorConfig
from models.common.models.qwen3_32b.hf_adaptor import Qwen3_32BForCausalLM
from models.common.models.qwen3_32b.model import _slice_last_token_tile
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.modules.sampling.sampling_state_1d import SamplingState1D
from models.common.sampling.sampling_params import SamplingParams


@dataclass(frozen=True)
class Qwen3_32BExecutorConfig(ModelExecutorConfig):
    """Immutable execution policy for Qwen3-32B."""


def _facade_target(facade: Any) -> Any:
    """Return the composed owner, or a test-fabricated facade itself."""

    return getattr(facade, "__dict__", {}).get("_model_executor", facade)


def _delegate_to_model_executor(method_name: str) -> Callable[..., Any]:
    """Create one signature-preserving facade method."""

    model_executor_method = getattr(ModelExecutor, method_name)

    @wraps(model_executor_method)
    def delegated(self, *args, **kwargs):
        return model_executor_method(_facade_target(self), *args, **kwargs)

    return delegated


class _ExecutorFacadeSurface:
    """Descriptors copied onto the public facade without inheritance."""

    @property
    def model_config(self):
        target = _facade_target(self)
        return target.model.config if target is self else target.model_config

    @property
    def cluster_shape(self) -> list[int]:
        target = _facade_target(self)
        return list(target.mesh_device.shape) if target is self else target.cluster_shape

    @property
    def paged_kv_cache_config(self) -> PagedKVCacheConfig:
        target = _facade_target(self)
        return target.kv_cache_manager.config if target is self else target.paged_kv_cache_config

    @property
    def terminal(self) -> bool:
        target = _facade_target(self)
        return bool(target._terminal) if target is self else target.terminal

    @property
    def already_warmed_up_prefill(self) -> bool:
        target = _facade_target(self)
        return target.warmup.already_warmed_up_prefill if target is self else target.already_warmed_up_prefill

    @already_warmed_up_prefill.setter
    def already_warmed_up_prefill(self, value: bool) -> None:
        target = _facade_target(self)
        if target is self:
            return ModelExecutor.already_warmed_up_prefill.fset(target, value)
        target.already_warmed_up_prefill = value

    configure_paged_kv_cache = _delegate_to_model_executor("configure_paged_kv_cache")
    allocate_kv_cache = _delegate_to_model_executor("allocate_kv_cache")
    compile_prefill = _delegate_to_model_executor("compile_prefill")
    compile_decode = _delegate_to_model_executor("compile_decode")
    prefill_forward = _delegate_to_model_executor("prefill_forward")
    decode_forward = _delegate_to_model_executor("decode_forward")
    can_trace_prefill = _delegate_to_model_executor("can_trace_prefill")
    read_decode_output = _delegate_to_model_executor("read_decode_output")
    process_decode_output_host = _delegate_to_model_executor("process_decode_output_host")
    warmup_model_prefill = _delegate_to_model_executor("warmup_model_prefill")
    warmup_model_decode = _delegate_to_model_executor("warmup_model_decode")
    cleanup = _delegate_to_model_executor("cleanup")

    def __getattr__(self, name: str) -> Any:
        executor = getattr(self, "__dict__", {}).get("_model_executor")
        if executor is None:
            raise AttributeError(name)
        return getattr(executor, name)

    def __setattr__(self, name: str, value: Any) -> None:
        executor = getattr(self, "__dict__", {}).get("_model_executor")
        if name == "_model_executor" or executor is None:
            object.__setattr__(self, name, value)
            return
        setattr(executor, name, value)


_FACADE_SURFACE = (
    "model_config",
    "cluster_shape",
    "paged_kv_cache_config",
    "terminal",
    "already_warmed_up_prefill",
    "configure_paged_kv_cache",
    "allocate_kv_cache",
    "compile_prefill",
    "compile_decode",
    "prefill_forward",
    "decode_forward",
    "can_trace_prefill",
    "read_decode_output",
    "process_decode_output_host",
    "warmup_model_prefill",
    "warmup_model_decode",
    "cleanup",
    "__getattr__",
    "__setattr__",
)


def _with_executor_facade(cls):
    for name in _FACADE_SURFACE:
        if name not in cls.__dict__:
            setattr(cls, name, _ExecutorFacadeSurface.__dict__[name])
    return cls


@_with_executor_facade
class Qwen3_32BExecutor:
    """Qwen3-32B policy facade over one family-neutral executor."""

    requires_prefill_trace_warmup = True
    _owner_name = "Qwen3_32BExecutor"
    request_state_fields = ("prompt_tokens", "output_tokens", "slot_remap")

    def __init__(self, model: Any, runtime_config: Any, config: Qwen3_32BExecutorConfig) -> None:
        if not isinstance(config, Qwen3_32BExecutorConfig):
            raise TypeError("config must be a Qwen3_32BExecutorConfig")

        sampling_state_controller, sampling_state = _create_sampling_state(model, config.device_sampling_enabled)
        trace_capture_prime_sequence_lengths = _resolve_trace_capture_prime_sequence_lengths(
            runtime_config,
            num_devices=int(model.config.mesh_device.get_num_devices()),
        )
        self._model_executor = ModelExecutor(
            model,
            runtime_config,
            config,
            owner_name="Qwen3_32BExecutor",
            disable_batched_prefill=bool(runtime_config.disable_batched_prefill) or config.device_sampling_enabled,
            trace_capture_prime_sequence_lengths=trace_capture_prime_sequence_lengths,
            sampling_state_controller=sampling_state_controller,
            sampling_state=sampling_state,
            sampling_type=Sampling1D,
            request_state_fields=self.request_state_fields,
            prefill_warmup=_warmup_q128_around_prefill,
        )
        self._model_executor._q128_topk_tile_ends_warmed = set()


def _create_sampling_state(model: Any, enabled: bool) -> tuple[Any, Any]:
    if not enabled:
        return None, None
    sampling = getattr(model, "sampling", None)
    if not isinstance(sampling, Sampling1D):
        raise TypeError("device sampling requires model.sampling to be a Sampling1D")
    is_resolved = getattr(getattr(sampling, "config", None), "is_resolved", None)
    if not callable(is_resolved) or not is_resolved():
        raise ValueError("model.sampling must have a resolved Sampling1DConfig")
    controller = SamplingState1D(sampling)
    return controller, controller.create_state()


def _warmup_q128_around_prefill(
    executor: ModelExecutor,
    default_warmup: Callable[[], None],
    *,
    kv_cache: Any,
    can_sample_on_device: bool,
    enable_trace: bool,
) -> None:
    """Prime lane-independent Q128 top-k programs before trace activation."""

    if enable_trace:
        _warmup_q128_topk_tile_ends(
            executor,
            kv_cache=kv_cache,
            can_sample_on_device=can_sample_on_device,
            enable_trace=True,
        )
    default_warmup()
    if not enable_trace:
        _warmup_q128_topk_tile_ends(
            executor,
            kv_cache=kv_cache,
            can_sample_on_device=can_sample_on_device,
            enable_trace=False,
        )


def _warmup_q128_topk_tile_ends(
    executor: ModelExecutor,
    *,
    kv_cache: Any,
    can_sample_on_device: bool,
    enable_trace: bool,
) -> None:
    """Prime Q128 top-k slice programs when sampler capacity exceeds lane capacity."""

    if (
        enable_trace in executor._q128_topk_tile_ends_warmed
        or not can_sample_on_device
        or (enable_trace and executor.traced_executor is None)
        or 128 not in executor.warmup.config.prefill_sequence_lengths
        or not executor.prefill_runtime.config.static_q128_topk_supported
        or executor.warmup.config.prime_q128_tile_ends
    ):
        return
    sampling = SamplingParams(
        temperature=torch.ones(1),
        top_k=torch.full((1,), 32, dtype=torch.int32),
        top_p=torch.full((1,), 0.08),
    )
    execution = executor.traced_executor if enable_trace else executor.eager_executor
    for sequence_length in (32, 64, 96):
        page_table_width = (
            sequence_length + executor.page_table_layout.block_size - 1
        ) // executor.page_table_layout.block_size
        executor.compile_prefill(
            tokens=torch.zeros((1, sequence_length), dtype=torch.long),
            page_table=torch.zeros((1, page_table_width), dtype=torch.int32),
            prompt_lens=torch.full((1,), sequence_length, dtype=torch.long),
            empty_slots=[0],
            kv_cache=kv_cache,
            sampling_params=sampling,
            execution=execution,
        )
    executor._q128_topk_tile_ends_warmed.add(enable_trace)


def build_qwen3_32b_executor(llm: Qwen3_32BForCausalLM, config: Qwen3_32BExecutorConfig) -> Qwen3_32BExecutor:
    return Qwen3_32BExecutor(llm.model, llm.runtime_config, config)


def _resolve_trace_capture_prime_sequence_lengths(runtime_config: Any, *, num_devices: int) -> tuple[int, ...]:
    """Prime every advertised T3K prefill trace body independently of batching policy."""

    if num_devices != 8:
        return ()
    advertised = tuple(getattr(runtime_config, "trace_prefill_supported_seq_lens", (128,)))
    return tuple(length for length in advertised if runtime_config.can_enable_trace(length, 0))


def _compat_executor_config(model, *, trace_mode: str, device_sampling_enabled: bool) -> Qwen3_32BExecutorConfig:
    runtime_config = getattr(model, "model_args", None)
    if runtime_config is None:
        raise ValueError("Qwen3_32B compatibility executor requires model.model_args")
    block_size = 32
    max_seq_len = int(getattr(runtime_config, "max_seq_len", model.config.max_seq_len))
    max_batch_size = int(getattr(runtime_config, "max_batch_size", model.config.max_batch_size))
    max_num_blocks = ((max_seq_len + block_size - 1) // block_size) * max_batch_size
    return Qwen3_32BExecutorConfig(
        trace=TraceConfig(mode=trace_mode),
        warmup=WarmupConfig(
            prefill_seq_lens=tuple(getattr(runtime_config, "trace_prefill_supported_seq_lens", (128, 1024))),
            prefill_batch_sizes=(1,),
            include_decode_top_k=device_sampling_enabled,
        ),
        paged_kv_cache=PagedKVCacheConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            dtype=getattr(runtime_config, "kv_cache_dtype", ttnn.bfloat8_b),
        ),
        device_sampling_enabled=device_sampling_enabled,
    )


class EagerQwen3_32BExecutor(Qwen3_32BExecutor):
    """Compatibility wrapper over the model-owned eager runtime."""

    def __init__(self, model, mesh_device):
        del mesh_device
        super().__init__(
            model,
            model.model_args,
            _compat_executor_config(model, trace_mode="none", device_sampling_enabled=False),
        )


class TracedQwen3_32BExecutor(Qwen3_32BExecutor):
    """Compatibility wrapper over the model-owned traced runtime."""

    def __init__(
        self,
        model,
        mesh_device,
        ondevice_decode_loop: bool = False,
        fast_prefill_last_token: bool = False,
        trace_mode: str = "all",
    ):
        del mesh_device, fast_prefill_last_token
        super().__init__(
            model,
            model.model_args,
            _compat_executor_config(
                model,
                trace_mode=trace_mode,
                device_sampling_enabled=bool(ondevice_decode_loop),
            ),
        )
        # Transitional compatibility for demo perf helpers that identify
        # trace-capable executors by the legacy trace bookkeeping attributes.
        self.trace_id_prefill = defaultdict(lambda: None)
        self.trace_ids_decode = defaultdict(lambda: None)


def run_prefill(model, token_ids_tt, *, start_pos: int = 0):
    return model.prefill_from_token_ids(token_ids_tt, start_pos=start_pos)


def run_decode(model, token_id_tt, *, current_pos: int):
    return model.decode_from_token_ids(token_id_tt, current_pos=current_pos)


def run_lm_head(model, hidden_tt):
    if len(hidden_tt.shape) == 4 and hidden_tt.shape[2] > 32:
        old = hidden_tt
        hidden_tt = _slice_last_token_tile(old, hidden_tt.shape[2] - 1)
        ttnn.deallocate(old)
    return model.lm_logits(hidden_tt)
