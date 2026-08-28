# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Llama 3 family policy over the shared model executor lifecycle.

The four public executor types in this module are composition facades.  Each
owns one :class:`~models.common.models.executor.ModelExecutor` and supplies
only the warmup, sampling-state, and prefill policy that differs across the
Llama 3 products.  The facades deliberately do not subclass ``ModelExecutor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Sequence

import torch

from models.common.llm_runtime.config import PagedKVCacheConfig
from models.common.llm_runtime.execution import EagerExecutor, TracedExecutor
from models.common.models.executor import ModelExecutor, ModelExecutorConfig
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.modules.sampling.sampling_state_1d import SamplingState1D
from models.common.sampling import SamplingParams


@dataclass(frozen=True)
class Llama32_1BExecutorConfig(ModelExecutorConfig):
    """Immutable execution policy for Llama 3.2-1B."""


@dataclass(frozen=True)
class Llama32_3BExecutorConfig(ModelExecutorConfig):
    """Immutable execution policy for Llama 3.2-3B."""


@dataclass(frozen=True)
class Llama3ExecutorConfig(ModelExecutorConfig):
    """Immutable execution policy for Llama 3.1-8B."""

    # Qualification-only escape hatch for the BH seeded cross-cardinality
    # experiment. Serving and ordinary demos retain the safe default.
    allow_batched_prefill_with_device_sampling_for_diagnostics: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.allow_batched_prefill_with_device_sampling_for_diagnostics, bool):
            raise TypeError("allow_batched_prefill_with_device_sampling_for_diagnostics must be bool")
        if self.allow_batched_prefill_with_device_sampling_for_diagnostics and not self.device_sampling_enabled:
            raise ValueError(
                "allow_batched_prefill_with_device_sampling_for_diagnostics requires device_sampling_enabled"
            )


@dataclass(frozen=True)
class Llama33_70BExecutorConfig(ModelExecutorConfig):
    """Immutable execution policy for Llama 3.3-70B."""


def _delegate_to_model_executor(method_name: str) -> Callable[..., Any]:
    """Create one signature-preserving facade method."""

    model_executor_method = getattr(ModelExecutor, method_name)

    @wraps(model_executor_method)
    def delegated(self, *args, **kwargs):
        return model_executor_method(_facade_target(self), *args, **kwargs)

    return delegated


def _facade_target(facade: Any) -> Any:
    """Return the composed owner, or a test-fabricated facade itself."""

    return getattr(facade, "__dict__", {}).get("_model_executor", facade)


class _ExecutorFacadeSurface:
    """Descriptors copied onto public facades without creating a hierarchy."""

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
        return target._terminal if target is self else target.terminal

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
    """Install the common composition surface while retaining class identity."""

    for name in _FACADE_SURFACE:
        if name not in cls.__dict__:
            setattr(cls, name, _ExecutorFacadeSurface.__dict__[name])
    return cls


class _Llama32RequestSurface:
    """The pre-history request contract retained by Llama 3.2-1B/3B."""

    def compile_prefill(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        prompt_lens: torch.Tensor | None = None,
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,
        kv_cache: Any = None,
        sampling_params: Any = None,
        execution: EagerExecutor | TracedExecutor | None = None,
    ) -> None:
        return ModelExecutor.compile_prefill(
            _facade_target(self),
            tokens=tokens,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            execution=execution,
        )

    def compile_decode(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: Any = None,
        sampling_params: Any = None,
        reset_batch: bool = False,
        execution: EagerExecutor | TracedExecutor | None = None,
    ) -> None:
        return ModelExecutor.compile_decode(
            _facade_target(self),
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
            execution=execution,
        )

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        *,
        prompt_lens: torch.Tensor | None = None,
        start_pos: torch.Tensor | None = None,
        empty_slots: Sequence[int] | None = None,
        kv_cache: Any = None,
        sampling_params: Any = None,
        execution: EagerExecutor | TracedExecutor | None = None,
    ) -> Any:
        return ModelExecutor.prefill_forward(
            _facade_target(self),
            tokens,
            page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            empty_slots=empty_slots,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            execution=execution,
        )

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        *,
        kv_cache: Any = None,
        sampling_params: Any = None,
        reset_batch: bool = False,
        read_from_device: bool = True,
        execution: EagerExecutor | TracedExecutor | None = None,
    ) -> Any:
        return ModelExecutor.decode_forward(
            _facade_target(self),
            tokens,
            start_pos,
            page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
            read_from_device=read_from_device,
            execution=execution,
        )


_LLAMA32_REQUEST_SURFACE = ("compile_prefill", "compile_decode", "prefill_forward", "decode_forward")


def _with_llama32_request_surface(cls):
    for name in _LLAMA32_REQUEST_SURFACE:
        setattr(cls, name, _Llama32RequestSurface.__dict__[name])
    return cls


def _with_q128_warmup_surface(cls):
    def delegated_q128_warmup(
        self,
        *,
        kv_cache: Any,
        can_sample_on_device: bool,
        enable_trace: bool,
    ) -> None:
        return _warmup_q128_topk_tile_ends(
            _facade_target(self),
            kv_cache=kv_cache,
            can_sample_on_device=can_sample_on_device,
            enable_trace=enable_trace,
        )

    cls._warmup_q128_topk_tile_ends = delegated_q128_warmup
    return cls


@_with_executor_facade
@_with_llama32_request_surface
@_with_q128_warmup_surface
class Llama32_1BExecutor:
    """Llama 3.2-1B facade over one family-neutral executor."""

    requires_prefill_trace_warmup = True
    _owner_name = "Llama32_1BExecutor"

    def __init__(self, model: Any, runtime_config: Any, config: Llama32_1BExecutorConfig) -> None:
        if not isinstance(config, Llama32_1BExecutorConfig):
            raise TypeError("config must be a Llama32_1BExecutorConfig")
        self._model_executor = ModelExecutor(
            model,
            runtime_config,
            config,
            owner_name="Llama32_1BExecutor",
            prefill_warmup=_warmup_q128_before_prefill,
        )
        self._model_executor._q128_topk_tile_ends_warmed = set()


@_with_executor_facade
@_with_llama32_request_surface
@_with_q128_warmup_surface
class Llama32_3BExecutor:
    """Llama 3.2-3B facade over one family-neutral executor."""

    requires_prefill_trace_warmup = True
    _owner_name = "Llama32_3BExecutor"

    def __init__(self, model: Any, runtime_config: Any, config: Llama32_3BExecutorConfig) -> None:
        if not isinstance(config, Llama32_3BExecutorConfig):
            raise TypeError("config must be a Llama32_3BExecutorConfig")
        self._model_executor = ModelExecutor(
            model,
            runtime_config,
            config,
            owner_name="Llama32_3BExecutor",
            prefill_warmup=_warmup_q128_before_prefill,
        )
        self._model_executor._q128_topk_tile_ends_warmed = set()


@_with_executor_facade
class Llama3Executor:
    """Llama 3.1-8B facade over one family-neutral executor."""

    requires_prefill_trace_warmup = True
    _owner_name = "Llama3Executor"
    request_state_fields = ("prompt_tokens", "output_tokens", "slot_remap")

    def __init__(self, model: Any, runtime_config: Any, config: Llama3ExecutorConfig) -> None:
        if not isinstance(config, Llama3ExecutorConfig):
            raise TypeError("config must be a Llama3ExecutorConfig")
        sampling_state_controller, sampling_state = _create_sampling_state(model, config.device_sampling_enabled)
        disable_batched_prefill = (
            False
            if config.allow_batched_prefill_with_device_sampling_for_diagnostics
            else bool(runtime_config.disable_batched_prefill) or config.device_sampling_enabled
        )
        self._model_executor = ModelExecutor(
            model,
            runtime_config,
            config,
            owner_name="Llama3Executor",
            disable_batched_prefill=disable_batched_prefill,
            sampling_state_controller=sampling_state_controller,
            sampling_state=sampling_state,
            sampling_type=Sampling1D,
            request_state_fields=("prompt_tokens", "output_tokens", "slot_remap"),
        )


@_with_executor_facade
@_with_q128_warmup_surface
class Llama33_70BExecutor:
    """Llama 3.3-70B facade over one family-neutral executor."""

    requires_prefill_trace_warmup = True
    _owner_name = "Llama33_70BExecutor"
    request_state_fields = ("prompt_tokens", "output_tokens", "slot_remap")

    def __init__(self, model: Any, runtime_config: Any, config: Llama33_70BExecutorConfig) -> None:
        if not isinstance(config, Llama33_70BExecutorConfig):
            raise TypeError("config must be a Llama33_70BExecutorConfig")
        sampling_state_controller, sampling_state = _create_sampling_state(model, config.device_sampling_enabled)
        prefill_sequence_lengths = getattr(runtime_config, "trace_prefill_warmup_seq_lens", ())
        if not prefill_sequence_lengths:
            prefill_sequence_lengths = getattr(runtime_config, "trace_prefill_supported_seq_lens", (128,))
        self._model_executor = ModelExecutor(
            model,
            runtime_config,
            config,
            owner_name="Llama33_70BExecutor",
            prefill_sequence_lengths=prefill_sequence_lengths,
            disable_batched_prefill=bool(runtime_config.disable_batched_prefill) or config.device_sampling_enabled,
            sampling_state_controller=sampling_state_controller,
            sampling_state=sampling_state,
            sampling_type=Sampling1D,
            request_state_fields=("prompt_tokens", "output_tokens", "slot_remap"),
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


def _warmup_q128_before_prefill(
    executor: ModelExecutor,
    default_warmup: Callable[[], None],
    *,
    kv_cache: Any,
    can_sample_on_device: bool,
    enable_trace: bool,
) -> None:
    _warmup_q128_topk_tile_ends(
        executor,
        kv_cache=kv_cache,
        can_sample_on_device=can_sample_on_device,
        enable_trace=enable_trace,
    )
    return default_warmup()


def _warmup_q128_around_prefill(
    executor: ModelExecutor,
    default_warmup: Callable[[], None],
    *,
    kv_cache: Any,
    can_sample_on_device: bool,
    enable_trace: bool,
) -> None:
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
    """Prime the single-user Q128 top-k slice programs when required."""

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


def build_llama32_1b_executor(llm: Any, config: Llama32_1BExecutorConfig) -> Llama32_1BExecutor:
    return Llama32_1BExecutor(llm.model, llm.runtime_config, config)


def build_llama32_3b_executor(llm: Any, config: Llama32_3BExecutorConfig) -> Llama32_3BExecutor:
    return Llama32_3BExecutor(llm.model, llm.runtime_config, config)


def build_llama3_executor(llm: Any, config: Llama3ExecutorConfig) -> Llama3Executor:
    return Llama3Executor(llm.model, llm.runtime_config, config)


def build_llama33_70b_executor(llm: Any, config: Llama33_70BExecutorConfig) -> Llama33_70BExecutor:
    return Llama33_70BExecutor(llm.model, llm.runtime_config, config)
