# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen2/Qwen2.5 family policy over the shared model executor lifecycle.

The four public executor types are composition façades. Each owns one
:class:`~models.common.models.executor.ModelExecutor`; none subclasses it.
Only the narrow pre-history request contract and the 7B Q128 warmup policy
remain family-owned here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Sequence

import torch

from models.common.llm_runtime.execution import EagerExecutor, TracedExecutor
from models.common.models.executor import ModelExecutor, ModelExecutorConfig
from models.common.modules.sampling.sampling_1d import Sampling1D
from models.common.sampling import SamplingParams


@dataclass(frozen=True)
class Qwen2ExecutorConfig(ModelExecutorConfig):
    """Immutable execution policy for Qwen2-7B."""


@dataclass(frozen=True)
class Qwen25ExecutorConfig(ModelExecutorConfig):
    """Immutable execution policy for Qwen2.5-7B."""


@dataclass(frozen=True)
class Qwen25_72BExecutorConfig(ModelExecutorConfig):
    """Immutable execution policy for Qwen2.5-72B."""


@dataclass(frozen=True)
class Qwen25Coder32BExecutorConfig(ModelExecutorConfig):
    """Immutable execution policy for Qwen2.5-Coder-32B."""


def _model_executor_target(executor: Any) -> Any:
    """Resolve a real core or a fabricated object used by host contracts."""

    return executor.__dict__.get("_model_executor", executor)


def _delegate_to_model_executor(method_name: str) -> Callable[..., Any]:
    """Create one signature-preserving composition delegate."""

    model_executor_method = getattr(ModelExecutor, method_name)

    @wraps(model_executor_method)
    def delegated(self, *args, **kwargs):
        target = _model_executor_target(self)
        if target is self:
            return model_executor_method(self, *args, **kwargs)
        return getattr(target, method_name)(*args, **kwargs)

    return delegated


class _ExecutorFacadeSurface:
    """Descriptors copied onto public façades without creating a hierarchy."""

    @property
    def model_config(self):
        target = _model_executor_target(self)
        if target is self:
            return ModelExecutor.model_config.__get__(self, type(self))
        return target.model_config

    @property
    def cluster_shape(self) -> list[int]:
        target = _model_executor_target(self)
        if target is self:
            return ModelExecutor.cluster_shape.__get__(self, type(self))
        return target.cluster_shape

    @property
    def paged_kv_cache_config(self):
        target = _model_executor_target(self)
        if target is self:
            return ModelExecutor.paged_kv_cache_config.__get__(self, type(self))
        return target.paged_kv_cache_config

    @property
    def terminal(self) -> bool:
        target = _model_executor_target(self)
        if target is self:
            return ModelExecutor.terminal.__get__(self, type(self))
        return target.terminal

    @property
    def already_warmed_up_prefill(self) -> bool:
        target = _model_executor_target(self)
        if target is self:
            return ModelExecutor.already_warmed_up_prefill.__get__(self, type(self))
        return target.already_warmed_up_prefill

    @already_warmed_up_prefill.setter
    def already_warmed_up_prefill(self, value: bool) -> None:
        target = _model_executor_target(self)
        if target is self:
            return ModelExecutor.already_warmed_up_prefill.__set__(self, value)
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

    def cleanup(self) -> None:
        target = _model_executor_target(self)
        if target is not self:
            return target.cleanup()

        # Older fabricated integration objects attach a state controller to
        # every executor. Qwen2 never owned native sampling state, so keep its
        # historical release order while invoking the common cleanup body.
        controller = self.__dict__.get("sampling_state_controller")
        if controller is not None:
            self.__dict__["sampling_state_controller"] = None
        try:
            return ModelExecutor.cleanup(self)
        finally:
            if controller is not None:
                self.__dict__["sampling_state_controller"] = controller

    def __getattr__(self, name: str) -> Any:
        if name == "_model_executor":
            raise AttributeError(name)
        core = self.__dict__.get("_model_executor")
        if core is None:
            raise AttributeError(name)
        return getattr(core, name)

    def __setattr__(self, name: str, value: Any) -> None:
        core = self.__dict__.get("_model_executor")
        if name == "_model_executor" or core is None:
            object.__setattr__(self, name, value)
            return
        setattr(core, name, value)


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


class _Qwen2RequestSurface:
    """The pre-history request contract retained by every Qwen2 product."""

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
            _model_executor_target(self),
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
            _model_executor_target(self),
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
            _model_executor_target(self),
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
            _model_executor_target(self),
            tokens,
            start_pos,
            page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
            read_from_device=read_from_device,
            execution=execution,
        )


_QWEN2_REQUEST_SURFACE = ("compile_prefill", "compile_decode", "prefill_forward", "decode_forward")


def _with_qwen2_request_surface(cls):
    for name in _QWEN2_REQUEST_SURFACE:
        setattr(cls, name, _Qwen2RequestSurface.__dict__[name])
    return cls


def _warmup_model_prefill_q128(
    self,
    *,
    kv_cache: Any,
    can_sample_on_device: bool,
    enable_trace: bool,
) -> None:
    target = _model_executor_target(self)
    if target is not self:
        return target.warmup_model_prefill(
            kv_cache=kv_cache,
            can_sample_on_device=can_sample_on_device,
            enable_trace=enable_trace,
        )

    target._ensure_active()

    def default_warmup() -> None:
        return target.warmup.warmup_prefill(
            kv_cache=kv_cache,
            can_sample_on_device=can_sample_on_device,
            enable_trace=enable_trace,
        )

    return _warmup_q128_around_prefill(
        target,
        default_warmup,
        kv_cache=kv_cache,
        can_sample_on_device=can_sample_on_device,
        enable_trace=enable_trace,
    )


def _warmup_q128_topk_tile_ends_method(
    self,
    *,
    kv_cache: Any,
    can_sample_on_device: bool,
    enable_trace: bool,
) -> None:
    return _warmup_q128_topk_tile_ends(
        _model_executor_target(self),
        kv_cache=kv_cache,
        can_sample_on_device=can_sample_on_device,
        enable_trace=enable_trace,
    )


@_with_executor_facade
@_with_qwen2_request_surface
class Qwen2Executor:
    """Qwen2-7B façade with explicit Q128 top-k priming."""

    requires_prefill_trace_warmup = True
    _owner_name = "Qwen2Executor"
    warmup_model_prefill = _warmup_model_prefill_q128
    _warmup_q128_topk_tile_ends = _warmup_q128_topk_tile_ends_method

    def __init__(self, model: Any, runtime_config: Any, config: Qwen2ExecutorConfig) -> None:
        if not isinstance(config, Qwen2ExecutorConfig):
            raise TypeError("config must be a Qwen2ExecutorConfig")
        self._model_executor = ModelExecutor(
            model,
            runtime_config,
            config,
            owner_name="Qwen2Executor",
            sampling_type=Sampling1D,
            prefill_warmup=_warmup_q128_around_prefill,
        )
        self._model_executor._q128_topk_tile_ends_warmed = set()


@_with_executor_facade
@_with_qwen2_request_surface
class Qwen25Executor:
    """Qwen2.5-7B façade with explicit Q128 top-k priming."""

    requires_prefill_trace_warmup = True
    _owner_name = "Qwen25Executor"
    warmup_model_prefill = _warmup_model_prefill_q128
    _warmup_q128_topk_tile_ends = _warmup_q128_topk_tile_ends_method

    def __init__(self, model: Any, runtime_config: Any, config: Qwen25ExecutorConfig) -> None:
        if not isinstance(config, Qwen25ExecutorConfig):
            raise TypeError("config must be a Qwen25ExecutorConfig")
        self._model_executor = ModelExecutor(
            model,
            runtime_config,
            config,
            owner_name="Qwen25Executor",
            sampling_type=Sampling1D,
            prefill_warmup=_warmup_q128_around_prefill,
        )
        self._model_executor._q128_topk_tile_ends_warmed = set()


@_with_executor_facade
@_with_qwen2_request_surface
class Qwen25_72BExecutor:
    """Qwen2.5-72B façade with explicit Q128 top-k priming."""

    requires_prefill_trace_warmup = True
    _owner_name = "Qwen25_72BExecutor"
    warmup_model_prefill = _warmup_model_prefill_q128
    _warmup_q128_topk_tile_ends = _warmup_q128_topk_tile_ends_method

    def __init__(self, model: Any, runtime_config: Any, config: Qwen25_72BExecutorConfig) -> None:
        if not isinstance(config, Qwen25_72BExecutorConfig):
            raise TypeError("config must be a Qwen25_72BExecutorConfig")
        self._model_executor = ModelExecutor(
            model,
            runtime_config,
            config,
            owner_name="Qwen25_72BExecutor",
            sampling_type=Sampling1D,
            prefill_warmup=_warmup_q128_around_prefill,
        )
        self._model_executor._q128_topk_tile_ends_warmed = set()

    def allocate_kv_cache(self, kv_cache_shape=None, dtype=None, num_layers=None):
        return ModelExecutor.allocate_kv_cache(
            _model_executor_target(self),
            kv_cache_shape=kv_cache_shape,
            dtype=dtype,
            num_layers=num_layers,
        )


@_with_executor_facade
@_with_qwen2_request_surface
class Qwen25Coder32BExecutor:
    """Qwen2.5-Coder-32B façade with explicit Q128 top-k priming."""

    requires_prefill_trace_warmup = True
    _owner_name = "Qwen25Coder32BExecutor"
    warmup_model_prefill = _warmup_model_prefill_q128
    _warmup_q128_topk_tile_ends = _warmup_q128_topk_tile_ends_method

    def __init__(self, model: Any, runtime_config: Any, config: Qwen25Coder32BExecutorConfig) -> None:
        if not isinstance(config, Qwen25Coder32BExecutorConfig):
            raise TypeError("config must be a Qwen25Coder32BExecutorConfig")
        self._model_executor = ModelExecutor(
            model,
            runtime_config,
            config,
            owner_name="Qwen25Coder32BExecutor",
            sampling_type=Sampling1D,
            prefill_warmup=_warmup_q128_around_prefill,
        )
        self._model_executor._q128_topk_tile_ends_warmed = set()

    def allocate_kv_cache(self, kv_cache_shape=None, dtype=None, num_layers=None):
        return ModelExecutor.allocate_kv_cache(
            _model_executor_target(self),
            kv_cache_shape=kv_cache_shape,
            dtype=dtype,
            num_layers=num_layers,
        )


def _warmup_q128_around_prefill(
    executor: ModelExecutor,
    default_warmup: Callable[[], None],
    *,
    kv_cache: Any,
    can_sample_on_device: bool,
    enable_trace: bool,
) -> None:
    """Preserve traced-before/eager-after Q128 warmup ordering."""

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
    executor: Any,
    *,
    kv_cache: Any,
    can_sample_on_device: bool,
    enable_trace: bool,
) -> None:
    """Prime every Q128 top-k tile end once per eager/traced target."""

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


def build_qwen2_7b_executor(llm: Any, config: Qwen2ExecutorConfig) -> Qwen2Executor:
    return Qwen2Executor(llm.model, llm.runtime_config, config)


def build_qwen25_7b_executor(llm: Any, config: Qwen25ExecutorConfig) -> Qwen25Executor:
    return Qwen25Executor(llm.model, llm.runtime_config, config)


def build_qwen25_72b_executor(llm: Any, config: Qwen25_72BExecutorConfig) -> Qwen25_72BExecutor:
    return Qwen25_72BExecutor(llm.model, llm.runtime_config, config)


def build_qwen25_coder_32b_executor(
    llm: Any,
    config: Qwen25Coder32BExecutorConfig,
) -> Qwen25Coder32BExecutor:
    return Qwen25Coder32BExecutor(llm.model, llm.runtime_config, config)
