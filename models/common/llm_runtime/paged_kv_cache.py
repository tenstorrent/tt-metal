# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exclusive physical ownership for one paged KV-cache pool."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import torch

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig


class PagedKVCacheState(str, Enum):
    UNRESOLVED = "unresolved"
    CONFIGURED = "configured"
    BOUND = "bound"
    RELEASED = "released"


@dataclass(frozen=True)
class PagedKVCacheContext:
    """Read-only compile context containing borrowed tensor references."""

    config: PagedKVCacheConfig
    tensors: tuple[tuple[Any, Any], ...]
    cache_shapes: tuple[tuple[int, int, int, int], ...]
    per_layer_dtypes: tuple[ttnn.DataType, ...]


@dataclass(frozen=True)
class _LayerKVSpec:
    local_kv_heads: int
    head_dim: int
    dtype: ttnn.DataType


def torch_dtype_for_ttnn(dtype: ttnn.DataType) -> torch.dtype:
    """Return the documented torch storage/request surrogate for a TT dtype."""

    mapping = {
        ttnn.bfloat16: torch.bfloat16,
        ttnn.bfloat8_b: torch.bfloat16,
        ttnn.bfloat4_b: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.int32: torch.int32,
        ttnn.uint32: torch.uint32,
        ttnn.uint16: torch.uint16,
        ttnn.uint8: torch.uint8,
    }
    try:
        return mapping[dtype]
    except KeyError as error:
        raise ValueError(f"No torch compatibility mapping exists for TT dtype {dtype!r}") from error


class PagedKVCacheManager:
    """Own one model-bound paged KV-cache allocation from configure to release.

    The returned cache is a borrowed compatibility handle. The manager remains
    the only owner allowed to replace or deallocate its physical tensors.
    """

    def __init__(self, model: Any, config: PagedKVCacheConfig):
        self._model = model
        self._mesh_device = _model_mesh_device(model)
        self._num_devices = _model_num_devices(model, self._mesh_device)
        self._layer_specs, model_paged_configs = _model_layer_specs(model, self._num_devices)
        self._validate_static_model_contract(config, model_paged_configs)

        self._config = config
        self._state = PagedKVCacheState.CONFIGURED if config.is_resolved() else PagedKVCacheState.UNRESOLVED
        self._configuration_replaced = False
        self._bound_cache: list[list[Any]] | None = None
        self._bound_context: PagedKVCacheContext | None = None
        self._owned_tensors: tuple[Any, ...] = ()
        self._release_in_progress = False
        self._allocated_bytes = 0

    @property
    def config(self) -> PagedKVCacheConfig:
        return self._config

    @property
    def state(self) -> PagedKVCacheState:
        return self._state

    @property
    def bound_cache(self) -> list[list[Any]] | None:
        """Return the exact borrowed compatibility handle, if bound."""

        return self._bound_cache

    @property
    def bound_context(self) -> PagedKVCacheContext | None:
        """Return immutable metadata and borrowed tensor references for compile."""

        return self._bound_context

    @property
    def num_blocks(self) -> int | None:
        return self._config.num_blocks

    @property
    def capacity_tokens(self) -> int | None:
        return self._config.capacity_tokens

    @property
    def max_capacity_tokens(self) -> int:
        return self._config.max_capacity_tokens

    @property
    def per_layer_dtypes(self) -> tuple[ttnn.DataType, ...]:
        return tuple(spec.dtype for spec in self._layer_specs)

    @property
    def cache_shapes(self) -> tuple[tuple[int, int, int, int], ...]:
        if self._config.num_blocks is None:
            return ()
        return tuple(
            (
                self._config.num_blocks,
                spec.local_kv_heads,
                self._config.block_size,
                spec.head_dim,
            )
            for spec in self._layer_specs
        )

    @property
    def cache_shape(self) -> tuple[int, int, int, int] | None:
        shapes = self.cache_shapes
        if shapes and all(shape == shapes[0] for shape in shapes[1:]):
            return shapes[0]
        return None

    @property
    def allocated_bytes(self) -> int:
        return self._allocated_bytes

    def configure(self, config: PagedKVCacheConfig) -> None:
        """Install one immutable resolved replacement before allocation."""

        if self._state in (PagedKVCacheState.BOUND, PagedKVCacheState.RELEASED):
            raise RuntimeError(f"Cannot configure paged KV cache while manager is {self._state.value}")
        if self._config.is_resolved() or self._configuration_replaced:
            raise RuntimeError("Paged KV cache configuration can be resolved only once")
        if not config.is_resolved():
            raise ValueError("Replacement PagedKVCacheConfig must contain num_blocks")

        for field in ("block_size", "max_num_blocks", "dtype", "memory_config"):
            if getattr(config, field) != getattr(self._config, field):
                raise ValueError(f"Resolved PagedKVCacheConfig may only replace num_blocks; {field} changed")

        self._config = config
        self._configuration_replaced = True
        self._state = PagedKVCacheState.CONFIGURED

    def validate_vllm_cache_spec(
        self,
        *,
        block_size: int,
        dtype: torch.dtype,
        num_blocks: int | None = None,
    ) -> None:
        """Validate vLLM's torch-facing cache request against model-owned TT policy."""

        if block_size != self._config.block_size:
            raise ValueError(
                f"vLLM block_size {block_size} does not match configured block_size {self._config.block_size}"
            )
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"vLLM cache dtype must be torch.dtype, got {type(dtype).__name__}")

        mismatches = [
            (layer, spec.dtype, torch_dtype_for_ttnn(spec.dtype))
            for layer, spec in enumerate(self._layer_specs)
            if dtype != torch_dtype_for_ttnn(spec.dtype)
        ]
        if mismatches:
            details = ", ".join(
                f"layer {layer}: {device_dtype!r} expects {torch_dtype!r}"
                for layer, device_dtype, torch_dtype in mismatches
            )
            raise ValueError(f"vLLM cache dtype {dtype!r} is incompatible with model KV dtype ({details})")

        if num_blocks is not None:
            if not isinstance(num_blocks, int) or isinstance(num_blocks, bool) or num_blocks <= 0:
                raise ValueError("vLLM num_blocks must be a positive integer")
            if num_blocks > self._config.max_num_blocks:
                raise ValueError(
                    f"vLLM num_blocks ({num_blocks}) exceeds configured maximum ({self._config.max_num_blocks})"
                )

    def allocate(self) -> list[list[Any]]:
        """Allocate, bind, and return one borrowed physical cache handle."""

        if self._state == PagedKVCacheState.UNRESOLVED:
            raise RuntimeError("Paged KV cache capacity must be resolved before allocation")
        if self._state == PagedKVCacheState.BOUND:
            raise RuntimeError("Paged KV cache has already been allocated")
        if self._state == PagedKVCacheState.RELEASED:
            raise RuntimeError("Paged KV cache manager is released and terminal")
        if self._owned_tensors:
            failures = self._deallocate_owned_tensors()
            if failures:
                raise RuntimeError(
                    f"Failed to finish cleaning {len(failures)} tensor(s) from a previous KV allocation"
                ) from failures[0]

        cache: list[list[Any]] = []
        allocated: list[Any] = []
        host_staging: dict[tuple[tuple[int, int, int, int], torch.dtype], torch.Tensor] = {}
        cache_path = _model_cache_path(self._model)
        dtypes_by_shape: dict[tuple[int, int, int, int], set[ttnn.DataType]] = {}
        for shape, spec in zip(self.cache_shapes, self._layer_specs):
            dtypes_by_shape.setdefault(shape, set()).add(spec.dtype)
        try:
            for shape, spec in zip(self.cache_shapes, self._layer_specs):
                host_key = (shape, torch_dtype_for_ttnn(spec.dtype))
                host_tensor = host_staging.get(host_key)
                if host_tensor is None:
                    host_tensor = torch.zeros(shape, dtype=host_key[1])
                    host_staging[host_key] = host_tensor
                pair = []
                for kv in ("k", "v"):
                    cache_file_name = None
                    if cache_path is not None and len(dtypes_by_shape[shape]) == 1:
                        cache_file_name = cache_path / f"empty_{kv}cache_paged_attention{shape}"
                    tensor = self._allocate_tensor(host_tensor, spec.dtype, cache_file_name)
                    allocated.append(tensor)
                    pair.append(tensor)
                cache.append(pair)
        except BaseException as primary:
            self._set_owned_tensors(allocated)
            cleanup_failures = self._deallocate_owned_tensors(reverse=True)
            _attach_cleanup_failures(primary, cleanup_failures)
            raise

        self._set_owned_tensors(allocated)
        context = PagedKVCacheContext(
            config=self._config,
            tensors=tuple(tuple(pair) for pair in cache),
            cache_shapes=self.cache_shapes,
            per_layer_dtypes=self.per_layer_dtypes,
        )
        try:
            self._model.set_kv_cache(cache)
        except BaseException as primary:
            try:
                self._model.set_kv_cache(None)
            except BaseException as cleanup_error:
                # The model may retain a partial binding. Keep manager ownership
                # so a later release can retry unbinding before deallocation.
                self._bound_cache = cache
                self._bound_context = context
                self._state = PagedKVCacheState.BOUND
                _attach_cleanup_failures(primary, [cleanup_error])
            else:
                cleanup_failures = self._deallocate_owned_tensors(reverse=True)
                _attach_cleanup_failures(primary, cleanup_failures)
            raise

        self._bound_cache = cache
        self._bound_context = context
        self._state = PagedKVCacheState.BOUND
        return cache

    def validate_borrowed_handle(self, cache: Any) -> None:
        if self._state != PagedKVCacheState.BOUND or self._bound_cache is None:
            raise RuntimeError("Paged KV cache is not allocated and bound")
        if cache is not self._bound_cache:
            raise ValueError("Request KV cache is not the exact manager-owned borrowed handle")
        try:
            supplied_tensors = tuple(tensor for pair in cache for tensor in pair)
        except TypeError as error:
            raise ValueError("Request KV cache no longer contains the manager-owned K/V tensor pairs") from error
        if len(supplied_tensors) != len(self._owned_tensors) or any(
            supplied is not owned for supplied, owned in zip(supplied_tensors, self._owned_tensors)
        ):
            raise ValueError("Request KV cache no longer contains the exact manager-owned K/V tensors")

    def release(self) -> None:
        """Unbind then deallocate every owned tensor exactly once."""

        if self._state == PagedKVCacheState.RELEASED:
            return
        if self._state == PagedKVCacheState.BOUND and not self._release_in_progress:
            # Never deallocate while the model still retains the installed handles.
            self._model.set_kv_cache(None)
            self._bound_cache = None
            self._bound_context = None
            self._release_in_progress = True

        failures = self._deallocate_owned_tensors()
        if failures:
            raise RuntimeError(f"Failed to deallocate {len(failures)} paged KV cache tensor(s)") from failures[0]
        self._bound_cache = None
        self._bound_context = None
        self._release_in_progress = False
        self._state = PagedKVCacheState.RELEASED

    def _set_owned_tensors(self, tensors) -> None:
        self._owned_tensors = tuple(tensors)
        self._allocated_bytes = sum(_tensor_nbytes(tensor) for tensor in self._owned_tensors)

    def _deallocate_owned_tensors(self, *, reverse: bool = False) -> list[BaseException]:
        failures = []
        remaining = []
        tensors = reversed(self._owned_tensors) if reverse else self._owned_tensors
        for tensor in tensors:
            try:
                ttnn.deallocate(tensor)
            except BaseException as error:
                failures.append(error)
                remaining.append(tensor)
        if reverse:
            remaining.reverse()
        self._set_owned_tensors(remaining)
        return failures

    def _allocate_tensor(self, host_tensor: torch.Tensor, dtype: ttnn.DataType, cache_file_name: Path | None):
        return ttnn.as_tensor(
            host_tensor,
            device=self._mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self._mesh_device),
            layout=ttnn.TILE_LAYOUT,
            memory_config=self._config.memory_config,
            dtype=dtype,
            cache_file_name=cache_file_name,
        )

    def _validate_static_model_contract(self, config, model_paged_configs) -> None:
        for layer, paged in enumerate(model_paged_configs):
            if paged is None:
                raise ValueError(f"Model layer {layer} is not configured for paged attention")
            if paged.block_size != config.block_size:
                raise ValueError(
                    f"Model layer {layer} block_size {paged.block_size} does not match {config.block_size}"
                )
            if paged.max_num_blocks != config.max_num_blocks:
                raise ValueError(
                    f"Model layer {layer} max_num_blocks {paged.max_num_blocks} does not match "
                    f"{config.max_num_blocks}"
                )

        model_dtypes = self.per_layer_dtypes
        if len(set(model_dtypes)) == 1 and model_dtypes[0] != config.dtype:
            raise ValueError(
                f"Configured KV dtype {config.dtype!r} does not match model-owned dtype {model_dtypes[0]!r}"
            )


def _model_mesh_device(model: Any):
    model_config = getattr(model, "config", None)
    mesh_device = getattr(model_config, "mesh_device", None) or getattr(model, "mesh_device", None)
    if mesh_device is None:
        raise ValueError("Model config must provide mesh_device")
    return mesh_device


def _model_cache_path(model: Any) -> Path | None:
    model_args = getattr(model, "model_args", None)
    cache_path = getattr(model_args, "model_cache_path", None)
    return Path(cache_path) if cache_path else None


def _model_num_devices(model: Any, mesh_device: Any) -> int:
    model_config = getattr(model, "config", None)
    num_devices = getattr(model_config, "num_devices", None) or getattr(model, "num_devices", None)
    if num_devices is None and hasattr(mesh_device, "get_num_devices"):
        num_devices = mesh_device.get_num_devices()
    if not isinstance(num_devices, int) or isinstance(num_devices, bool) or num_devices <= 0:
        raise ValueError("Model config must provide a positive num_devices")
    return num_devices


def _model_layer_specs(model: Any, num_devices: int):
    model_config = getattr(model, "config", None)
    block_configs = getattr(model_config, "block_configs", None)
    if block_configs is not None:
        attention_configs = [getattr(block, "attention_config", None) for block in block_configs]
    else:
        layers = getattr(model, "layers", None)
        if layers is None:
            raise ValueError("Model must expose config.block_configs or layers")
        attention_configs = [getattr(getattr(layer, "attention", None), "config", None) for layer in layers]

    expected_layers = getattr(model_config, "n_layers", None) or getattr(model, "n_layers", None)
    if expected_layers is not None and len(attention_configs) != expected_layers:
        raise ValueError(
            f"Model exposes {len(attention_configs)} attention configs but declares {expected_layers} layers"
        )
    if not attention_configs or any(config is None for config in attention_configs):
        raise ValueError("Every model layer must expose an attention config")

    specs = []
    paged_configs = []
    for layer, attention in enumerate(attention_configs):
        local_kv_heads = getattr(attention, "n_local_kv_heads", None)
        if local_kv_heads is None:
            n_kv_heads = getattr(attention, "n_kv_heads", None)
            if not isinstance(n_kv_heads, int) or n_kv_heads <= 0 or n_kv_heads % num_devices:
                raise ValueError(f"Model layer {layer} n_kv_heads must be positive and divisible by num_devices")
            local_kv_heads = n_kv_heads // num_devices
        head_dim = getattr(attention, "head_dim", None)
        if not isinstance(head_dim, int) or head_dim <= 0:
            raise ValueError(f"Model layer {layer} must provide a positive head_dim")
        dtype = getattr(attention, "kv_cache_dtype", None)
        if dtype is None:
            raise ValueError(f"Model layer {layer} must own kv_cache_dtype")
        specs.append(_LayerKVSpec(local_kv_heads=local_kv_heads, head_dim=head_dim, dtype=dtype))
        paged_configs.append(getattr(attention, "paged_attention_config", None))
    return tuple(specs), tuple(paged_configs)


def _tensor_nbytes(tensor: Any) -> int:
    buffer_page_size = getattr(tensor, "buffer_page_size", None)
    buffer_num_pages = getattr(tensor, "buffer_num_pages", None)
    if callable(buffer_page_size) and callable(buffer_num_pages):
        return int(buffer_page_size()) * int(buffer_num_pages())
    volume = tensor.volume() if callable(getattr(tensor, "volume", None)) else math.prod(tensor.shape)
    element_size = tensor.element_size() if callable(getattr(tensor, "element_size", None)) else 0
    return int(volume) * int(element_size)


def _attach_cleanup_failures(primary: BaseException, failures: list[BaseException]) -> None:
    if not failures:
        return
    try:
        primary.cleanup_failures = tuple(failures)
    except BaseException:
        pass
    add_note = getattr(primary, "add_note", None)
    if add_note is not None:
        for failure in failures:
            add_note(f"cleanup failure: {type(failure).__name__}: {failure}")
