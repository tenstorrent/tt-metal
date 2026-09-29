# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fully resolved construction policy for one prefill runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from models.common.llm_runtime.config import PageTableLayout
from models.common.llm_runtime.output_reader import OutputReader
from models.common.llm_runtime.prefill.sampling_helpers import _TILE_SIZE

_SUPPORTED_PREFILL_BATCH_SIZES = (1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class PrefillRuntimeConfig:
    """Validated collaborators, geometry, and static prefill capabilities."""

    model: Any
    mesh_device: Any
    output_reader: OutputReader
    page_table_layout: PageTableLayout  # Current geometry; may be replaced once before execution.
    page_table_layout_ceiling: PageTableLayout  # Construction-time upper bound retained across replacement.
    max_batch_size: int
    max_prefill_chunk_size: int
    supports_batched_prefill: bool | None
    disable_batched_prefill: bool
    max_prefill_batch_size: int
    batched_prefill_batched_extract: bool
    cluster_shape: tuple[int, int]
    device_sampling_enabled: bool
    can_enable_trace: Callable[[int, int], bool]
    trace_capture_prime_sequence_lengths: tuple[int, ...]
    allow_force_argmax: bool
    max_device_top_k: int
    sampling_batch_size: int
    static_q128_topk_supported: bool
    sampling_state_controller: Any
    sampling_state: Any

    @classmethod
    def resolve(
        cls,
        *,
        model: Any,
        output_reader: OutputReader,
        page_table_layout: PageTableLayout,
        max_batch_size: int,
        max_prefill_chunk_size: int,
        supports_batched_prefill: bool | None = None,
        disable_batched_prefill: bool = False,
        max_prefill_batch_size: int = 8,
        batched_prefill_batched_extract: bool = True,
        device_sampling_enabled: bool,
        can_enable_trace: Callable[[int, int], bool],
        trace_capture_prime_sequence_lengths: tuple[int, ...] = (),
        sampling_state_controller: Any = None,
        sampling_state: Any = None,
    ) -> "PrefillRuntimeConfig":
        """Validate construction inputs and derive every static capability."""

        _require_positive_int("max_batch_size", max_batch_size)
        _require_positive_int("max_prefill_chunk_size", max_prefill_chunk_size)
        _require_optional_bool("supports_batched_prefill", supports_batched_prefill)
        _require_bool("disable_batched_prefill", disable_batched_prefill)
        _require_positive_int("max_prefill_batch_size", max_prefill_batch_size)
        _require_supported_prefill_batch_size(max_prefill_batch_size)
        _require_bool("batched_prefill_batched_extract", batched_prefill_batched_extract)
        if not isinstance(output_reader, OutputReader):
            raise TypeError("output_reader must be an OutputReader")
        mesh_device = output_reader.mesh_device
        try:
            resolved_cluster_shape = tuple(int(value) for value in mesh_device.shape)
        except (AttributeError, TypeError, ValueError) as error:
            raise TypeError("output_reader.mesh_device must provide a two-dimensional shape") from error
        if len(resolved_cluster_shape) != 2:
            raise ValueError("output_reader.mesh_device shape must contain rows and columns")
        for index, value in enumerate(resolved_cluster_shape):
            _require_positive_int(f"cluster_shape[{index}]", value)
        if not isinstance(device_sampling_enabled, bool):
            raise TypeError("device_sampling_enabled must be bool")
        if not callable(can_enable_trace):
            raise TypeError("can_enable_trace must be callable")
        _validate_trace_capture_prime_sequence_lengths(trace_capture_prime_sequence_lengths)
        if any(length > max_prefill_chunk_size for length in trace_capture_prime_sequence_lengths):
            raise ValueError("trace capture prime sequence lengths cannot exceed max_prefill_chunk_size")

        if not isinstance(page_table_layout, PageTableLayout):
            raise TypeError("page_table_layout must be a PageTableLayout")
        model_mesh = getattr(getattr(model, "config", None), "mesh_device", None)
        if model_mesh is None:
            model_mesh = getattr(model, "mesh_device", None)
        if model_mesh is not mesh_device:
            raise ValueError("model and prefill runtime must use the same mesh device")

        allow_force_argmax = False
        max_device_top_k = 0
        sampling_batch_size = int(max_batch_size)
        static_q128_topk_supported = False
        if device_sampling_enabled:
            sampler = getattr(model, "sampling", None)
            sampler_config = getattr(sampler, "config", None)
            if sampler_config is None:
                raise TypeError("device sampling requires model.sampling.config")
            allow_force_argmax = getattr(sampler_config, "allow_force_argmax", None)
            if not isinstance(allow_force_argmax, bool):
                raise TypeError("model.sampling.config.allow_force_argmax must be bool")
            max_device_top_k = getattr(sampler_config, "max_top_k", None)
            _require_positive_int("model.sampling.config.max_top_k", max_device_top_k)
            sampling_batch_size = getattr(sampler_config, "max_batch_size", None)
            _require_positive_int("model.sampling.config.max_batch_size", sampling_batch_size)
            static_q128_topk_supported = sampling_batch_size >= _TILE_SIZE

        return cls(
            model=model,
            mesh_device=mesh_device,
            output_reader=output_reader,
            page_table_layout=page_table_layout,
            max_batch_size=max_batch_size,
            max_prefill_chunk_size=max_prefill_chunk_size,
            supports_batched_prefill=supports_batched_prefill,
            disable_batched_prefill=disable_batched_prefill,
            max_prefill_batch_size=max_prefill_batch_size,
            batched_prefill_batched_extract=batched_prefill_batched_extract,
            cluster_shape=resolved_cluster_shape,
            device_sampling_enabled=device_sampling_enabled,
            can_enable_trace=can_enable_trace,
            trace_capture_prime_sequence_lengths=trace_capture_prime_sequence_lengths,
            allow_force_argmax=allow_force_argmax,
            max_device_top_k=max_device_top_k,
            sampling_batch_size=sampling_batch_size,
            static_q128_topk_supported=static_q128_topk_supported,
            sampling_state_controller=sampling_state_controller,
            sampling_state=sampling_state,
            page_table_layout_ceiling=page_table_layout,
        )

    def __post_init__(self) -> None:
        _require_positive_int("max_batch_size", self.max_batch_size)
        _require_positive_int("max_prefill_chunk_size", self.max_prefill_chunk_size)
        _require_optional_bool("supports_batched_prefill", self.supports_batched_prefill)
        _require_bool("disable_batched_prefill", self.disable_batched_prefill)
        _require_positive_int("max_prefill_batch_size", self.max_prefill_batch_size)
        _require_supported_prefill_batch_size(self.max_prefill_batch_size)
        _require_bool("batched_prefill_batched_extract", self.batched_prefill_batched_extract)
        _require_positive_int("sampling_batch_size", self.sampling_batch_size)
        if not isinstance(self.output_reader, OutputReader):
            raise TypeError("output_reader must be an OutputReader")
        if not isinstance(self.page_table_layout, PageTableLayout):
            raise TypeError("page_table_layout must be a PageTableLayout")
        if not isinstance(self.cluster_shape, tuple):
            raise TypeError("cluster_shape must be a tuple")
        if len(self.cluster_shape) != 2:
            raise ValueError("cluster_shape must contain rows and columns")
        for index, value in enumerate(self.cluster_shape):
            _require_positive_int(f"cluster_shape[{index}]", value)
        if not isinstance(self.device_sampling_enabled, bool):
            raise TypeError("device_sampling_enabled must be bool")
        if not isinstance(self.allow_force_argmax, bool):
            raise TypeError("allow_force_argmax must be bool")
        if self.device_sampling_enabled:
            _require_positive_int("max_device_top_k", self.max_device_top_k)
        elif self.max_device_top_k != 0:
            raise ValueError("max_device_top_k must be zero when device sampling is disabled")
        if not isinstance(self.static_q128_topk_supported, bool):
            raise TypeError("static_q128_topk_supported must be bool")
        _validate_trace_capture_prime_sequence_lengths(self.trace_capture_prime_sequence_lengths)
        if not isinstance(self.page_table_layout_ceiling, PageTableLayout):
            raise TypeError("page_table_layout_ceiling must be a PageTableLayout")
        if self.page_table_layout.block_size != self.page_table_layout_ceiling.block_size:
            raise ValueError("page_table_layout_ceiling cannot change block_size")
        if self.page_table_layout.raw_capacity_width > self.page_table_layout_ceiling.raw_capacity_width:
            raise ValueError("page_table_layout_ceiling must cover page_table_layout capacity")
        if (
            self.page_table_layout.prefill_width > self.page_table_layout_ceiling.prefill_width
            or self.page_table_layout.decode_width > self.page_table_layout_ceiling.decode_width
        ):
            raise ValueError("page_table_layout_ceiling must cover canonical page-table geometry")
        if not callable(self.can_enable_trace):
            raise TypeError("can_enable_trace must be callable")
        if self.mesh_device is None:
            raise ValueError("mesh_device is required")
        model_mesh = getattr(getattr(self.model, "config", None), "mesh_device", None)
        if model_mesh is None:
            model_mesh = getattr(self.model, "mesh_device", None)
        if model_mesh is not self.mesh_device:
            raise ValueError("model and prefill runtime must use the same mesh device")
        if self.output_reader.mesh_device is not self.mesh_device:
            raise ValueError("output_reader and prefill runtime must use the same mesh device")

        expected_argmax = False
        expected_max_top_k = 0
        expected_sampling_batch_size = self.max_batch_size
        expected_q128 = False
        if self.device_sampling_enabled:
            sampler_config = getattr(getattr(self.model, "sampling", None), "config", None)
            if sampler_config is None:
                raise TypeError("device sampling requires model.sampling.config")
            expected_argmax = getattr(sampler_config, "allow_force_argmax", None)
            if not isinstance(expected_argmax, bool):
                raise TypeError("model.sampling.config.allow_force_argmax must be bool")
            expected_max_top_k = getattr(sampler_config, "max_top_k", None)
            _require_positive_int("model.sampling.config.max_top_k", expected_max_top_k)
            expected_sampling_batch_size = getattr(sampler_config, "max_batch_size", None)
            _require_positive_int("model.sampling.config.max_batch_size", expected_sampling_batch_size)
            expected_q128 = expected_sampling_batch_size >= _TILE_SIZE
        if self.allow_force_argmax is not expected_argmax:
            raise ValueError("allow_force_argmax must match the resolved sampler capability")
        if self.max_device_top_k != expected_max_top_k:
            raise ValueError("max_device_top_k must match the resolved sampler capability")
        if self.sampling_batch_size != expected_sampling_batch_size:
            raise ValueError("sampling_batch_size must match the resolved sampler capacity")
        if self.static_q128_topk_supported is not expected_q128:
            raise ValueError("static_q128_topk_supported must match the resolved sampler capacity")
        if (self.sampling_state_controller is None) != (self.sampling_state is None):
            raise ValueError("sampling_state_controller and sampling_state must be supplied together")
        if self.sampling_state_controller is not None:
            if getattr(self.sampling_state_controller, "sampling", None) is not getattr(self.model, "sampling", None):
                raise ValueError("sampling state controller must borrow model.sampling")
            if not callable(getattr(self.sampling_state_controller, "admit", None)):
                raise TypeError("sampling state controller must provide admit()")

    def with_page_table_layout(self, layout: PageTableLayout) -> "PrefillRuntimeConfig":
        """Return the same resolved policy with a smaller final KV geometry."""

        if not isinstance(layout, PageTableLayout):
            raise TypeError("layout must be a PageTableLayout")
        if layout.block_size != self.page_table_layout.block_size:
            raise ValueError("page-table layout replacement cannot change block_size")
        if layout.raw_capacity_width > self.page_table_layout_ceiling.raw_capacity_width:
            raise ValueError("page-table layout replacement cannot exceed the construction-time capacity ceiling")
        if (
            layout.prefill_width > self.page_table_layout_ceiling.prefill_width
            or layout.decode_width > self.page_table_layout_ceiling.decode_width
        ):
            raise ValueError("page-table layout replacement cannot expand canonical geometry")
        return replace(self, page_table_layout=layout)


def _require_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be bool")


def _require_optional_bool(name: str, value: Any) -> None:
    if value is not None:
        _require_bool(name, value)


def _require_supported_prefill_batch_size(value: int) -> None:
    if value not in _SUPPORTED_PREFILL_BATCH_SIZES:
        raise ValueError(f"max_prefill_batch_size must be one of {_SUPPORTED_PREFILL_BATCH_SIZES}")


def _validate_trace_capture_prime_sequence_lengths(values: Any) -> None:
    if not isinstance(values, tuple):
        raise TypeError("trace_capture_prime_sequence_lengths must be a tuple")
    if len(set(values)) != len(values):
        raise ValueError("trace_capture_prime_sequence_lengths must not contain duplicates")
    for index, value in enumerate(values):
        _require_positive_int(f"trace_capture_prime_sequence_lengths[{index}]", value)
