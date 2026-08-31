# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Fully resolved construction policy for one prefill runtime."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

from models.common.llm_runtime.config import PageTableLayout
from models.common.llm_runtime.output_reader import OutputReader
from models.common.llm_runtime.prefill.sampling_helpers import _TILE_SIZE


@dataclass(frozen=True)
class PrefillRuntimeConfig:
    """Validated collaborators, geometry, and static prefill capabilities."""

    model: Any
    mesh_device: Any
    output_reader: OutputReader
    page_table_layout: PageTableLayout
    max_batch_size: int
    max_prefill_chunk_size: int
    cluster_shape: tuple[int, int]
    device_sampling_enabled: bool
    can_enable_trace: Callable[[int, int], bool]
    allow_force_argmax: bool
    sampling_batch_size: int
    static_q128_topk_supported: bool
    max_page_table_capacity_width: int
    max_prefill_page_table_width: int
    max_decode_page_table_width: int

    @classmethod
    def resolve(
        cls,
        *,
        model: Any,
        output_reader: OutputReader,
        page_table_layout: PageTableLayout,
        max_batch_size: int,
        max_prefill_chunk_size: int,
        device_sampling_enabled: bool,
        can_enable_trace: Callable[[int, int], bool],
    ) -> "PrefillRuntimeConfig":
        """Validate construction inputs and derive every static capability."""

        _require_positive_int("max_batch_size", max_batch_size)
        _require_positive_int("max_prefill_chunk_size", max_prefill_chunk_size)
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

        if not isinstance(page_table_layout, PageTableLayout):
            raise TypeError("page_table_layout must be a PageTableLayout")
        model_mesh = getattr(getattr(model, "config", None), "mesh_device", None)
        if model_mesh is None:
            model_mesh = getattr(model, "mesh_device", None)
        if model_mesh is not mesh_device:
            raise ValueError("model and prefill runtime must use the same mesh device")

        allow_force_argmax = False
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
            cluster_shape=resolved_cluster_shape,
            device_sampling_enabled=device_sampling_enabled,
            can_enable_trace=can_enable_trace,
            allow_force_argmax=allow_force_argmax,
            sampling_batch_size=sampling_batch_size,
            static_q128_topk_supported=static_q128_topk_supported,
            max_page_table_capacity_width=page_table_layout.raw_capacity_width,
            max_prefill_page_table_width=page_table_layout.prefill_width,
            max_decode_page_table_width=page_table_layout.decode_width,
        )

    def __post_init__(self) -> None:
        _require_positive_int("max_batch_size", self.max_batch_size)
        _require_positive_int("max_prefill_chunk_size", self.max_prefill_chunk_size)
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
        if not isinstance(self.static_q128_topk_supported, bool):
            raise TypeError("static_q128_topk_supported must be bool")
        width_ceilings = (
            ("max_page_table_capacity_width", self.max_page_table_capacity_width),
            ("max_prefill_page_table_width", self.max_prefill_page_table_width),
            ("max_decode_page_table_width", self.max_decode_page_table_width),
        )
        for name, value in width_ceilings:
            _require_positive_int(name, value)
        if self.page_table_layout.raw_capacity_width > self.max_page_table_capacity_width:
            raise ValueError("max_page_table_capacity_width must cover page_table_layout")
        if self.page_table_layout.prefill_width > self.max_prefill_page_table_width:
            raise ValueError("max_prefill_page_table_width must cover page_table_layout")
        if self.page_table_layout.decode_width > self.max_decode_page_table_width:
            raise ValueError("max_decode_page_table_width must cover page_table_layout")
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
        expected_sampling_batch_size = self.max_batch_size
        expected_q128 = False
        if self.device_sampling_enabled:
            sampler_config = getattr(getattr(self.model, "sampling", None), "config", None)
            if sampler_config is None:
                raise TypeError("device sampling requires model.sampling.config")
            expected_argmax = getattr(sampler_config, "allow_force_argmax", None)
            if not isinstance(expected_argmax, bool):
                raise TypeError("model.sampling.config.allow_force_argmax must be bool")
            expected_sampling_batch_size = getattr(sampler_config, "max_batch_size", None)
            _require_positive_int("model.sampling.config.max_batch_size", expected_sampling_batch_size)
            expected_q128 = expected_sampling_batch_size >= _TILE_SIZE
        if self.allow_force_argmax is not expected_argmax:
            raise ValueError("allow_force_argmax must match the resolved sampler capability")
        if self.sampling_batch_size != expected_sampling_batch_size:
            raise ValueError("sampling_batch_size must match the resolved sampler capacity")
        if self.static_q128_topk_supported is not expected_q128:
            raise ValueError("static_q128_topk_supported must match the resolved sampler capacity")

    def with_page_table_layout(self, layout: PageTableLayout) -> "PrefillRuntimeConfig":
        """Return the same resolved policy with a smaller final KV geometry."""

        if not isinstance(layout, PageTableLayout):
            raise TypeError("layout must be a PageTableLayout")
        if layout.block_size != self.page_table_layout.block_size:
            raise ValueError("page-table layout replacement cannot change block_size")
        if layout.raw_capacity_width > self.max_page_table_capacity_width:
            raise ValueError("page-table layout replacement cannot exceed the construction-time capacity ceiling")
        if (
            layout.prefill_width > self.max_prefill_page_table_width
            or layout.decode_width > self.max_decode_page_table_width
        ):
            raise ValueError("page-table layout replacement cannot expand canonical geometry")
        return replace(self, page_table_layout=layout)


def _require_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
