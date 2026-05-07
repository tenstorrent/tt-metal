# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Mistral-4 sequence / batch parallel MLA (2D mesh rows).

:class:`MistralSmall4MLA2D` subclasses :class:`MistralSmall4MLA1D`: prefill all-gathers sequence along mesh rows
before the inner ``mla1d`` prefill, then reduce-scatters with scale ``1 / mesh_rows``. Same pattern as the
DeepSeek-V3 ``MLA2D`` reference in tt-metal, implemented here without importing that module.

:class:`TtMistral4MLA2D` is the eager PyTorch wrapper (same tensors as :class:`~.mla1d.TtMistral4MLA1D`).
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

import ttnn
from models.demos.mistral_small_4_119B.tt.mla.mla1d import (
    MistralSmall4MLA1D,
    MLALoadResult,
    TtMistral4MLA1D,
    _coerce_mistral4_text_config,
    load_ttmistral4_mla_from_sharded_safetensors,
)
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.config_dataclass import (
    AllGatherAsyncConfig,
    KvCacheConfig,
    MeshDeviceStub,
    ReduceScatterAsyncMinimalConfig,
)
from models.demos.mistral_small_4_119B.tt_utils.run_config import (
    MESH_DEVICE_STATE_DICT_KEY,
    ModelDecodeConfig,
    ModelPrefillConfig,
    ModelState,
    RunDecodeConfig,
    RunPrefillConfig,
    WeightConfig,
)
from models.tt_transformers.tt.common import PagedAttentionConfig


class MistralSmall4MLA2D(MistralSmall4MLA1D):
    """Batch and sequence-parallel MLA for Mistral-4 (same pattern as DeepSeek ``MLA2D``)."""

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        (state_dict,) = state_dicts
        assert state_dict is not None, "State dict must be provided for weight conversion."
        return {
            "mla1d": super().convert_weights(hf_config, (state_dict,) * mesh_device.shape[0], output_path, mesh_device)
        }

    @classmethod
    def _convert_weight(
        cls,
        path: Path,
        torch_metaweight: torch.Tensor,
        dims: tuple[int | None, int | None],
        mesh_device: ttnn.MeshDevice,
        memory_config: ttnn.MemoryConfig,
        padding_needed: tuple[int, int, int] = (0, 0, 0),
    ) -> ttnn.Tensor:
        if dims[0] is not None:
            slices = torch.split(torch_metaweight, 1, dim=dims[0])
            torch_metaweight = slices[0]
            dims = (None, dims[1])
        return super()._convert_weight(path, torch_metaweight, dims, mesh_device, memory_config, padding_needed)

    @classmethod
    def prefill_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        batch_size_per_row: int,
    ) -> ModelPrefillConfig:
        super_cfg = super().prefill_model_config(hf_config, mesh_device, batch_size_per_row=batch_size_per_row)
        input_memory_config = super_cfg.pop("input_memory_config")
        return {
            "mla1d": super_cfg,
            "input_memory_config": input_memory_config,
            "seq_ag_prefill": AllGatherAsyncConfig(
                mesh_device=MeshDeviceStub(mesh_device.shape),
                cluster_axis=0,
                dim=2,
            ),
            "seq_rs_prefill": ReduceScatterAsyncMinimalConfig(
                cluster_axis=0,
                dim=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        }

    @classmethod
    def decode_model_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.Device,
        batch_size_per_row: int,
    ) -> ModelDecodeConfig:
        super_cfg = super().decode_model_config(hf_config, mesh_device, batch_size_per_row=batch_size_per_row)
        input_memory_config = super_cfg.pop("input_memory_config")
        return {
            "mla1d": super_cfg,
            "input_memory_config": input_memory_config,
        }

    @classmethod
    def create_state(
        cls,
        hf_config: PretrainedConfig,
        paged_config: PagedAttentionConfig,
        mesh_device: ttnn.MeshDevice,
        ccl: CCL,
        cache: torch.Tensor | None = None,
        kv_cache_override: KvCacheConfig | None = None,
    ) -> ModelState:
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "mla1d": super().create_state(
                hf_config,
                paged_config,
                mesh_device,
                ccl,
                None if cache is None else cache.reshape(mesh_device.shape[0], -1, *cache.shape[1:]),
                kv_cache_override,
            ),
            "ccl": ccl,
        }

    @classmethod
    def forward_decode(
        cls,
        x: ttnn.Tensor,
        position_idxs: ttnn.Tensor,
        cfg: RunDecodeConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        return super().forward_decode(
            x,
            position_idxs=position_idxs,
            row_idx=None,
            cfg=cfg["mla1d"],
            rope_tensors=rope_tensors,
            page_table=page_table,
        )

    @classmethod
    def forward_prefill(
        cls,
        x: ttnn.Tensor,
        batch_idx: int,
        cfg: RunPrefillConfig,
        rope_tensors: dict,
        page_table: ttnn.Tensor,
    ) -> ttnn.Tensor:
        scale = 1 / cfg["mla1d"]["mesh_shape"][0]
        ccl = cfg["ccl"]

        x_next = ttnn.experimental.all_gather_async(x, **ccl.populate_all_gather_runtime_args(cfg["seq_ag_prefill"]))
        batch_size_per_row = cfg["mla1d"]["batch_size_per_row"]
        x_out = super().forward_prefill(
            x_next,
            batch_idx=batch_idx % batch_size_per_row,
            row_idx=batch_idx // batch_size_per_row,
            cfg=cfg["mla1d"],
            rope_tensors=rope_tensors,
            page_table=page_table,
        )
        ttnn.deallocate(x_next)

        x_rs = (
            ttnn.experimental.reduce_scatter_minimal_async(
                x_out, **ccl.populate_reduce_scatter_runtime_args(cfg["seq_rs_prefill"])
            )
            * scale
        )
        return x_rs


class TtMistral4MLA2D(TtMistral4MLA1D):
    """Extends ``TtMistral4MLA1D`` for API parity with DeepSeek ``MLA2D`` / eventual seq-parallel path."""

    parallel_mesh_rows: int = 1

    def __init__(self, config: Mistral4Config, layer_idx: int, *, parallel_mesh_rows: int = 1) -> None:
        super().__init__(config, layer_idx)
        self.parallel_mesh_rows = parallel_mesh_rows

    def load_from_model_dir(
        self,
        model_dir: str | Path,
        *,
        strict: bool = False,
    ) -> MLALoadResult:
        return load_ttmistral4_mla2d_from_sharded_safetensors(self, Path(model_dir), self.layer_idx, strict=strict)


def load_ttmistral4_mla2d_from_sharded_safetensors(
    mla2d: TtMistral4MLA2D,
    model_dir: str | Path,
    layer_idx: int,
    *,
    strict: bool = False,
) -> MLALoadResult:
    """Alias of MLA1D loading for naming parity with DeepSeek ``MLA2D`` helpers."""
    return load_ttmistral4_mla_from_sharded_safetensors(mla2d, model_dir, layer_idx, strict=strict)


def build_and_load_mla2d(
    config,
    model_dir: str | Path,
    layer_idx: int,
    *,
    strict: bool = False,
    parallel_mesh_rows: int = 1,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[TtMistral4MLA2D, MLALoadResult]:
    """Construct ``TtMistral4MLA2D``, load sharded weights, and optionally cast/move."""
    text_config = _coerce_mistral4_text_config(config)
    mla2d = TtMistral4MLA2D(text_config, layer_idx, parallel_mesh_rows=parallel_mesh_rows)
    result = load_ttmistral4_mla2d_from_sharded_safetensors(mla2d, model_dir, layer_idx, strict=strict)
    if dtype is not None or device is not None:
        to_kw: dict = {}
        if dtype is not None:
            to_kw["dtype"] = dtype
        if device is not None:
            to_kw["device"] = device
        mla2d.to(**to_kw)
    return mla2d, result


__all__ = [
    "MistralSmall4MLA2D",
    "TtMistral4MLA2D",
    "build_and_load_mla2d",
    "load_ttmistral4_mla2d_from_sharded_safetensors",
]
