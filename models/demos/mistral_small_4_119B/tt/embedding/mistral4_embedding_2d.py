# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""2D embedding: same 1D hidden path as ``Mistral4Embedding1D``, then row-axis ``reduce_scatter``.

Mirrors the DeepSeek V3 ``Embedding2D`` pattern: after all-gather (or the single-device replicated
path), ``reduce_scatter_minimal_async`` along ``dim=2`` with ``cluster_axis=0`` (mesh rows), then
multiply by ``1 / mesh_rows`` so replicated row contributions combine correctly.

When the mesh has only one row (``mesh_device.shape[0] == 1``), there is no row-wise collective to
run; we omit ``reduce_scatter`` from the config and ``_forward`` matches the 1D path (e.g. unit
``(1, 1)`` bring-up).
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.mistral_small_4_119B.tt.embedding.mistral4_embedding_1d import Mistral4Embedding1D
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.config_dataclass import ReduceScatterAsyncMinimalConfig
from models.demos.mistral_small_4_119B.tt_utils.run_config import ModelDecodeConfig, ModelPrefillConfig


def _mesh_has_multiple_rows(mesh_device: ttnn.MeshDevice) -> bool:
    return int(mesh_device.shape[0]) > 1


class Mistral4Embedding2D(Mistral4Embedding1D):
    """``embed_tokens`` with 1D hidden sharding + all-gather, then optional row ``reduce_scatter``."""

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice) -> ModelPrefillConfig:
        cfg = super().prefill_model_config(hf_config, mesh_device)
        if not _mesh_has_multiple_rows(mesh_device):
            return cfg
        assert "reduce_scatter" not in cfg
        cfg["reduce_scatter"] = ReduceScatterAsyncMinimalConfig(
            cluster_axis=0,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        assert "reduce_scatter_scale" not in cfg
        cfg["reduce_scatter_scale"] = 1.0 / mesh_device.shape[0]
        return cfg

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice) -> ModelDecodeConfig:
        cfg = super().decode_model_config(hf_config, mesh_device)
        if not _mesh_has_multiple_rows(mesh_device):
            return cfg
        assert "reduce_scatter" not in cfg
        cfg["reduce_scatter"] = ReduceScatterAsyncMinimalConfig(
            cluster_axis=0,
            dim=2,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        assert "reduce_scatter_scale" not in cfg
        cfg["reduce_scatter_scale"] = 1.0 / mesh_device.shape[0]
        return cfg

    @classmethod
    def create_state(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice, ccl: CCL | None = None) -> dict:
        if _mesh_has_multiple_rows(mesh_device) and ccl is None:
            raise ValueError(
                "Mistral4Embedding2D requires a CCL instance when mesh_device.shape[0] > 1 (row reduce-scatter)."
            )
        return Mistral4Embedding1D.create_state(hf_config, mesh_device, ccl)

    @classmethod
    def _forward(cls, x, cfg):
        x = Mistral4Embedding1D._forward(x, cfg)
        if "reduce_scatter" not in cfg:
            return x
        scale = cfg["reduce_scatter_scale"]
        ccl = cfg["ccl"]
        assert ccl is not None, "reduce_scatter requires CCL in run config"
        x = ttnn.experimental.reduce_scatter_minimal_async(
            x, **ccl.populate_reduce_scatter_runtime_args(cfg["reduce_scatter"])
        )
        x = x * scale
        return x
