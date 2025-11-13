# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.embedding.embedding1d import Embedding1D
from models.demos.deepseek_v3.utils.config_dataclass import OpConfigBase, ReduceScatterAsyncMinimalConfig
from models.demos.deepseek_v3.utils.run_config import MESH_DEVICE_STATE_DICT_KEY


class Embedding2D(Embedding1D):
    """Embedding module with  tensor and batch parallelism from TTT code.
    Uses DRAM-sharded weights split over rows and replicated over columns"""

    @classmethod
    def _embedding_config(
        cls,
        hf_config: PretrainedConfig,
        mesh_device: ttnn.MeshDevice,
        memory_config: ttnn.MemoryConfig,
        output_dtype: ttnn.DataType,
    ) -> dict[str, OpConfigBase]:
        """Config for the Embedding module."""
        cfg = super()._embedding_config(hf_config, mesh_device, memory_config, output_dtype)
        assert "reduce_scatter" not in cfg
        cfg["reduce_scatter"] = ReduceScatterAsyncMinimalConfig(
            cluster_axis=0,
            dim=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

        assert "reduce_scatter_scale" not in cfg
        cfg["reduce_scatter_scale"] = 1.0 / mesh_device.shape[0]

        return cfg

    @classmethod
    def create_state(cls, hf_config: PretrainedConfig, mesh_device: ttnn.Device, ccl: CCL) -> dict[str, ttnn.Tensor]:
        """Create the state for the embedding module."""
        return {
            MESH_DEVICE_STATE_DICT_KEY: mesh_device,
            "ccl": ccl,
        }

    @classmethod
    def _forward(cls, x, cfg):
        scale = cfg["reduce_scatter_scale"]
        x = super()._forward(x, cfg)

        ccl = cfg["ccl"]

        x = ttnn.experimental.reduce_scatter_minimal_async(
            x, **ccl.populate_reduce_scatter_runtime_args(cfg["reduce_scatter"])
        )
        x = x * scale
        return x
