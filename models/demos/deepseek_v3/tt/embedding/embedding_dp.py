# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


from pathlib import Path

import torch
from loguru import logger
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.tt.ccl import CCL
from models.demos.deepseek_v3.tt.embedding.embedding1d import Embedding1D
from models.demos.deepseek_v3.utils.config_dataclass import OpConfigBase, ReduceScatterAsyncMinimalConfig
from models.demos.deepseek_v3.utils.config_helpers import shard_and_save
from models.demos.deepseek_v3.utils.run_config import MESH_DEVICE_STATE_DICT_KEY, WeightConfig


class EmbeddingDP(Embedding1D):
    """Embedding module with  tensor and batch parallelism from TTT code.
    Uses DRAM-sharded weights split over rows and replicated over columns"""

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dicts: tuple[dict[str, torch.Tensor] | None, ...],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightConfig:
        # Check that there is only one state dict
        assert (
            len(state_dicts) == 1 and state_dicts[0] is not None
        ), f"Embedding expects exactly one non-padding state dict, got {len(state_dicts)}"
        (state_dict,) = state_dicts

        # Get the embedding weight from the state dict (in the full model: model.embed_tokens.weight)
        torch_weight = state_dict["weight"]

        # Split the last dim in 2 so that it can be sharded across the mesh
        assert (
            torch_weight.shape[-1] == hf_config.hidden_size
        ), "Embedding size does not match the hf_config hidden size"
        assert (
            torch_weight.shape[-1] % mesh_device.get_num_devices() == 0
        ), "Embedding weight last dimension must be divisible by the number of devices"

        # Save to disk with standard naming - "embedding" must match the op name used in the model config
        # so that RunConfig can populate it with the actual weight tensors at runtime
        return {
            "embedding": {
                "weight": shard_and_save(
                    output_path / "embedding.weight",
                    # Convert to TTNN tensor with 1D sharding across final dimension
                    torch_weight,
                    shard_dims=(None, -1),
                    mesh_device=mesh_device,
                    remove_dims=(False, False),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            }
        }

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
    def forward_prefill(cls, x, cfg):
        return cls._forward(x, cfg)

    @classmethod
    def forward_decode(cls, x, cfg):
        return cls._forward(x, cfg)

    @classmethod
    def _forward(cls, x, cfg):
        assert len(x.shape) == 3, "Ids tensor must be 3D: [1, 1, batch]"

        # logger.info(f"embedding_dp forward x shape: {x.shape}")
        # TODO: remove this padding once all gather async supports subtile gathering
        # Add padding so that the batch dimension is divisible by TILE_SIZE
        logger.info(f"embedding_dp forward x shape: {x.shape}")
        _, _, original_seq_len = x.shape
        block_size = ttnn.TILE_SIZE
        if original_seq_len % block_size == 0:
            # better assert that original_seq_len is divisible by block_size ??
            embeddings = ttnn.embedding(x, **cfg["embedding"])
        else:
            x_padded = ttnn.pad(x, [(0, 0), (0, 0), (0, block_size - original_seq_len % block_size)], 0)
            embeddings = ttnn.embedding(x_padded, **cfg["embedding"])
            ttnn.deallocate(x_padded)

        embeddings = ttnn.unsqueeze(embeddings, 0)

        embeddings_tc = ttnn.typecast(embeddings, **cfg["typecast"])
        ttnn.deallocate(embeddings)

        logger.info(f"embedding_dp forward embeddings_tc shape: {embeddings_tc.shape}")
        return embeddings_tc
