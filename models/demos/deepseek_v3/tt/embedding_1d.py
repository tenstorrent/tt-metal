# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import EmbeddingConfig, ModelConfig, WeightsConfig, WeightStub


class Embedding_1D(AbstractModule):
    """Embedding module with 1D tensor parallelism from TTT code.
    Uses DRAM-sharded weights split 1D across all wormholes"""

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
    ) -> WeightsConfig:
        """DRAM-sharded weights split 1D across all wormholes

        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation names to their TTNN weight file paths
        """

        # Get the embedding weight from the state dict (in the full model: model.embed_tokens.weight)
        torch_weight = state_dict["weight"]

        # Convert to TTNN tensor with 1D sharding across final dimension
        ttnn_weight = ttnn.as_tensor(
            torch_weight,
            dtype=ttnn.bfloat8_b,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=[-2, -1], mesh_shape=list(mesh_device.shape)),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Save to disk with standard naming - "embedding" must match the op name used in the model config
        # so that RunConfig can populate it with the actual weight tensors at runtime
        return {"weight": WeightStub.from_weight(ttnn_weight, output_path / "embedding.weight")}

    @classmethod
    def prefill_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice, **kwargs) -> ModelConfig:
        """Prefill model config for an embedding with 1D tensor parallelism.
        Same as decode.

        Returns:
            Dict containing operator configurations for prefill mode
        """

        return Embedding_1D._embedding_config(mesh_device)

    @classmethod
    def decode_model_config(cls, hf_config: PretrainedConfig, mesh_device: ttnn.MeshDevice, **kwargs) -> ModelConfig:
        """Generate decode operator configuration for this embedding layer.
        Same as prefill.

        Returns:
            Dict containing operator configurations for decode mode
        """
        return Embedding_1D._embedding_config(mesh_device)

    @staticmethod
    def _embedding_config(mesh_device: ttnn.MeshDevice) -> ModelConfig:
        """Config for the Embedding1D module."""
        return EmbeddingConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            # mesh_device=MeshDeviceStub(tuple(mesh_device.shape)),
        )

    @classmethod
    def forward(cls, x, cfg):
        """Forward pass of the embedding.

        Args:
            x: Input tensor (token indices)
            cfg: RunConfig containing weights and op configurations

        Returns:
            Output tensor after embedding lookup
        """
        return ttnn.embedding(x, **cfg)
