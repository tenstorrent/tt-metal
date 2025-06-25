# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import EmbeddingConfig
from models.demos.deepseek_v3.utils.config_helpers import save_and_get_path


class Embedding_1D(AbstractModule):
    """Embedding module with 1D tensor parallelism from TTT code.
    Uses DRAM-sharded weights split 1D across all wormholes"""

    @staticmethod
    def convert_weights(hf_config, state_dict, output_path, mesh_device):
        """DRAM-sharded weights split 1D across all wormholes

        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation names to their TTNN weight file paths
        """
        weight_config = {}

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
        weight_config["embedding"] = {
            "weight": save_and_get_path(output_path / "embedding.weight", ttnn_weight),
        }

        return weight_config

    @staticmethod
    def prefill_model_config(hf_config, mesh_device):
        """Prefill model config for an embedding with 1D tensor parallelism.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for prefill mode
        """
        config = {"mode": "prefill"}

        # Embedding configuration for prefill mode
        config["embedding"] = EmbeddingConfig(
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        return config

    @staticmethod
    def decode_model_config(hf_config, mesh_device):
        """Generate decode operator configuration for this embedding layer.
        Same as prefill mode for this module.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for decode mode
        """
        config = Embedding_1D.prefill_model_config(hf_config, mesh_device)
        config["mode"] = "decode"
        return config

    def __init__(self, hf_config, mesh_device):
        """Initialize the embedding with the given HuggingFace config and mesh device."""
        super().__init__(hf_config, mesh_device)
        # Embedding doesn't need dynamic program configs or temporary tensors

    def forward(self, x, cfg, mesh_device):
        """Forward pass of the embedding.

        Args:
            x: Input tensor (token indices)
            cfg: RunConfig containing weights and op configurations
            mesh_device: TTNN mesh device for multi-device operations

        Returns:
            Output tensor after embedding lookup
        """
        return ttnn.embedding(x, **cfg["embedding"])
