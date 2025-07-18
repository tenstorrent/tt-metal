# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.config_dataclass import FromWeightConfig, MeshDeviceStub, RMSNormConfig
from models.demos.deepseek_v3.utils.config_helpers import (
    COMPUTE_KERNEL_CONFIG_HIFI2,
    NORM_CATEGORIES,
    save_and_get_path,
)
from models.demos.deepseek_v3.utils.run_config import ModelDecodeConfig, ModelPrefillConfig, WeightConfig


class RMSNorm(AbstractModule):
    """Distributed RMSNorm module with 1D tensor parallelism from TTT code.
    Uses DRAM-sharded weights split 1D across 8 wormholes of 8x4 mesh device"""

    @classmethod
    def convert_weights(
        cls,
        hf_config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        output_path: Path,
        mesh_device: ttnn.Device,
        norm_category: str,
    ) -> WeightConfig:
        """DRAM-sharded weights split 1D across all wormholes

        Args:
            hf_config: HuggingFace model configuration object
            state_dict: PyTorch state dict for this layer
            output_path: Path to save converted weights
            mesh_device: TTNN mesh device

        Returns:
            Dict mapping operation names to their TTNN weight file paths
        """
        assert norm_category in NORM_CATEGORIES, f"Invalid norm category: {norm_category}"
        decoder_norm = norm_category == "attention_norm" or norm_category == "mlp_norm"

        # Get the embedding weight from the state dict (in the full model: model.embed_tokens.weight)
        torch_weight = state_dict["weight"]

        # Convert to TTNN tensor with 1D sharded across columns of mesh device
        # Reshape to tile width sticks for optimal performance
        torch_weight = torch_weight.reshape([1, 1, torch_weight.shape[-1] // ttnn.TILE_SIZE, ttnn.TILE_SIZE])
        ttnn_weight = ttnn.as_tensor(
            torch_weight,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=list(mesh_device.shape))
            if decoder_norm
            else ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Save to disk with standard naming - "rmsnorm" must match the op name used in the model config
        # so that RunConfig can populate it with the actual weight tensors at runtime
        return {
            "weight": save_and_get_path(output_path / (norm_category + ".weight"), ttnn_weight),
        }

        return weight_config

    @classmethod
    def decode_model_config(cls, hf_config, mesh_device, norm_category) -> ModelDecodeConfig:
        """Generate decode operator configuration for this rmsnorm layer.

        Args:
            hf_config: HuggingFace model configuration object
            mesh_device: TTNN mesh device

        Returns:
            Dict containing operator configurations for decode mode
        """
        assert norm_category in NORM_CATEGORIES, f"Invalid norm category: {norm_category}"

        output_memcfg = ttnn.DRAM_MEMORY_CONFIG
        stats_memcfg = None
        is_distributed = False
        if norm_category == "attention_norm" or norm_category == "mlp_norm":
            assert (
                list(mesh_device.shape)[1] == 8
            ), f"Only ?x8 mesh devices are supported for Decoder RMSNorm, got {list(mesh_device.shape)}"
            is_distributed = True
            output_memcfg = None
            stats_memcfg = ttnn.create_sharded_memory_config(
                shape=[1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE * list(mesh_device.shape)[1]],
                core_grid=ttnn.CoreGrid(y=1, x=1),
                strategy=ttnn.ShardStrategy.WIDTH,
            )

        # RMSNorm configuration for decode mode
        return RMSNormConfig(
            weight=FromWeightConfig(MeshDeviceStub(mesh_device.shape)),
            epsilon=hf_config.rms_norm_eps,
            compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
            is_distributed=is_distributed,
            output_memcfg=output_memcfg,
            stats_memcfg=stats_memcfg,
            output_dtype=ttnn.bfloat16,
            topology=ttnn.Topology.Linear,
            norm_category=norm_category,
            mesh_device=MeshDeviceStub(mesh_device.shape),
        )

    @classmethod
    def prefill_model_config(cls, hf_config, mesh_device, norm_category) -> ModelPrefillConfig:
        """Prefill model config for an RMSNorm with 1D tensor parallelism."""
        config = RMSNorm.decode_model_config(hf_config, mesh_device, norm_category)
        config.stats_memcfg = ttnn.DRAM_MEMORY_CONFIG
        config.norm_category = norm_category
        return config

    @staticmethod
    def _rmsnorm_forward(x, cfg):
        """Forward pass of the embedding.

        Args:
            x: Input tensor (token indices)
            cfg: RunConfig containing weights and op configurations

        Returns:
            Output tensor after embedding lookup
        """
        program_config = None
        if x.is_sharded():
            grid_size_x = x.memory_config().shard_spec.grid.bounding_box().grid_size().x
            grid_size_y = x.memory_config().shard_spec.grid.bounding_box().grid_size().y
            shard_shape = x.memory_config().shard_spec.shape
            program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=(grid_size_x, grid_size_y),
                subblock_w=1,
                block_h=shard_shape[0] // ttnn.TILE_SIZE,
                block_w=shard_shape[1] // ttnn.TILE_SIZE,
                inplace=False,
            )

        if cfg.is_distributed:
            return RMSNorm._distributed_rmsnorm(
                x,
                epsilon=cfg.epsilon,
                weight=cfg.weight,
                compute_kernel_config=cfg.compute_kernel_config,
                program_config=program_config,
                output_memcfg=cfg.output_memcfg,
                mesh_device=cfg.mesh_device,
                stats_memcfg=cfg.stats_memcfg,
                output_dtype=cfg.output_dtype,
                topology=cfg.topology,
            )
        else:
            return ttnn.rms_norm(
                x,
                epsilon=cfg.epsilon,
                weight=cfg.weight,
                compute_kernel_config=cfg.compute_kernel_config,
                program_config=program_config,
                memory_config=cfg.stats_memcfg,
            )

    @staticmethod
    def _distributed_rmsnorm(
        inp,
        epsilon,
        weight,
        compute_kernel_config,
        program_config,
        output_memcfg,
        mesh_device,
        stats_memcfg,
        output_dtype,
        topology=ttnn.Topology.Linear,
    ):
        # Run distributed rmsnorm part 1
        tt_stats = ttnn.rms_norm_pre_all_gather(inp, program_config=program_config, dtype=output_dtype)
        # AllGather stats
        tt_stats = ttnn.all_gather(
            tt_stats,
            dim=3,
            num_links=1,
            cluster_axis=0,
            mesh_device=mesh_device,
            memory_config=stats_memcfg,
            topology=topology,
        )
        # Run distributed rmsnorm part 2
        tt_out = ttnn.rms_norm_post_all_gather(
            inp,
            epsilon=epsilon,
            weight=weight,
            program_config=program_config,
            stats=tt_stats,
            memory_config=output_memcfg,
            dtype=output_dtype,
        )
        tt_stats.deallocate(True)

        return tt_out

    @classmethod
    def forward_decode(cls, x, cfg):
        return RMSNorm._rmsnorm_forward(x, cfg)

    @classmethod
    def forward_prefill(cls, x, cfg):
        return RMSNorm._rmsnorm_forward(x, cfg)
