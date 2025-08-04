# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod

import ttnn
from models.demos.deepseek_v3.utils.abstract_module import AbstractModule
from models.demos.deepseek_v3.utils.run_config import RunDecodeConfig, RunPrefillConfig


class RMSNormBase(AbstractModule):
    @classmethod
    def is_device_supported(cls, mesh_device: ttnn.Device) -> bool:
        """
        The RMS norm is only run on a 1D device.

        Args:
            mesh_device: The mesh device to check.

        Returns:
            True if the device is supported, False otherwise.
        """
        return tuple(mesh_device.shape)[0] == 1

    @classmethod
    def _get_pc(
        cls, sharded_activation_memory_config: ttnn.ShardSpec
    ) -> ttnn.LayerNormDefaultProgramConfig | ttnn.LayerNormShardedMultiCoreProgramConfig:
        if (
            sharded_activation_memory_config.shard_spec is None
            or sharded_activation_memory_config.buffer_type != ttnn.BufferType.L1
        ):
            return ttnn.LayerNormDefaultProgramConfig()

        # If the activation is sharded, we need to use an optimized rmsnorm
        activation_grid_bounding_box_size = sharded_activation_memory_config.shard_spec.grid.bounding_box().grid_size()
        shard_height, shard_width = sharded_activation_memory_config.shard_spec.shape
        return ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=activation_grid_bounding_box_size,
            subblock_w=1,
            block_h=ttnn.core.divup(shard_height, ttnn.TILE_SIZE),
            block_w=ttnn.core.divup(shard_width, ttnn.TILE_SIZE),
            inplace=False,
        )

    @classmethod
    def forward_decode(cls, x: ttnn.Tensor, cfg: RunDecodeConfig) -> ttnn.Tensor:
        return cls._rmsnorm_forward(x, cfg)

    @classmethod
    def forward_prefill(cls, x: ttnn.Tensor, cfg: RunPrefillConfig) -> ttnn.Tensor:
        return cls._rmsnorm_forward(x, cfg)

    @classmethod
    @abstractmethod
    def _rmsnorm_forward(cls, x: ttnn.Tensor, cfg: RunPrefillConfig | RunDecodeConfig) -> ttnn.Tensor:
        """Forward implementation of RMSNorm layer for both prefill and decode modes.
        Args:
            x: Input tensor (token indices)
            cfg: RunConfig containing weights and op configurations

        Returns:
            Output tensor after RMSNorm computation
        """
        raise NotImplementedError("This method should be implemented in subclasses")
