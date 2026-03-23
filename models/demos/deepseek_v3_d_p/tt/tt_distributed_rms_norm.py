# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Distributed RMSNorm module for DeepSeek V3.

This module performs distributed RMSNorm across chips using:
1. rms_norm_pre_all_gather - compute local sum(x^2) statistics
2. all_gather - gather statistics across chips
3. rms_norm_post_all_gather - normalize using global statistics

Supports both DRAM interleaved and L1 sharded memory configurations.
"""

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule

# DeepSeek 671B RMSNorm dimensions
HIDDEN_DIM = 7168
EPSILON = 1e-6


class TtDistributedRmsNorm(LightweightModule):
    """
    Distributed RMSNorm with hidden dimension sharded across chips.

    Architecture:
        Input: x [batch, seq_len, hidden_dim / num_devices]
        1. Pre-all-gather: Each device computes local sum(x^2) statistics
        2. All-gather: Gather statistics across cluster_axis (mesh columns)
        3. Post-all-gather: Normalize using gathered global statistics

    Weight Sharding:
        - Weight gamma: Shard on dimension 2 across mesh columns
          Shape: [1, 1, hidden_dim // 32, 32] (reshaped for optimal performance)
          mesh_mapper dims=(None, 2)
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        hidden_dim: int = HIDDEN_DIM,
        epsilon: float = EPSILON,
        torch_weight: torch.Tensor = None,
        cluster_axis: int = 1,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        input_memcfg: ttnn.MemoryConfig = None,
        sharded_progcfg: ttnn.LayerNormShardedMultiCoreProgramConfig = None,
        stats_memcfg: ttnn.MemoryConfig = None,
    ):
        """
        Initialize TtDistributedRmsNorm module.

        Args:
            mesh_device: TTNN mesh device
            hidden_dim: Hidden dimension (default: 7168 for DeepSeek 671B)
            epsilon: Small value for numerical stability (default: 1e-6)
            torch_weight: Optional torch tensor of shape [hidden_dim] for gamma weights
            cluster_axis: Mesh dimension to gather along (default: 1 for columns)
            num_links: Number of ethernet links for CCL (default: 1)
            topology: CCL topology - Linear or Ring (default: Linear)
            input_memcfg: Optional memory config for input (e.g., L1 sharded)
            sharded_progcfg: Optional sharded program config for layernorm
            stats_memcfg: Optional memory config for gathered stats (e.g., L1 sharded)
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.num_devices = mesh_device.get_num_devices()
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology

        # Memory configs (None = use DRAM interleaved)
        self.input_memcfg = input_memcfg
        self.sharded_progcfg = sharded_progcfg
        self.stats_memcfg = stats_memcfg

        logger.debug(f"Initializing TtDistributedRmsNorm with hidden_dim={hidden_dim}, epsilon={epsilon}")
        logger.debug(f"Mesh shape: {mesh_device.shape}, num_devices={self.num_devices}")
        logger.debug(f"CCL config: cluster_axis={cluster_axis}, num_links={num_links}, topology={topology}")

        # Create sharded weight
        if torch_weight is not None:
            logger.debug("Creating weight from provided torch tensor")
            self.weight = self._create_sharded_weight_from_torch(torch_weight)
        else:
            logger.debug("Creating random sharded weight")
            self.weight = self._create_random_sharded_weight()

    def _create_sharded_weight_from_torch(self, torch_weight: torch.Tensor) -> ttnn.Tensor:
        """
        Convert torch weight to sharded ttnn tensor.

        Args:
            torch_weight: PyTorch weight tensor of shape [hidden_dim]

        Returns:
            Sharded ttnn tensor with shape [1, 1, hidden_dim // 32, 32]
        """
        assert (
            torch_weight.shape[-1] == self.hidden_dim
        ), f"Weight shape mismatch: expected hidden_dim={self.hidden_dim}, got {torch_weight.shape[-1]}"

        # Reshape weight to [1, 1, hidden_dim // 32, 32] for optimal performance
        torch_weight_reshaped = torch_weight.reshape(1, 1, self.hidden_dim // 32, 32)
        logger.debug(f"Weight reshaped from {torch_weight.shape} to {torch_weight_reshaped.shape}")

        # Create mesh mapper: replicate across mesh rows (dim 0), shard along dim 2 across mesh cols
        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device,
            mesh_shape=self.mesh_device.shape,
            dims=(None, 2),
        )

        tt_weight = ttnn.from_torch(
            torch_weight_reshaped,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
        )

        logger.debug(f"Created sharded weight: {tt_weight.shape}")
        return tt_weight

    def _create_random_sharded_weight(self) -> ttnn.Tensor:
        """Create random sharded weight."""
        torch_weight = torch.rand(self.hidden_dim, dtype=torch.float32) * 2 - 1
        return self._create_sharded_weight_from_torch(torch_weight)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass with distributed RMSNorm.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim / num_devices]
               Expected to be sharded across mesh columns along last dimension.

        Returns:
            Output tensor [batch, seq_len, hidden_dim / num_devices]
            Normalized with same sharding as input.
        """
        logger.debug(f"Forward pass: input shape={x.shape}")

        # Optional: Move input to specified memory config
        if self.input_memcfg is not None:
            x = ttnn.to_memory_config(x, memory_config=self.input_memcfg)
            logger.debug("Moved input to specified memory config")

        # Step 1: Pre-all-gather - each device computes local sum(x^2)
        tt_stats = ttnn.rms_norm_pre_all_gather(
            x,
            dtype=ttnn.bfloat16,
            program_config=self.sharded_progcfg,
        )
        logger.debug(f"Pre-all-gather stats shape: {tt_stats.shape}")

        # Step 2: All-gather stats across cluster_axis
        all_gather_kwargs = {
            "input_tensor": tt_stats,
            "dim": 3,
            "cluster_axis": self.cluster_axis,
            "num_links": self.num_links,
            "topology": self.topology,
        }
        if self.stats_memcfg is not None:
            all_gather_kwargs["memory_config"] = self.stats_memcfg

        tt_gathered_stats = ttnn.all_gather(**all_gather_kwargs)
        ttnn.deallocate(tt_stats)
        logger.debug(f"Gathered stats shape: {tt_gathered_stats.shape}")

        # Step 3: Post-all-gather - normalize using gathered global stats
        tt_output = ttnn.rms_norm_post_all_gather(
            x,
            tt_gathered_stats,
            epsilon=self.epsilon,
            weight=self.weight,
            dtype=ttnn.bfloat16,
            program_config=self.sharded_progcfg,
        )
        ttnn.deallocate(tt_gathered_stats)
        logger.debug(f"Output shape: {tt_output.shape}")

        return tt_output
