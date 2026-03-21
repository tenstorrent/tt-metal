# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

"""
Reusable all-to-all operations for MoE infrastructure.

This module provides AllToAllDispatcher and AllToAllCombiner classes that can be
configured for different architectures (DeepSeek-V3, GPT-OSS, etc.).
"""

from typing import Optional, Tuple

from loguru import logger

import ttnn


class AllToAllDispatcher:
    """
    Handles all-to-all dispatch operations for MoE routing.

    This class dispatches tokens to their assigned experts across devices,
    supporting different topologies and configurations.
    """

    @staticmethod
    def dispatch(
        x: ttnn.Tensor,
        indices: ttnn.Tensor,
        expert_mapping_tensors: ttnn.Tensor,
        cluster_axis: int = 0,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        output_shard_dim: Optional[int] = None,
        sparsity_enabled: bool = False,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Dispatch tokens to experts using all-to-all communication.

        Args:
            x: Input tensor to dispatch (batch x seq x hidden_size)
            indices: Expert assignment indices (batch x seq x num_experts_per_tok)
            expert_mapping_tensors: Mapping of experts to devices
            cluster_axis: Axis for collective communication (0 for expert parallel)
            memory_config: Memory configuration (L1 or DRAM)
            topology: Communication topology (Linear for DeepSeek, Ring for GPT-OSS)
            output_shard_dim: Optional shard dimension for output (GPT-OSS specific)
            sparsity_enabled: Enable sparse operations for GPT-OSS

        Returns:
            Tuple of (dispatched_tensor, dispatch_metadata)
        """

        # Determine default memory config if not provided
        if memory_config is None:
            memory_config = ttnn.DRAM_MEMORY_CONFIG

        logger.debug(
            f"AllToAllDispatcher: input_shape={x.shape}, indices_shape={indices.shape}, "
            f"cluster_axis={cluster_axis}, topology={topology}, sparsity={sparsity_enabled}"
        )

        # Perform all-to-all dispatch
        dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(
            x,
            indices,
            expert_mapping_tensors,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
            topology=topology,
        )

        # Apply output sharding if needed (GPT-OSS specific)
        if output_shard_dim is not None and sparsity_enabled:
            # TODO: Apply sparse sharding when implementing ThroughputExpert
            pass

        logger.debug(
            f"AllToAllDispatcher: output_shape={dispatch_output.shape}, " f"metadata_shape={dispatch_metadata.shape}"
        )

        return dispatch_output, dispatch_metadata


class AllToAllCombiner:
    """
    Handles all-to-all combine operations for MoE output aggregation.

    This class combines expert outputs back to original token positions,
    supporting different topologies and post-processing operations.
    """

    @staticmethod
    def combine(
        experts_output: ttnn.Tensor,
        dispatch_metadata: ttnn.Tensor,
        expert_mapping_tensors: ttnn.Tensor,
        cluster_axis: int = 0,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        output_shard_dim: Optional[int] = None,
        apply_all_reduce: bool = False,
        all_reduce_axis: Optional[int] = None,
    ) -> ttnn.Tensor:
        """
        Combine expert outputs using all-to-all communication.

        Args:
            experts_output: Expert computation outputs
            dispatch_metadata: Metadata from dispatch operation
            expert_mapping_tensors: Mapping of experts to devices
            cluster_axis: Axis for collective communication (0 for expert parallel)
            memory_config: Memory configuration (L1 or DRAM)
            topology: Communication topology (Linear for DeepSeek, Ring for GPT-OSS)
            output_shard_dim: Optional shard dimension for output (GPT-OSS specific)
            apply_all_reduce: Whether to apply all-reduce after combine (GPT-OSS)
            all_reduce_axis: Axis for all-reduce operation (typically 1 for TP)

        Returns:
            Combined output tensor
        """

        # Determine default memory config if not provided
        if memory_config is None:
            memory_config = ttnn.DRAM_MEMORY_CONFIG

        logger.debug(
            f"AllToAllCombiner: input_shape={experts_output.shape}, "
            f"metadata_shape={dispatch_metadata.shape}, cluster_axis={cluster_axis}, "
            f"topology={topology}, all_reduce={apply_all_reduce}"
        )

        # Perform all-to-all combine
        combine_output = ttnn.all_to_all_combine(
            experts_output,
            dispatch_metadata,
            expert_mapping_tensors,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
            topology=topology,
        )

        # Apply output sharding if needed (GPT-OSS specific)
        if output_shard_dim is not None:
            # TODO: Apply output dimension sharding for GPT-OSS
            pass

        # Apply all-reduce if needed (GPT-OSS specific)
        if apply_all_reduce and all_reduce_axis is not None:
            logger.debug(f"Applying all-reduce on axis {all_reduce_axis}")
            combine_output = ttnn.all_reduce(
                combine_output,
                cluster_axis=all_reduce_axis,
                memory_config=memory_config,
            )

        logger.debug(f"AllToAllCombiner: output_shape={combine_output.shape}")

        return combine_output


class AllToAllConfig:
    """
    Configuration container for all-to-all operations.

    This class helps manage the configuration parameters for dispatch and combine
    operations, making it easier to support different architectures.
    """

    def __init__(
        self,
        cluster_axis: int = 0,
        memory_config: Optional[ttnn.MemoryConfig] = None,
        dispatch_topology: ttnn.Topology = ttnn.Topology.Linear,
        combine_topology: ttnn.Topology = ttnn.Topology.Linear,
        output_shard_dim: Optional[int] = None,
        sparsity_enabled: bool = False,
        apply_all_reduce: bool = False,
        all_reduce_axis: Optional[int] = None,
    ):
        """
        Initialize all-to-all configuration.

        Args:
            cluster_axis: Axis for collective communication
            memory_config: Memory configuration for operations
            dispatch_topology: Topology for dispatch operation
            combine_topology: Topology for combine operation
            output_shard_dim: Optional output sharding dimension
            sparsity_enabled: Enable sparse operations for GPT-OSS
            apply_all_reduce: Apply all-reduce after combine
            all_reduce_axis: Axis for all-reduce operation
        """
        self.cluster_axis = cluster_axis
        self.memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
        self.dispatch_topology = dispatch_topology
        self.combine_topology = combine_topology
        self.output_shard_dim = output_shard_dim
        self.sparsity_enabled = sparsity_enabled
        self.apply_all_reduce = apply_all_reduce
        self.all_reduce_axis = all_reduce_axis

    @classmethod
    def from_config(cls, config: dict, ep_axis: Optional[int] = None) -> "AllToAllConfig":
        """
        Create configuration from dictionary.

        Args:
            config: Configuration dictionary
            ep_axis: Expert parallel axis override

        Returns:
            AllToAllConfig instance
        """
        collective_config = config.get("collective", {})

        # Determine cluster axis
        cluster_axis = ep_axis if ep_axis is not None else collective_config.get("cluster_axis", 0)

        # Parse topologies
        dispatch_topology = collective_config.get("dispatch_topology", "Linear")
        if isinstance(dispatch_topology, str):
            dispatch_topology = ttnn.Topology.Linear if dispatch_topology == "Linear" else ttnn.Topology.Ring

        combine_topology = collective_config.get("combine_topology", "Linear")
        if isinstance(combine_topology, str):
            combine_topology = ttnn.Topology.Linear if combine_topology == "Linear" else ttnn.Topology.Ring

        # Parse memory config
        memory_str = collective_config.get("memory_config", "DRAM")
        memory_config = ttnn.L1_MEMORY_CONFIG if memory_str == "L1" else ttnn.DRAM_MEMORY_CONFIG

        return cls(
            cluster_axis=cluster_axis,
            memory_config=memory_config,
            dispatch_topology=dispatch_topology,
            combine_topology=combine_topology,
            output_shard_dim=collective_config.get("output_shard_dim"),
            sparsity_enabled=collective_config.get("sparsity_enabled", False),
            apply_all_reduce=collective_config.get("apply_all_reduce", False),
            all_reduce_axis=collective_config.get("all_reduce_axis"),
        )

    def __repr__(self):
        return (
            f"AllToAllConfig(cluster_axis={self.cluster_axis}, "
            f"dispatch_topology={self.dispatch_topology}, "
            f"combine_topology={self.combine_topology}, "
            f"sparsity={self.sparsity_enabled}, "
            f"all_reduce={self.apply_all_reduce})"
        )
