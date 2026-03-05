"""
Expert-centric MoE Dispatch Module (TTNN Implementation)

This module implements token dispatching for Mixture-of-Experts (MoE) layers using TTNN.

This is a TTNN wrapper around the prefill_dispatch operation, which performs the same
logic as the PyTorch reference implementation but on Tenstorrent hardware.
"""

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtDispatchModule(LightweightModule):
    """Expert-centric MoE dispatch module."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        num_chips: int,
        experts_per_chip: int,
        n_routed_experts: int,
        num_experts_per_tok: int,
        metadata_len: int,
        max_dispatched_tokens_per_expert: int,
        seq_len_per_chip: int,
        hidden_dim: int = 7 * 1024,
        cluster_axis: int = 0,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
    ):
        """
        Initialize dispatch module with configuration parameters.

        Args:
            num_chips: Number of chips in the system
            experts_per_chip: Number of experts per chip
            n_routed_experts: Total number of routed experts across all chips
            metadata_len: Length of metadata per token (stores: chip, token, topk_indice, routed_expert, weight)
            max_dispatched_tokens_per_expert: Maximum number of tokens that can be dispatched to each expert
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.num_chips = num_chips
        self.experts_per_chip = experts_per_chip
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.metadata_len = metadata_len
        self.max_dispatched_tokens_per_expert = max_dispatched_tokens_per_expert
        self.seq_len_per_chip = seq_len_per_chip
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology

    @staticmethod
    def shard_offset_tensor(
        mesh_device: ttnn.MeshDevice,
        chip_to_n_routed_expert_offset: torch.Tensor,
    ) -> ttnn.Tensor:
        """
        Convert and shard the offset tensor for dispatch operation.

        Args:
            mesh_device: The mesh device to place the tensor on
            chip_to_n_routed_expert_offset: Base offset for each expert from each chip
                Shape: (num_chips, n_routed_experts) - from get_gate_outputs()

        Returns:
            TTNN tensor sharded across mesh devices
        """
        mesh_mapper = ttnn.ShardTensor2dMesh(
            mesh_device,
            mesh_shape=mesh_device.shape,
            dims=(0, None),
        )
        return ttnn.from_torch(
            chip_to_n_routed_expert_offset,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.int32,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        weights: ttnn.Tensor,
        indices: ttnn.Tensor,
        tt_chip_to_n_routed_expert_offset: ttnn.Tensor,
    ):
        """
        Route tokens from their original positions to expert-specific buffers distributed across chips.

        Simulates MoE dispatch: each token is routed to multiple experts based on router indices.
        Tokens are gathered into per-expert buffers with metadata tracking their origin for later recombination.

        Args:
            x: Input tensor of shape (num_chips, seq_len, hidden_dim)
            weights: Router weights of shape (num_chips, seq_len, num_experts_per_tok)
            indices: Expert indices of shape (num_chips, seq_len, num_experts_per_tok)
            tt_chip_to_n_routed_expert_offset: Base offset for each expert from each chip (TTNN tensor)
                Shape: (num_chips, n_routed_experts) - use shard_offset_tensor() to create

        Returns:
            dispatched: Dispatched tokens of shape (num_chips, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
            metadata: Metadata tensor of shape (num_chips, experts_per_chip, max_dispatched_tokens_per_expert, metadata_len)
        """
        (
            tt_dispatched_buffer,
            tt_dispatch_metadata,
        ) = ttnn.experimental.deepseek.prefill_dispatch(
            input_tensor=x,
            weights_tensor=weights,
            indices_tensor=indices,
            chip_to_n_routed_expert_offset_tensor=tt_chip_to_n_routed_expert_offset,
            num_chips=self.num_chips,
            experts_per_chip=self.experts_per_chip,
            n_routed_experts=self.n_routed_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            metadata_len=self.metadata_len,
            max_dispatched_tokens_per_expert=self.max_dispatched_tokens_per_expert,
            cluster_axis=self.cluster_axis,
            num_links=self.num_links,
            topology=self.topology,
        )

        torch.set_printoptions(profile="full")
        # logger.info(f"{indices.shape=}")
        # logger.info(f"{indices=}")
        # logger.info(f"{host_tt_dispatched_metadata=}")
        # logger.info(f"{host_tt_dispatched_metadata[..., 0]=}")
        torch.set_printoptions(profile="default")

        tt_dispatched_buffer_shape = tt_dispatched_buffer.shape
        tt_dispatched_metadata_shape = tt_dispatch_metadata.shape
        logger.info(f"{tt_dispatched_buffer_shape=}")
        logger.info(f"{tt_dispatched_metadata_shape=}")
        # ttnn.visualize_tensor(tt_dispatch_buffer, header="Dispatch Buffer")
        # ttnn.visualize_tensor(tt_dispatch_metadata, header="Dispatch Metadata")

        return (tt_dispatched_buffer, tt_dispatch_metadata)
