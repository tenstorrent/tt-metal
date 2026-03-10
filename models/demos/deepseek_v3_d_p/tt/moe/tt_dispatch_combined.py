"""
Expert-centric MoE Dispatch Module — Combined Transfer Variant (TTNN Implementation)

This module implements token dispatching for Mixture-of-Experts (MoE) layers using TTNN,
sending metadata and payload in a single fabric/DRAM transfer per token-expert pair.

The output is a combined buffer where each row contains [padded_metadata | payload]
instead of producing separate metadata and payload tensors.
"""

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtDispatchCombinedModule(LightweightModule):
    """Expert-centric MoE dispatch module with combined metadata+payload transfer."""

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
        super().__init__()
        self.mesh_device = mesh_device
        self.num_chips = num_chips
        self.experts_per_chip = experts_per_chip
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.metadata_len = metadata_len
        self.max_dispatched_tokens_per_expert = max_dispatched_tokens_per_expert
        self.seq_len_per_chip = seq_len_per_chip
        self.hidden_dim = hidden_dim
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology

        # L1 alignment is 16 bytes on TT hardware
        l1_alignment = 16
        metadata_bytes = metadata_len * 4  # sizeof(int32)
        padded_metadata_bytes = ((metadata_bytes + l1_alignment - 1) // l1_alignment) * l1_alignment
        self.padded_metadata_bf16 = padded_metadata_bytes // 2
        self.combined_width = self.padded_metadata_bf16 + hidden_dim

        self.combined_shape = (
            num_chips,
            self.experts_per_chip,
            self.max_dispatched_tokens_per_expert,
            self.combined_width,
        )

        # Host-side prep buffers (same as original dispatch)
        self.chip_to_n_routed_expert_counter = torch.zeros((self.num_chips, self.n_routed_experts), dtype=torch.int32)
        self.chip_to_n_routed_expert_offset = torch.zeros((self.num_chips, self.n_routed_experts), dtype=torch.int32)
        self.chip_to_routed_expert_tokens = torch.zeros((self.num_chips, self.experts_per_chip), dtype=torch.int32)

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ):
        """
        Route tokens to expert-specific buffers with combined metadata+payload.

        Args:
            x: Input tensor of shape (num_chips, seq_len, hidden_dim)
            weights: Router weights of shape (num_chips, seq_len, num_experts_per_tok)
            indices: Expert indices of shape (num_chips, seq_len, num_experts_per_tok)

        Returns:
            combined: Combined buffer of shape (num_chips, experts_per_chip, max_dispatched_tokens_per_expert, combined_width)
            experts_counter: Counter of shape (num_chips, experts_per_chip)
            chip_to_n_routed_expert_offset: Offsets (for testing)
            cum_sum: Cumulative sum (for testing)
        """
        # TEMPORARY HOST FALLBACK for offset computation
        self.chip_to_n_routed_expert_counter.zero_()

        mesh_composer = ttnn.create_mesh_composer(
            self.mesh_device,
            ttnn.MeshComposerConfig(
                dims=[0, 1],
            ),
        )
        fallback_indices = ttnn.to_torch(indices, mesh_composer=mesh_composer)

        for chip in range(self.num_chips):
            for token in range(self.seq_len_per_chip):
                for topk_indice in range(self.num_experts_per_tok):
                    routed_expert = fallback_indices[chip, token, topk_indice]
                    self.chip_to_n_routed_expert_counter[chip, routed_expert] += 1

        cum_sum = torch.cumsum(self.chip_to_n_routed_expert_counter, dim=0)
        chip_to_n_routed_expert_offset = torch.vstack(
            [torch.zeros([1, self.n_routed_experts], dtype=torch.int32), cum_sum[:-1]]
        )
        chip_to_routed_expert_tokens = cum_sum[-1].view(self.num_chips, self.experts_per_chip)

        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device,
            mesh_shape=self.mesh_device.shape,
            dims=(0, None),
        )

        chip_to_n_routed_expert_offset_ttnn = ttnn.from_torch(
            chip_to_n_routed_expert_offset,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            dtype=ttnn.int32,
        )

        (
            tt_combined_buffer,
            tt_chip_to_routed_expert_tokens,
        ) = ttnn.experimental.deepseek.prefill_dispatch_combined(
            input_tensor=x,
            weights_tensor=weights,
            indices_tensor=indices,
            chip_to_n_routed_expert_offset_tensor=chip_to_n_routed_expert_offset_ttnn,
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

        logger.info(f"Combined buffer shape: {tt_combined_buffer.shape}")
        logger.info(f"padded_metadata_bf16={self.padded_metadata_bf16}, combined_width={self.combined_width}")

        return (
            tt_combined_buffer,
            chip_to_routed_expert_tokens,
            chip_to_n_routed_expert_offset,
            cum_sum,
        )
