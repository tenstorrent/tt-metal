"""
Expert-centric MoE Combine Module (TTNN Implementation)

This module implements the combine operation for Mixture-of-Experts (MoE) layers using TTNN.

This is a TTNN wrapper around the prefill_combine operation, which performs the same
logic as the PyTorch reference implementation but on Tenstorrent hardware.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtCombineModule(LightweightModule):
    """TTNN wrapper for MoE combine operation."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        num_chips: int,
        experts_per_chip: int,
        num_experts_per_tok: int,
        seq_len_per_chip: int,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
    ):
        """
        Initialize combine module wrapper.

        Args:
            mesh_device: TTNN mesh device
            num_chips: Number of chips in the system
            experts_per_chip: Number of experts per chip
            num_experts_per_tok: Number of experts each token is routed to
            seq_len_per_chip: Sequence length per chip
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.num_chips = num_chips
        self.experts_per_chip = experts_per_chip
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip
        self.num_links = num_links
        self.topology = topology

    def forward(
        self,
        dispatched_buffer: ttnn.Tensor,
        dispatched_metadata: ttnn.Tensor,
        experts_tok_counter: ttnn.Tensor,
    ):
        """
        Combine expert outputs back to original token positions using TTNN operation.

        Args:
            dispatched_buffer: Dispatched tokens (num_chips, experts_per_chip, max_tokens, hidden_dim)
            dispatched_metadata: Metadata tensor with token routing information
            experts_tok_counter: Counter tracking tokens per expert (num_chips, experts_per_chip)

        Returns:
            output: Combined output tensor (num_chips, seq_len_per_chip, num_experts_per_tok, hidden_dim)
        """
        output = ttnn.experimental.deepseek.prefill_combine(
            dispatched_buffer,
            dispatched_metadata,
            experts_tok_counter,
            num_chips=self.num_chips,
            experts_per_chip=self.experts_per_chip,
            num_experts_per_tok=self.num_experts_per_tok,
            seq_len_per_chip=self.seq_len_per_chip,
            cluster_axis=0,  # Linear topology along axis 0
            num_links=self.num_links,
            topology=self.topology,
        )
        return output
