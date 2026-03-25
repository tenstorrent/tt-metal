# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

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
        dispatch_group_size: int,
        num_dispatch_groups: int,
        experts_per_chip: int,
        num_experts_per_tok: int,
        seq_len_per_chip: int,
        cluster_axis: int = 0,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        memory_config: ttnn.MemoryConfig = None,
        init_zeros: bool = True,
        distributed_zero_init: bool = True,
        inline_zero_init: bool = False,
    ):
        """
        Initialize combine module wrapper.

        Args:
            mesh_device: TTNN mesh device
            dispatch_group_size: Number of chips in each dispatch group
            num_dispatch_groups: Number of parallel dispatch groups
            experts_per_chip: Number of experts per chip
            num_experts_per_tok: Number of experts each token is routed to
            seq_len_per_chip: Sequence length per chip
            memory_config: Output memory configuration (L1 or DRAM interleaved)
            init_zeros: Whether to zero-initialize the output buffer
            distributed_zero_init: Use distributed multi-core DRAM zero init (True) or legacy single-core (False)
        """
        super().__init__()
        self.mesh_device = mesh_device
        self.dispatch_group_size = dispatch_group_size
        self.num_dispatch_groups = num_dispatch_groups
        self.experts_per_chip = experts_per_chip
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip
        self.cluster_axis = cluster_axis
        self.num_links = num_links
        self.topology = topology
        self.memory_config = memory_config
        self.init_zeros = init_zeros
        self.distributed_zero_init = distributed_zero_init
        self.inline_zero_init = inline_zero_init

    def forward(
        self,
        dispatched_buffer: ttnn.Tensor,
        dispatched_metadata: ttnn.Tensor,
        expert_token_counts: ttnn.Tensor,
    ):
        """
        Combine expert outputs back to original token positions using TTNN operation.

        Args:
            dispatched_buffer: Dispatched tokens (dispatch_group_size, experts_per_chip, max_tokens, emb_dim)
            dispatched_metadata: Metadata tensor with token routing information
            expert_token_counts: Counter tracking tokens per expert (dispatch_group_size, experts_per_chip)

        Returns:
            output: Combined output tensor (dispatch_group_size, seq_len_per_chip, num_experts_per_tok, emb_dim)
        """
        output = ttnn.experimental.deepseek_prefill.combine(
            dispatched_buffer,
            dispatched_metadata,
            expert_token_counts,
            dispatch_group_size=self.dispatch_group_size,
            experts_per_chip=self.experts_per_chip,
            num_experts_per_tok=self.num_experts_per_tok,
            seq_len_per_chip=self.seq_len_per_chip,
            cluster_axis=self.cluster_axis,
            num_links=self.num_links,
            topology=self.topology,
            memory_config=self.memory_config,
            init_zeros=self.init_zeros,
            distributed_zero_init=self.distributed_zero_init,
            inline_zero_init=self.inline_zero_init,
        )
        return output
