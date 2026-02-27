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

        # Oversized buffer to simplify dispatch logic
        self.dispatched_shape = (num_chips, self.experts_per_chip, self.max_dispatched_tokens_per_expert, hidden_dim)
        self.dispatched_metadata_shape = (
            num_chips,
            self.experts_per_chip,
            self.max_dispatched_tokens_per_expert,
            self.metadata_len,
        )

        self.dispatched_buffer = torch.zeros(self.dispatched_shape, dtype=torch.float32)
        self.dispatched_metadata = torch.ones(self.dispatched_metadata_shape, dtype=torch.int32) * -1

        ###
        # prep data for efficient dispatch: count tokens per expert per chip to compute offsets for where to write in the dispatched buffer
        self.chip_to_n_routed_expert_counter = torch.zeros(
            (self.num_chips, self.n_routed_experts), dtype=torch.int32
        )  # amount of tokens dispatched to each expert from each chip
        self.chip_to_n_routed_expert_offset = torch.zeros(
            (self.num_chips, self.n_routed_experts), dtype=torch.int32
        )  # base offset for each expert from each chip in the dispatched buffer
        self.chip_to_routed_expert_tokens = torch.zeros(
            (self.num_chips, self.experts_per_chip), dtype=torch.int32
        )  # total tokens dispatched to each expert per chip

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ):
        """
        Route tokens from their original positions to expert-specific buffers distributed across chips.

        Simulates MoE dispatch: each token is routed to multiple experts based on router indices.
        Tokens are gathered into per-expert buffers with metadata tracking their origin for later recombination.

        Args:
            x: Input tensor of shape (num_chips, seq_len, hidden_dim)
            weights: Router weights of shape (num_chips, seq_len, num_experts_per_tok)
            indices: Expert indices of shape (num_chips, seq_len, num_experts_per_tok)

        Returns:
            dispatched: Dispatched tokens of shape (num_chips, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
            metadata: Metadata tensor of shape (num_chips, experts_per_chip, max_dispatched_tokens_per_expert, metadata_len)
            experts_counter: Counter tracking tokens per expert of shape (num_chips, experts_per_chip)
        """

        # assert (
        #     self.num_chips == x.shape[0] == weights.shape[0] == indices.shape[0]
        # ), f"Mismatched num_chips across inputs. Expected {self.num_chips}, got {x.shape[0]}, {weights.shape[0]}, {indices.shape[0]}"
        # assert (
        #     self.seq_len_per_chip == x.shape[1] == weights.shape[1] == indices.shape[1]
        # ), f"Mismatched seq_len_per_chip across inputs. Expected {self.seq_len_per_chip}, got {x.shape[1]}, {weights.shape[1]}, {indices.shape[1]}"
        # assert (
        #     self.num_experts_per_tok == indices.shape[-1]
        # ), f"Last dimension of indices must match num_experts_per_tok {self.num_experts_per_tok}, got {indices.shape[-1]}"

        # TEMPORARY HOST FALLBACK
        ###
        # prep data for efficient dispatch: count tokens per expert per chip to compute offsets for where to write in the dispatched buffer
        mesh_composer = ttnn.create_mesh_composer(
            self.mesh_device,
            ttnn.MeshComposerConfig(
                dims=[0, 1],  # Axis 0: shard on tensor dim 0; Axis 1: replicated
            ),
        )
        fallback_x = ttnn.to_torch(x, mesh_composer=mesh_composer)
        fallback_weights = ttnn.to_torch(weights, mesh_composer=mesh_composer)
        fallback_indices = ttnn.to_torch(indices, mesh_composer=mesh_composer)

        for chip in range(self.num_chips):
            for token in range(self.seq_len_per_chip):
                for topk_indice in range(self.num_experts_per_tok):
                    routed_expert = fallback_indices[chip, token, topk_indice]
                    self.chip_to_n_routed_expert_counter[chip, routed_expert] += 1

        # this should be local to each chip
        cum_sum = torch.cumsum(self.chip_to_n_routed_expert_counter, dim=0)
        chip_to_n_routed_expert_offset = torch.vstack(
            [torch.zeros([1, self.n_routed_experts], dtype=torch.int32), cum_sum[:-1]]
        )  # base offset for each expert in the dispatched buffer
        # this should be local to each chip
        chip_to_routed_expert_tokens = cum_sum[-1].view(self.num_chips, self.experts_per_chip)

        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device,
            mesh_shape=self.mesh_device.shape,
            dims=(0, None),
        )

        ###

        # Convert chip_to_n_routed_expert_offset to ttnn tensor
        chip_to_n_routed_expert_offset_ttnn = ttnn.from_torch(
            chip_to_n_routed_expert_offset,
            mesh_mapper=mesh_mapper,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            dtype=ttnn.int32,
        )

        tt_dispatched_buffer, tt_dispatch_metadata, tt_chip_to_routed_expert_tokens = (
            ttnn.experimental.deepseek.prefill_dispatch(
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
            )
        )

        torch.set_printoptions(profile="full")
        # logger.info(f"{indices.shape=}")
        # logger.info(f"{indices=}")
        # logger.info(f"{host_tt_dispatched_metadata=}")
        # logger.info(f"{host_tt_dispatched_metadata[..., 0]=}")
        torch.set_printoptions(profile="default")

        tt_dispatched_buffer_shape = tt_dispatched_buffer.shape
        tt_dispatched_metadata_shape = tt_dispatch_metadata.shape
        tt_chip_to_routed_expert_tokens_shape = tt_chip_to_routed_expert_tokens.shape
        logger.info(f"{tt_dispatched_buffer_shape=}")
        logger.info(f"{tt_dispatched_metadata_shape=}")
        logger.info(f"{tt_chip_to_routed_expert_tokens_shape=}")
        # ttnn.visualize_tensor(tt_dispatch_buffer, header="Dispatch Buffer")
        # ttnn.visualize_tensor(tt_dispatch_metadata, header="Dispatch Metadata")
        # ttnn.visualize_tensor(tt_chip_to_routed_expert_tokens, header="Chip to Routed Expert Tokens")

        # chip_to_routed_expert_tokens is needed to run experts
        # metadata and chip_to_routed_expert_tokens are needed for combine step to route expert outputs back to original token positions

        # Return actual kernel outputs (no mockup)
        return (
            tt_dispatched_buffer,
            tt_dispatch_metadata,
            # tt_chip_to_routed_expert_tokens,
            chip_to_routed_expert_tokens,  # needed for combine, actually comes from previous op
            chip_to_n_routed_expert_offset,  # needed for testing
            cum_sum,  # needed for testing
        )
