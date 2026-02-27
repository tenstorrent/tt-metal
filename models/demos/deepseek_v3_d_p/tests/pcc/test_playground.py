import pytest
import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule

from tracy import signpost

"""
Design Note: Expert-Centric MoE Dispatch/Combine Prototype

Goals:
- Expert-centric buffer organization: [chips, experts_per_chip, tokens, hidden]
- Dense expert matmuls with no wasted compute (each expert only processes its routed
tokens)
- No wasted memory (compact buffers, no sparse token arrays)
- Capacity factor (CF) handles load imbalance: allocates CF × expected_load per expert
- Full metadata tracking for round-trip verification: dispatch → experts → combine
"""


class TorchDispatchModule(torch.nn.Module):
    """Expert-centric MoE dispatch module."""

    def __init__(
        self,
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

        assert (
            self.num_chips == x.shape[0] == weights.shape[0] == indices.shape[0]
        ), f"Mismatched num_chips across inputs. Expected {self.num_chips}, got {x.shape[0]}, {weights.shape[0]}, {indices.shape[0]}"
        assert (
            self.seq_len_per_chip == x.shape[1] == weights.shape[1] == indices.shape[1]
        ), f"Mismatched seq_len_per_chip across inputs. Expected {self.seq_len_per_chip}, got {x.shape[1]}, {weights.shape[1]}, {indices.shape[1]}"
        assert (
            self.num_experts_per_tok == indices.shape[-1]
        ), f"Last dimension of indices must match num_experts_per_tok {self.num_experts_per_tok}, got {indices.shape[-1]}"

        ###
        # prep data for efficient dispatch: count tokens per expert per chip to compute offsets for where to write in the dispatched buffer

        for chip in range(self.num_chips):
            for token in range(self.seq_len_per_chip):
                for topk_indice in range(self.num_experts_per_tok):
                    routed_expert = indices[chip, token, topk_indice]
                    self.chip_to_n_routed_expert_counter[chip, routed_expert] += 1

        # this should be local to each chip
        cum_sum = torch.cumsum(self.chip_to_n_routed_expert_counter, dim=0)
        chip_to_n_routed_expert_offset = torch.vstack(
            [torch.zeros([1, self.n_routed_experts], dtype=torch.int32), cum_sum[:-1]]
        )  # base offset for each expert in the dispatched buffer
        # this should be local to each chip
        chip_to_routed_expert_tokens = cum_sum[-1].view(self.num_chips, self.experts_per_chip).to(torch.int32)

        ###
        # dispatching tokens and metadata to experts
        for chip in range(self.num_chips):
            for token in range(self.seq_len_per_chip):
                for topk_indice in range(self.num_experts_per_tok):
                    routed_expert = indices[chip, token, topk_indice]
                    # logger.debug(f"Chip {chip} dispatching token {token} to expert [{topk_indice}]={routed_expert}")

                    expert_chip = routed_expert // self.experts_per_chip
                    expert_index_within_chip = routed_expert % self.experts_per_chip
                    dst_index = chip_to_n_routed_expert_offset[chip, routed_expert]

                    self.dispatched_buffer[expert_chip, expert_index_within_chip, dst_index] = x[chip, token]
                    self.dispatched_metadata[expert_chip, expert_index_within_chip, dst_index] = torch.tensor(
                        [chip, token, topk_indice, routed_expert, weights[chip, token, topk_indice]]
                        + [0] * (self.metadata_len - 5),
                        dtype=torch.float32,
                    )
                    chip_to_n_routed_expert_offset[chip, routed_expert] += 1

        # chip_to_routed_expert_tokens is needed to run experts
        # metadata and chip_to_routed_expert_tokens are needed for combine step to route expert outputs back to original token positions
        return self.dispatched_buffer, self.dispatched_metadata, chip_to_routed_expert_tokens


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


class TorchCombineModule(torch.nn.Module):
    """Expert-centric MoE combine module."""

    def __init__(
        self,
        num_chips: int,
        experts_per_chip: int,
        num_experts_per_tok: int,
        seq_len_per_chip: int,
    ):
        """
        Initialize combine module with configuration parameters.

        Args:
            num_chips: Number of chips in the system
            experts_per_chip: Number of experts per chip
            num_experts_per_tok: Number of experts each token is routed to
        """
        super().__init__()
        self.num_chips = num_chips
        self.experts_per_chip = experts_per_chip
        self.num_experts_per_tok = num_experts_per_tok
        self.seq_len_per_chip = seq_len_per_chip

    def forward(
        self,
        dispatched: torch.Tensor,
        metadata: torch.Tensor,
        experts_counter: torch.Tensor,
    ):
        """
        Combine expert outputs back to original token positions.

        Args:
            dispatched: Dispatched tokens of shape (num_chips, experts_per_chip, max_dispatched_tokens_per_expert, hidden_dim)
            metadata: Metadata tensor containing token positions
            experts_counter: Counter tracking tokens per expert
            seq_len: Sequence length per chip (used for output shape)

        Returns:
            y: Combined output tensor of shape (num_chips, seq_len, num_experts_per_tok, hidden_dim)
        """
        # Infer hidden_dim from dispatched tensor shape
        hidden_dim = dispatched.shape[-1]

        y = torch.zeros(
            (self.num_chips, self.seq_len_per_chip, self.num_experts_per_tok, hidden_dim), dtype=torch.bfloat16
        )

        for chips in range(self.num_chips):
            for experts in range(self.experts_per_chip):
                for i in range(experts_counter[chips, experts]):
                    chip = int(metadata[chips, experts, i, 0])
                    token = int(metadata[chips, experts, i, 1])
                    topk_indice = int(metadata[chips, experts, i, 2])
                    y[chip, token, topk_indice] = dispatched[chips, experts, i]

        return y


class TtCombineModule(LightweightModule):
    """TTNN wrapper for MoE combine operation."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        num_chips: int,
        experts_per_chip: int,
        num_experts_per_tok: int,
        seq_len_per_chip: int,
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
            num_links=1,
            topology=ttnn.Topology.Linear,
        )
        return output


def compute_constants(seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor):
    experts_per_chip = n_routed_experts // num_chips
    metadata_len = 5  # chip, token, topk_indice, routed_expert, weight
    balanced_load = num_chips * seq_len_per_chip * num_experts_per_tok // n_routed_experts
    max_dispatched_tokens_per_expert = int(balanced_load * capacity_factor)
    return experts_per_chip, metadata_len, max_dispatched_tokens_per_expert


def initialize_test_inputs(
    num_chips: int,
    seq_len_per_chip: int,
    hidden_dim: int,
    n_routed_experts: int,
    num_experts_per_tok: int,
    max_dispatched_tokens_per_expert: int,
    seed: int = 42,
    validate: bool = True,
):
    """
    Initialize test inputs (x, weights, indices) with random data.

    Args:
        num_chips: Number of chips in the system
        seq_len_per_chip: Sequence length per chip
        hidden_dim: Hidden dimension
        n_routed_experts: Total number of routed experts across all chips
        num_experts_per_tok: Number of experts each token is routed to
        seed: Random seed for reproducibility

    Returns:
        x: Input tensor (num_chips, seq_len_per_chip, hidden_dim)
        weights: Router weights (num_chips, seq_len_per_chip, num_experts_per_tok)
        indices: Expert indices (num_chips, seq_len_per_chip, num_experts_per_tok)
    """
    torch.manual_seed(seed)

    input_shape = (num_chips, seq_len_per_chip, hidden_dim)
    x = torch.randn(input_shape, dtype=torch.bfloat16)

    weights_shape = (num_chips, seq_len_per_chip, num_experts_per_tok)
    indices_shape = (num_chips, seq_len_per_chip, num_experts_per_tok)

    weights = torch.randn(weights_shape, dtype=torch.bfloat16)
    indices = torch.randint(0, n_routed_experts, indices_shape, dtype=torch.int32)

    # Validate expert activations
    if validate:
        expert_activations = torch.zeros((n_routed_experts,), dtype=torch.int32)
        for c in range(indices.shape[0]):
            for t in range(indices.shape[1]):
                for k in range(indices.shape[2]):
                    expert_activations[indices[c, t, k]] += 1
        checksum = expert_activations.sum().item()
        logger.info(f"{expert_activations.shape=}")
        assert (
            checksum == num_chips * seq_len_per_chip * num_experts_per_tok
        ), f"Expected checksum {num_chips * seq_len_per_chip * num_experts_per_tok}, got {checksum}"
        assert (
            expert_activations.max().item() <= max_dispatched_tokens_per_expert
        ), f"Expected max activations per expert to be <= {max_dispatched_tokens_per_expert}, got {expert_activations.max().item()}"

    return x, weights, indices


def initialize_predictable_test_inputs(
    num_chips: int,
    seq_len_per_chip: int,
    hidden_dim: int,
    n_routed_experts: int,
    num_experts_per_tok: int,
    max_dispatched_tokens_per_expert: int,
):
    """
    Initialize test inputs with predictable patterns for debugging.

    Pattern:
    - x: Simple sequential values starting from 0.0
    - weights: Sequential values 1.0, 2.0, 3.0, 4.0 (for num_experts_per_tok=4)
    - indices: Round-robin pattern cycling through experts

    This makes it easy to verify writes:
    - Token 0 -> experts [0, 1, 2, 3]
    - Token 1 -> experts [4, 5, 6, 7]
    - Token 2 -> experts [8, 9, 10, 11]
    - etc.
    """
    input_shape = (num_chips, seq_len_per_chip, hidden_dim)
    # Fill with sequential values: 0.0, 1.0, 2.0, ...
    x = torch.arange(num_chips * seq_len_per_chip * hidden_dim, dtype=torch.float32).reshape(input_shape)
    x = x.to(torch.bfloat16)

    weights_shape = (num_chips, seq_len_per_chip, num_experts_per_tok)
    indices_shape = (num_chips, seq_len_per_chip, num_experts_per_tok)

    # Simple sequential weights: 1.0, 2.0, 3.0, 4.0 for each token
    weights = torch.zeros(weights_shape, dtype=torch.bfloat16)
    for k in range(num_experts_per_tok):
        weights[:, :, k] = float(k + 1)  # 1.0, 2.0, 3.0, 4.0

    # Round-robin indices pattern
    indices = torch.zeros(indices_shape, dtype=torch.int32)
    expert_idx = 0
    for chip in range(num_chips):
        for token in range(seq_len_per_chip):
            for k in range(num_experts_per_tok):
                if chip % 2 == 0:
                    indices[chip, token, k] = max(
                        0, expert_idx % (n_routed_experts) - 1
                    )  # max (0, x -1) to create a of unequal distribution
                else:
                    indices[chip, token, k] = n_routed_experts - 1 - (expert_idx % n_routed_experts)  # reverse order
                expert_idx += 1

    return x, weights, indices


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor",
    [
        (32, 64, 16, 4, 2, 2),
        (512, 32, 256, 8, 4, 2),
        (4096, 32, 256, 8, 32, 2),
    ],
    ids=["xs", "small", "large"],
)
def test_torch_dispatch_combine(
    seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor
):
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor
    )
    print("\n")

    # Initialize inputs using helper function
    x, weights, indices = initialize_test_inputs(
        num_chips=num_chips,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seed=42,
    )

    # Initialize dispatch and combine modules
    dispatch_module = TorchDispatchModule(
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
    )
    combine_module = TorchCombineModule(
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
    )

    # Forward pass through dispatch module
    logger.info(f"{x.shape=}")
    logger.info(f"{weights.shape=}")
    logger.info(f"{indices.shape=}")
    dispatched, metadata, experts_counter = dispatch_module(x, weights, indices)

    torch.set_printoptions(profile="full")
    logger.info(f"{experts_counter.shape=}")
    logger.info(f"{metadata.shape=}")
    logger.info(f"{dispatched.shape=}")
    torch.set_printoptions(profile="default")

    # Forward pass through combine module
    y = combine_module(
        dispatched,
        metadata,
        experts_counter,
    )
    logger.info(f"{y.shape=}")
    y /= num_experts_per_tok  # since we are summing contributions from multiple experts, we need to average them
    y = y.sum(dim=2)  # sum contributions from multiple experts per token
    logger.info(f"{y.shape=}")
    assert torch.allclose(
        x, y, atol=1e-6
    ), f"Expected output to match input, but got max diff {torch.max(torch.abs(x-y)).item()}"


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor",
    [
        # (32, 64, 16, 4, 2, 2),
        # (128, 7168, 16, 4, 2, 2),
        (512, 7168, 16, 4, 2, 2),
        # (1024, 7168, 16, 4, 2, 2),
        # (2048, 7168, 16, 4, 2, 2),
        # (3200, 7168, 16, 4, 2, 2),
        # (4096, 7168, 16, 4, 2, 2),
        # (512, 7 * 1024, 16, 4, 2, 2),
        # (512, 32, 256, 8, 4, 2),
        # (4096, 32, 256, 8, 32, 2),
    ],
    # ids=["xs", "small", "large"],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
        },
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (2, 1),  # SP=2, TP=1
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("use_predictable_data", [True, False], ids=["predictable", "random"])
@pytest.mark.parametrize("verbose", [False])
def test_ttnn_dispatch(
    mesh_device,
    seq_len_per_chip,
    hidden_dim,
    n_routed_experts,
    num_experts_per_tok,
    num_chips,
    capacity_factor,
    use_predictable_data,
    verbose,
):
    signpost(
        f"Dispatch {mesh_device=} {seq_len_per_chip=} {hidden_dim=} {n_routed_experts=} {num_experts_per_tok=} {num_chips=} {capacity_factor=} {use_predictable_data=}"
    )
    print("\n")

    # cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    # cfg.max_packet_payload_size_bytes = 7 * 1024

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor
    )
    logger.info(f"{experts_per_chip=}, {metadata_len=}, {max_dispatched_tokens_per_expert=}")

    num_devices = mesh_device.get_num_devices()
    assert num_chips == num_devices, f"num_chips {num_chips} must match number of devices in mesh {num_devices}"
    mesh_shape = mesh_device.shape
    logger.info(f"Testing with mesh_shape={mesh_shape}, num_devices={num_devices}")
    ttnn.visualize_mesh_device(mesh_device)

    # Initialize inputs using helper function
    if use_predictable_data:
        x, weights, indices = initialize_predictable_test_inputs(
            num_chips=num_chips,
            seq_len_per_chip=seq_len_per_chip,
            hidden_dim=hidden_dim,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        )
        logger.info("Using PREDICTABLE test data for debugging")
    else:
        x, weights, indices = initialize_test_inputs(
            num_chips=num_chips,
            seq_len_per_chip=seq_len_per_chip,
            hidden_dim=hidden_dim,
            n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            seed=42,
        )
        logger.info("Using RANDOM test data")

    mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None),
    )

    tt_x = ttnn.from_torch(
        x, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_weights = ttnn.from_torch(
        weights, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
    )
    tt_indices = ttnn.from_torch(
        indices, mesh_mapper=mesh_mapper, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, dtype=ttnn.int32
    )

    # ttnn.visualize_tensor(tt_x)
    # ttnn.visualize_tensor(tt_weights)
    # ttnn.visualize_tensor(tt_indices)

    # Initialize dispatch and combine modules
    dispatch_module = TorchDispatchModule(
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
    )

    tt_dispatch_module = TtDispatchModule(
        mesh_device=mesh_device,
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
    )
    # combine_module = TorchCombineModule(
    #     num_chips=num_chips,
    #     experts_per_chip=experts_per_chip,
    #     num_experts_per_tok=num_experts_per_tok,
    #     seq_len_per_chip=seq_len_per_chip,
    # )

    # Forward pass through dispatch module
    logger.info(f"{x.shape=}")
    logger.info(f"{weights.shape=}")
    logger.info(f"{indices.shape=}")
    dispatched, metadata, experts_counter = dispatch_module(x, weights, indices)

    # Forward pass through dispatch module
    tt_dispatched, tt_metadata, counter, offsets, cum_sum = tt_dispatch_module(tt_x, tt_weights, tt_indices)

    # Create mesh composer to concatenate sharded tensors
    # Mesh [2, 1] with sharding dims=(0, None): axis 0 sharded on dim 0, axis 1 replicated
    # Pattern from models/common/auto_compose.py: replicated axes use dim 0 and shape 1
    mesh_composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(
            dims=[0, 1],  # Axis 0: shard on tensor dim 0; Axis 1: replicated uses dim 0 (convention)
        ),
    )

    tt_out_dispatched = ttnn.to_torch(tt_dispatched, mesh_composer=mesh_composer, dtype=torch.float32)
    tt_out_metadata = ttnn.to_torch(tt_metadata, mesh_composer=mesh_composer)

    # Kernel counter is garbage - use torch-computed counter instead

    # Quick sanity check of first elements
    logger.info(f"{tt_out_dispatched[0][0][0][0]=} | {tt_out_dispatched[1][0][0][0]=}")
    logger.info(f"{dispatched[0][0][0][0]=} | {dispatched[1][0][0][0]=}")
    logger.info(f"{tt_out_metadata[0][0][0][0:4]=} | {tt_out_metadata[1][0][0][0:4]=}")
    logger.info(f"{metadata[0][0][0][0:4]=} | {metadata[1][0][0][0:4]=}")
    logger.info(f"{counter.shape=}, {counter=}")
    logger.info(f"{offsets.shape=}, {offsets=}")
    logger.info(f"{cum_sum.shape=}, {cum_sum=}")

    data_ok = True
    metadata_ok = True
    logger.warning("Comparing ALL dispatched buffer slots (including remote dispatch)...")
    for dst_chip_id in range(num_chips):
        for expert_id in range(experts_per_chip):
            count = counter[dst_chip_id, expert_id].item()
            out = tt_out_dispatched[dst_chip_id, expert_id, :count, :]
            ref = dispatched[dst_chip_id, expert_id, :count, :]
            if torch.allclose(out, ref, atol=1e-6):
                logger.info(f"✅ Data {dst_chip_id=} {expert_id=} {count=}")
            else:
                logger.error(f"❌ Data {dst_chip_id=} {expert_id=} {count=}")
                data_ok = False
                if verbose:
                    for slot in range(count):
                        torch_data = dispatched[dst_chip_id, expert_id, slot]
                        kernel_data = tt_out_dispatched[dst_chip_id, expert_id, slot]
                        data_match = torch.allclose(torch_data, kernel_data, atol=1e-6)
                        if not data_match:
                            logger.error(
                                f"    Slot {slot}: Data mismatch at chip={dst_chip_id}, expert={expert_id}, slot={slot}: "
                                f"{torch_data=}, {kernel_data=}"
                            )

    logger.info("Comparing ALL dispatched metadata slots (including remote dispatch)...")
    for dst_chip_id in range(num_chips):
        for expert_id in range(experts_per_chip):
            count = counter[dst_chip_id, expert_id].item()
            out = tt_out_metadata[dst_chip_id, expert_id, :count, :4]
            ref = metadata[dst_chip_id, expert_id, :count, :4]
            if torch.allclose(out, ref, atol=1e-6):
                logger.info(f"✅ Metadata {dst_chip_id=} {expert_id=} {count=}")
            else:
                logger.error(f"❌ Metadata {dst_chip_id=} {expert_id=} {count=}")
                metadata_ok = False
                if verbose:
                    for slot in range(count):
                        torch_data = metadata[dst_chip_id, expert_id, slot, :4]
                        kernel_data = tt_out_metadata[dst_chip_id, expert_id, slot, :4]
                        data_match = torch.allclose(torch_data, kernel_data, atol=1e-6)
                        if not data_match:
                            logger.error(
                                f"    Slot {slot}: Metadata mismatch at chip={dst_chip_id}, expert={expert_id}, slot={slot}: "
                                f"{torch_data=}, {kernel_data=}"
                            )
    assert data_ok and metadata_ok, f"Some slots did not match! {data_ok=} {metadata_ok=} Check logs for details."
    logger.info("✅ TTNN dispatch operation matches torch reference!")


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor",
    [
        (512, 7 * 1024, 16, 4, 2, 2),
        # Add more test cases as needed
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=7 * 1024),
        },
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (2, 1),  # SP=2, TP=1
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("use_predictable_data", [True, False], ids=["predictable", "random"])
def test_ttnn_combine(
    mesh_device,
    seq_len_per_chip,
    hidden_dim,
    n_routed_experts,
    num_experts_per_tok,
    num_chips,
    capacity_factor,
    use_predictable_data,
):
    """Test TTNN combine operation in isolation using torch reference inputs."""

    signpost(
        f"Combine {mesh_device=} {seq_len_per_chip=} {hidden_dim=} "
        f"{n_routed_experts=} {num_experts_per_tok=} {num_chips=} "
        f"{capacity_factor=} {use_predictable_data=}"
    )
    print("\n")

    # Compute configuration
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor
    )
    logger.info(f"{experts_per_chip=}, {metadata_len=}, {max_dispatched_tokens_per_expert=}")

    # Verify mesh configuration
    num_devices = mesh_device.get_num_devices()
    assert num_chips == num_devices
    mesh_shape = mesh_device.shape
    logger.info(f"Testing with mesh_shape={mesh_shape}, num_devices={num_devices}")

    # Step 1: Generate initial inputs using torch
    if use_predictable_data:
        x, weights, indices = initialize_predictable_test_inputs(
            num_chips,
            seq_len_per_chip,
            hidden_dim,
            n_routed_experts,
            num_experts_per_tok,
            max_dispatched_tokens_per_expert,
        )
        logger.info("Using PREDICTABLE test data for debugging")
    else:
        x, weights, indices = initialize_test_inputs(
            num_chips,
            seq_len_per_chip,
            hidden_dim,
            n_routed_experts,
            num_experts_per_tok,
            max_dispatched_tokens_per_expert,
            seed=42,
        )
        logger.info("Using RANDOM test data")

    # Step 2: Run torch dispatch to generate combine inputs
    torch_dispatch = TorchDispatchModule(
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
    )

    dispatched_buffer, dispatched_metadata, experts_tok_counter = torch_dispatch(x, weights, indices)

    logger.info("Torch dispatch outputs:")
    logger.info(f"  {dispatched_buffer.shape=}")
    logger.info(f"  {dispatched_metadata.shape=}")
    logger.info(f"  {experts_tok_counter.shape=}")

    # Step 3: Convert torch tensors to ttnn tensors
    mesh_mapper = ttnn.ShardTensor2dMesh(
        mesh_device,
        mesh_shape=mesh_device.shape,
        dims=(0, None),  # Shard on dim 0, replicate on dim 1
    )

    tt_dispatched_buffer = ttnn.from_torch(
        dispatched_buffer,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )

    tt_dispatched_metadata = ttnn.from_torch(
        dispatched_metadata,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    tt_experts_tok_counter = ttnn.from_torch(
        experts_tok_counter,
        mesh_mapper=mesh_mapper,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        dtype=ttnn.int32,
    )

    # Step 4: Run torch combine for reference output
    torch_combine = TorchCombineModule(
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
    )

    torch_output = torch_combine(
        dispatched_buffer,
        dispatched_metadata,
        experts_tok_counter,
    )

    logger.info(f"Torch combine output shape: {torch_output.shape}")

    # Step 5: Run ttnn combine
    tt_combine = TtCombineModule(
        mesh_device=mesh_device,
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        num_experts_per_tok=num_experts_per_tok,
        seq_len_per_chip=seq_len_per_chip,
    )

    tt_output = tt_combine(
        tt_dispatched_buffer,
        tt_dispatched_metadata,
        tt_experts_tok_counter,
    )

    logger.info(f"TTNN combine output shape: {tt_output.shape}")

    # Step 6: Convert ttnn output to torch for comparison
    mesh_composer = ttnn.create_mesh_composer(
        mesh_device,
        ttnn.MeshComposerConfig(
            dims=[0, 1],  # Axis 0: shard on tensor dim 0; Axis 1: replicated
        ),
    )

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=mesh_composer,
        dtype=torch.bfloat16,
    )

    # Step 7: Compute PCC and verify correctness
    logger.info("Computing PCC between torch and ttnn combine outputs...")

    # Quick sanity check of first elements
    logger.info(f"Sample torch output [0, 0, 0, :5]: {torch_output[0, 0, 0, :5]}")
    logger.info(f"Sample ttnn output [0, 0, 0, :5]:  {tt_output_torch[0, 0, 0, :5]}")
    if num_chips > 1:
        logger.info(f"Sample torch output [1, 0, 0, :5]: {torch_output[1, 0, 0, :5]}")
        logger.info(f"Sample ttnn output [1, 0, 0, :5]:  {tt_output_torch[1, 0, 0, :5]}")

    # Detailed per-chip, per-token, per-expert comparison
    data_ok = True
    mismatches = []
    matches = 0
    total_slots = 0

    logger.info("Comparing ALL combine output slots...")
    for chip_id in range(num_chips):
        for token_id in range(seq_len_per_chip):
            for topk_idx in range(num_experts_per_tok):
                total_slots += 1
                torch_data = torch_output[chip_id, token_id, topk_idx]
                ttnn_data = tt_output_torch[chip_id, token_id, topk_idx]

                if torch.allclose(torch_data, ttnn_data, atol=1e-2, rtol=1e-2):
                    matches += 1
                else:
                    data_ok = False
                    max_diff = torch.max(torch.abs(torch_data - ttnn_data)).item()
                    mismatches.append((chip_id, token_id, topk_idx, max_diff))

    # Report statistics
    logger.info(f"Matches: {matches}/{total_slots} ({100.0*matches/total_slots:.2f}%)")

    if not data_ok:
        # Show first 10 mismatches in detail
        logger.warning(f"Found {len(mismatches)} mismatches. Showing first 10:")
        for i, (chip_id, token_id, topk_idx, max_diff) in enumerate(mismatches[:10]):
            torch_sample = torch_output[chip_id, token_id, topk_idx, :5]
            ttnn_sample = tt_output_torch[chip_id, token_id, topk_idx, :5]
            logger.error(
                f"  [{i}] Mismatch at chip={chip_id}, token={token_id}, topk={topk_idx}: "
                f"max_diff={max_diff:.6f}"
            )
            logger.error(f"      torch[:5]={torch_sample}")
            logger.error(f"      ttnn[:5]={ttnn_sample}")

        # Show per-chip statistics
        logger.info("\nPer-chip statistics:")
        for chip_id in range(num_chips):
            chip_mismatches = [m for m in mismatches if m[0] == chip_id]
            chip_total = seq_len_per_chip * num_experts_per_tok
            chip_matches = chip_total - len(chip_mismatches)
            logger.info(
                f"  Chip {chip_id}: {chip_matches}/{chip_total} matches "
                f"({100.0*chip_matches/chip_total:.2f}%)"
            )

    # Assert all data matches
    assert data_ok, f"Combine data mismatch! {matches}/{total_slots} slots matched. Check logs for details."

    logger.info("✅ TTNN combine operation matches torch reference!")
