import pytest
import torch
from loguru import logger

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
        metadata_len: int,
        max_dispatched_tokens_per_expert: int,
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
        self.metadata_len = metadata_len
        self.max_dispatched_tokens_per_expert = max_dispatched_tokens_per_expert

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
        num_chips, seq_len, hidden_dim = x.shape
        num_experts_per_tok = indices.shape[-1]

        # Oversized buffer to simplify dispatch logic
        dispatched_shape = (num_chips, self.experts_per_chip, self.max_dispatched_tokens_per_expert, hidden_dim)
        dispatched_metadata_shape = (
            num_chips,
            self.experts_per_chip,
            self.max_dispatched_tokens_per_expert,
            self.metadata_len,
        )

        dispatched = torch.zeros(dispatched_shape, dtype=torch.float32)
        metadata = torch.ones(dispatched_metadata_shape, dtype=torch.float32) * -1

        ###
        # prep data for efficient dispatch: count tokens per expert per chip to compute offsets for where to write in the dispatched buffer
        chip_to_n_routed_expert_counter = torch.zeros(
            (num_chips, self.n_routed_experts), dtype=torch.int32
        )  # amount of tokens dispatched to each expert from each chip
        chip_to_n_routed_expert_offset = torch.zeros(
            (num_chips, self.n_routed_experts), dtype=torch.int32
        )  # base offset for each expert from each chip in the dispatched buffer
        chip_to_routed_expert_tokens = torch.zeros(
            (num_chips, self.experts_per_chip), dtype=torch.int32
        )  # total tokens dispatched to each expert per chip

        for chip in range(num_chips):
            for token in range(seq_len):
                for topk_indice in range(num_experts_per_tok):
                    routed_expert = indices[chip, token, topk_indice]
                    chip_to_n_routed_expert_counter[chip, routed_expert] += 1

        # this should be local to each chip
        cum_sum = torch.cumsum(chip_to_n_routed_expert_counter, dim=0)
        chip_to_n_routed_expert_offset = torch.vstack(
            [torch.zeros([1, self.n_routed_experts], dtype=torch.int32), cum_sum[:-1]]
        )  # base offset for each expert in the dispatched buffer
        # this should be local to each chip
        chip_to_routed_expert_tokens = cum_sum[-1].view(num_chips, self.experts_per_chip)

        ###
        # dispatching tokens and metadata to experts
        for chip in range(num_chips):
            for token in range(seq_len):
                for topk_indice in range(num_experts_per_tok):
                    routed_expert = indices[chip, token, topk_indice]
                    # logger.debug(f"Chip {chip} dispatching token {token} to expert [{topk_indice}]={routed_expert}")

                    expert_chip = routed_expert // self.experts_per_chip
                    expert_index_within_chip = routed_expert % self.experts_per_chip
                    dst_index = chip_to_n_routed_expert_offset[chip, routed_expert]

                    dispatched[expert_chip, expert_index_within_chip, dst_index] = x[chip, token]
                    metadata[expert_chip, expert_index_within_chip, dst_index] = torch.tensor(
                        [chip, token, topk_indice, routed_expert, weights[chip, token, topk_indice]]
                        + [0] * (self.metadata_len - 5),
                        dtype=torch.float32,
                    )
                    chip_to_n_routed_expert_offset[chip, routed_expert] += 1

        # chip_to_routed_expert_tokens is needed to run experts
        # metadata and chip_to_routed_expert_tokens are needed for combine step to route expert outputs back to original token positions
        return dispatched, metadata, chip_to_routed_expert_tokens


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
            (self.num_chips, self.seq_len_per_chip, self.num_experts_per_tok, hidden_dim), dtype=torch.float32
        )

        for chips in range(self.num_chips):
            for experts in range(self.experts_per_chip):
                for i in range(experts_counter[chips, experts]):
                    chip = int(metadata[chips, experts, i, 0])
                    token = int(metadata[chips, experts, i, 1])
                    topk_indice = int(metadata[chips, experts, i, 2])
                    y[chip, token, topk_indice] += dispatched[chips, experts, i]

        return y


def initialize_test_inputs(
    num_chips: int,
    seq_len_per_chip: int,
    hidden_dim: int,
    n_routed_experts: int,
    num_experts_per_tok: int,
    seed: int = 42,
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
    x = torch.randn(input_shape, dtype=torch.float32)

    weights_shape = (num_chips, seq_len_per_chip, num_experts_per_tok)
    indices_shape = (num_chips, seq_len_per_chip, num_experts_per_tok)

    weights = torch.randn(weights_shape, dtype=torch.float32)
    indices = torch.randint(0, n_routed_experts, indices_shape, dtype=torch.int32)

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
    experts_per_chip = n_routed_experts // num_chips
    metadata_len = 5  # chip, token, topk_indice, routed_expert, weight
    balanced_load = num_chips * seq_len_per_chip * num_experts_per_tok // n_routed_experts
    max_dispatched_tokens_per_expert = int(balanced_load * capacity_factor)

    print("\n")

    # Initialize inputs using helper function
    x, weights, indices = initialize_test_inputs(
        num_chips=num_chips,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        n_routed_experts=n_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        seed=42,
    )

    # Validate expert activations
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

    # Initialize dispatch and combine modules
    dispatch_module = TorchDispatchModule(
        num_chips=num_chips,
        experts_per_chip=experts_per_chip,
        n_routed_experts=n_routed_experts,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
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
