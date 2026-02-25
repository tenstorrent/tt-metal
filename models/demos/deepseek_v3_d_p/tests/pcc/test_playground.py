import pytest
import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule

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
        host_tt_dispatched_buffer = ttnn.to_torch(tt_dispatched_buffer, mesh_composer=mesh_composer)
        host_tt_dispatched_metadata = ttnn.to_torch(tt_dispatch_metadata, mesh_composer=mesh_composer)

        logger.info(f"{host_tt_dispatched_buffer[0][0][0][0]=}")
        logger.info(f"{host_tt_dispatched_buffer[1][0][0][0]=}")
        logger.warning(f"{host_tt_dispatched_metadata.shape=}")
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
        return tt_dispatched_buffer, tt_dispatch_metadata, tt_chip_to_routed_expert_tokens


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
                    y[chip, token, topk_indice] += dispatched[chips, experts, i]

        return y


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
                indices[chip, token, k] = expert_idx % n_routed_experts
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


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor",
    [
        (32, 64, 16, 4, 2, 2),
        # (512, 32, 256, 8, 4, 2),
        # (4096, 32, 256, 8, 32, 2),
    ],
    # ids=["xs", "small", "large"],
)
# @pytest.mark.parametrize(
#     "device_params",
#     [
#         {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
#     ],
#     indirect=["device_params"],
# )
@pytest.mark.parametrize(
    "mesh_device",
    [
        (2, 1),  # SP=2, TP=1
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize("use_predictable_data", [True, False], ids=["predictable", "random"])
def test_ttnn_dispatch_combine(
    mesh_device,
    seq_len_per_chip,
    hidden_dim,
    n_routed_experts,
    num_experts_per_tok,
    num_chips,
    capacity_factor,
    use_predictable_data,
):
    print("\n")

    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip, n_routed_experts, num_experts_per_tok, num_chips, capacity_factor
    )
    logger.info(f"{experts_per_chip=}, {metadata_len=}, {max_dispatched_tokens_per_expert=}")
    num_devices = mesh_device.get_num_devices()
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

    # Compute offset information (needed for comparison later)
    chip_to_n_routed_expert_counter = torch.zeros((num_chips, n_routed_experts), dtype=torch.int32)
    for chip in range(num_chips):
        for token in range(seq_len_per_chip):
            for topk_indice in range(num_experts_per_tok):
                routed_expert = indices[chip, token, topk_indice]
                chip_to_n_routed_expert_counter[chip, routed_expert] += 1

    cum_sum = torch.cumsum(chip_to_n_routed_expert_counter, dim=0)
    chip_to_n_routed_expert_offset = torch.vstack([torch.zeros([1, n_routed_experts], dtype=torch.int32), cum_sum[:-1]])
    chip_to_routed_expert_tokens = cum_sum[-1].view(num_chips, experts_per_chip)

    # Forward pass through dispatch module
    tt_dispatched, tt_metadata, tt_experts_counter = tt_dispatch_module(tt_x, tt_weights, tt_indices)

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
    tt_out_experts_counter = ttnn.to_torch(tt_experts_counter, mesh_composer=mesh_composer)

    # Kernel counter is garbage - use torch-computed counter instead
    tt_out_experts_counter = chip_to_routed_expert_tokens.to(torch.int32)

    # Quick sanity check of first elements
    logger.info(f"{tt_out_dispatched[0][0][0][0]=} | {dispatched[0][0][0][0]=}")
    logger.info(f"{tt_out_dispatched[1][0][0][0]=} | {dispatched[1][0][0][0]=}")

    # Compare local dispatch only (chip i writes to experts i*experts_per_chip ... (i+1)*experts_per_chip-1)
    logger.info("Comparing locally-dispatched slots only...")

    for chip_id in range(num_chips):
        for expert_id in range(experts_per_chip):
            # Compute global expert index
            global_expert_idx = chip_id * experts_per_chip + expert_id

            # Get count for this (chip, expert) pair
            count = int(chip_to_n_routed_expert_counter[chip_id, global_expert_idx])

            # Get start offset from offset tensor
            start_offset = int(chip_to_n_routed_expert_offset[chip_id, global_expert_idx])

            # Compare slots [start_offset : start_offset + count]
            for slot_idx in range(count):
                slot = start_offset + slot_idx

                # Compare data
                torch_data = dispatched[chip_id, expert_id, slot]
                kernel_data = tt_out_dispatched[chip_id, expert_id, slot]

                data_match = torch.allclose(torch_data, kernel_data, atol=1e-6)
                assert data_match, (
                    f"Data mismatch at chip={chip_id}, expert={expert_id}, slot={slot}: "
                    f"max_diff={torch.max(torch.abs(torch_data - kernel_data)).item()}"
                )

                # Compare metadata (first 4 fields: chip, token, k, routed_expert)
                # Skip weight field (index 4) as torch stores float value while kernel stores bfloat16 bits
                torch_meta = metadata[chip_id, expert_id, slot, :4]
                kernel_meta = tt_out_metadata[chip_id, expert_id, slot, :4]

                meta_match = torch.allclose(torch_meta, kernel_meta, atol=1e-6)
                assert meta_match, (
                    f"Metadata mismatch at chip={chip_id}, expert={expert_id}, slot={slot}: "
                    f"torch={torch_meta}, kernel={kernel_meta}"
                )

    logger.info("✓ All locally-dispatched slots match")

    # Verify remote dispatch is NOT implemented (should fail when you implement it)
    logger.info("Verifying remote dispatch is NOT yet implemented...")

    remote_slots_checked = 0
    remote_dispatch_working = False

    # Check a few remote dispatch slots - they should NOT match (should be garbage)
    for chip_id in range(num_chips):
        for expert_id in range(experts_per_chip):
            global_expert_idx = chip_id * experts_per_chip + expert_id

            # Check if OTHER chips dispatch to this expert (remote dispatch)
            for other_chip in range(num_chips):
                if other_chip == chip_id:
                    continue  # Skip local dispatch

                count = int(chip_to_n_routed_expert_counter[other_chip, global_expert_idx])
                if count == 0:
                    continue  # No remote dispatch from this chip to this expert

                # Found remote dispatch - check if kernel output matches torch
                start_offset = int(chip_to_n_routed_expert_offset[other_chip, global_expert_idx])
                slot = start_offset

                torch_data = dispatched[chip_id, expert_id, slot]
                kernel_data = tt_out_dispatched[chip_id, expert_id, slot]

                remote_slots_checked += 1

                # If they match, remote dispatch is working!
                if torch.allclose(torch_data, kernel_data, atol=1e-6):
                    remote_dispatch_working = True
                    logger.error(
                        f"❌ Remote dispatch is WORKING! Slot matched: "
                        f"chip {other_chip} -> expert {global_expert_idx} (on chip {chip_id}), slot {slot}"
                    )
                    logger.error(
                        "This test needs to be updated to compare remote dispatch slots. "
                        "Remove the remote dispatch check and include remote slots in comparison!"
                    )
                    break

            if remote_dispatch_working:
                break
        if remote_dispatch_working:
            break

    if remote_dispatch_working:
        raise AssertionError(
            "Remote dispatch appears to be implemented! "
            "Update this test to compare remote dispatch slots (not just local). "
            "Remove the remote dispatch verification section."
        )
    elif remote_slots_checked > 0:
        logger.info(
            f"✓ Verified {remote_slots_checked} remote slots do NOT match (remote dispatch not yet implemented)"
        )
    else:
        logger.info("ℹ No remote dispatch slots to check in this test configuration")

    # torch.set_printoptions(profile="full")
    # # logger.info(f"{indices.shape=}")
    # # logger.info(f"{indices=}")
    # logger.info(f"{tt_out_dispatched=}")
    # logger.info(f"{dispatched=}")
    # torch.set_printoptions(profile="default")

    # torch.set_printoptions(profile="full")
    # logger.info(f"{experts_counter.shape=}")
    # logger.info(f"{metadata.shape=}")
    # logger.info(f"{dispatched.shape=}")
    # torch.set_printoptions(profile="default")

    # # Forward pass through combine module
    # y = combine_module(
    #     dispatched,
    #     metadata,
    #     experts_counter,
    # )
    # logger.info(f"{y.shape=}")
    # y /= num_experts_per_tok  # since we are summing contributions from multiple experts, we need to average them
    # y = y.sum(dim=2)  # sum contributions from multiple experts per token
    # logger.info(f"{y.shape=}")
    # assert torch.allclose(
    #     x, y, atol=1e-6
    # ), f"Expected output to match input, but got max diff {torch.max(torch.abs(x-y)).item()}"
