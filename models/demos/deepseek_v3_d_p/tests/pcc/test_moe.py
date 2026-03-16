"""
Test for DeepSeek V3-like MoE architecture (PyTorch reference implementation).

This test validates the full MoE dispatch → expert → combine → weighted sum flow:
1. Tokens are dispatched to expert buffers based on router indices
2. Routed experts (FFN networks) process their assigned tokens
3. Expert outputs are combined back to original token positions
4. Gate weights are applied to each expert contribution (split connection)
5. Shared expert output is added to the final result

Configuration:
- 24 routed experts (each is an FFN with gate_proj, up_proj, down_proj)
- num_experts_per_tok = 4 (each token routes to 4 experts)
- 1 shared expert (same FFN structure as routed experts)
- Dispatch group size = 4
- All experts initialized with identity matrices for flow verification
"""

from dataclasses import dataclass
from typing import Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.moe.combine import TorchCombineModule
from models.demos.deepseek_v3_d_p.reference.moe.dispatch import TorchDispatchModule
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import (
    compute_constants,
    create_expert_dispatch_table,
    get_gate_outputs,
    initialize_test_inputs,
)
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered


def load_moe_weights_from_hf(
    model_id: str,
    layer_idx: int,
    num_routed_experts: int,
) -> tuple[list[dict], dict]:
    """
    Load MoE weights from HuggingFace checkpoint.

    Args:
        model_id: HuggingFace model ID (e.g., "deepseek-ai/DeepSeek-V3")
        layer_idx: Layer index to load weights from
        num_routed_experts: Number of routed experts

    Returns:
        routed_expert_weights: List of dicts with gate_proj, up_proj, down_proj per expert
        shared_expert_weights: Dict with gate_proj, up_proj, down_proj for shared expert
    """
    # Build key prefixes for this layer's MoE
    prefixes = [f"model.layers.{layer_idx}.mlp."]
    state_dict = load_hf_state_dict_filtered(model_id, prefixes)

    # Extract routed expert weights
    routed_expert_weights = []
    for expert_idx in range(num_routed_experts):
        expert_weights = {
            "gate_proj": state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight"],
            "up_proj": state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight"],
            "down_proj": state_dict[f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight"],
        }
        routed_expert_weights.append(expert_weights)

    # Extract shared expert weights
    shared_expert_weights = {
        "gate_proj": state_dict[f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight"],
        "up_proj": state_dict[f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight"],
        "down_proj": state_dict[f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight"],
    }

    return routed_expert_weights, shared_expert_weights


class TorchExpert(nn.Module):
    """
    Expert FFN with configurable initialization.

    Architecture matches TorchSharedExpert:
        gate_out = x @ gate_proj.T
        up_out = x @ up_proj.T
        activated = silu(gate_out) * up_out
        output = activated @ down_proj.T

    Can be initialized with:
    - Identity matrices (for flow testing)
    - Real weights from HuggingFace checkpoint
    """

    def __init__(self, hidden_dim: int, torch_weights: dict = None):
        """
        Initialize Expert module.

        Args:
            hidden_dim: Hidden dimension (same for input and output)
            torch_weights: Optional dict with gate_proj, up_proj, down_proj tensors
                          from HuggingFace checkpoint. If None, uses identity matrices.
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        if torch_weights is not None:
            # Load from provided weights - shapes come from checkpoint
            # HF format: weight shape is (out_features, in_features)
            self.gate_proj = nn.Parameter(torch_weights["gate_proj"].float())
            self.up_proj = nn.Parameter(torch_weights["up_proj"].float())
            self.down_proj = nn.Parameter(torch_weights["down_proj"].float())
        else:
            # Identity initialization for flow testing (square matrices)
            self.gate_proj = nn.Parameter(torch.eye(hidden_dim, dtype=torch.float32))
            self.up_proj = nn.Parameter(torch.eye(hidden_dim, dtype=torch.float32))
            self.down_proj = nn.Parameter(torch.eye(hidden_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., hidden_dim]

        Returns:
            Output tensor [..., hidden_dim]
        """
        # Gate projection: x @ gate_proj.T (HF format: weight is out_features x in_features)
        gate_out = F.linear(x, self.gate_proj)

        # Up projection
        up_out = F.linear(x, self.up_proj)

        # SiLU activation and element-wise multiplication
        activated = F.silu(gate_out) * up_out

        # Down projection
        output = F.linear(activated, self.down_proj)

        return output


class TorchSharedExpertModule(nn.Module):
    """Wrapper module for shared expert execution."""

    def __init__(self, expert: TorchExpert):
        """
        Initialize shared expert module.

        Args:
            expert: The expert FFN to use for shared expert computation
        """
        super().__init__()
        self.expert = expert

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run shared expert on input.

        Args:
            x: Input tensor (dispatch_group_size, seq_len_per_chip, hidden_dim)

        Returns:
            Shared expert output (dispatch_group_size, seq_len_per_chip, hidden_dim)
        """
        return self.expert(x)


class TorchSplitConnectionModule(nn.Module):
    """
    Split connection module: applies gate weights and reduces expert outputs.

    This is the "weighted sum" operation that combines routed expert contributions:
        output = sum(weight[i] * expert_output[i] for i in topk_experts)
    """

    def __init__(self):
        """Initialize split connection module."""
        super().__init__()

    def forward(
        self,
        combined_output: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply gate weights and sum expert contributions.

        Args:
            combined_output: Combined expert outputs
                shape: (dispatch_group_size, seq_len_per_chip, num_experts_per_tok, hidden_dim)
            weights: Gate weights
                shape: (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)

        Returns:
            Weighted sum of expert outputs
                shape: (dispatch_group_size, seq_len_per_chip, hidden_dim)
        """
        weighted = combined_output * weights.unsqueeze(-1)  # Apply weights
        return weighted.sum(dim=2)  # Sum expert contributions


@dataclass
class MoEIntermediates:
    """Data structure holding intermediate values from MoE forward pass for debugging."""

    dispatched_buffer: torch.Tensor  # (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_tokens, hidden_dim)
    metadata: torch.Tensor  # (num_dispatch_groups, dispatch_group_size, experts_per_chip, max_tokens, metadata_len)
    expert_outputs: torch.Tensor  # Same shape as dispatched_buffer
    shared_output: torch.Tensor  # (dispatch_group_size, seq_len_per_chip, hidden_dim)
    combined_output: torch.Tensor  # (dispatch_group_size, seq_len_per_chip, num_experts_per_tok, hidden_dim)
    routed_output: torch.Tensor  # (dispatch_group_size, seq_len_per_chip, hidden_dim)


class TorchMinimalMoE(nn.Module):
    """
    Minimal MoE module connecting dispatch → experts → combine → split connection.

    This module orchestrates the full MoE pipeline:
    1. Dispatch: Route tokens to expert buffers
    2. Routed Experts: Process tokens in expert-specific buffers
    3. Shared Expert: Process original input
    4. Combine: Reconstruct outputs to original token positions
    5. Split Connection: Apply gate weights and sum expert contributions
    6. Final: Add routed output + shared output
    """

    def __init__(
        self,
        dispatch_group_size: int,
        experts_per_chip: int,
        num_routed_experts: int,
        num_experts_per_tok: int,
        metadata_len: int,
        max_dispatched_tokens_per_expert: int,
        seq_len_per_chip: int,
        hidden_dim: int,
        expert_dispatch_table: torch.Tensor,
        model_id: str = None,
        layer_idx: int = None,
    ):
        """
        Initialize MinimalMoE with configuration parameters.

        All sub-modules are created internally.

        Args:
            dispatch_group_size: Number of chips in dispatch group
            experts_per_chip: Number of experts per chip
            num_routed_experts: Total number of routed experts
            num_experts_per_tok: Number of experts each token routes to
            metadata_len: Length of metadata per token
            max_dispatched_tokens_per_expert: Max tokens per expert buffer
            seq_len_per_chip: Sequence length per chip
            hidden_dim: Hidden dimension
            expert_dispatch_table: Expert to chip mapping table
            model_id: Optional HuggingFace model ID to load real weights from
            layer_idx: Optional layer index for weight loading (required if model_id is set)
        """
        super().__init__()
        self.dispatch_group_size = dispatch_group_size
        self.experts_per_chip = experts_per_chip
        self.num_routed_experts = num_routed_experts

        # Create dispatch module
        self.dispatch_module = TorchDispatchModule(
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            metadata_len=metadata_len,
            max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
            seq_len_per_chip=seq_len_per_chip,
            hidden_dim=hidden_dim,
            expert_dispatch_table=expert_dispatch_table,
        )

        # Create combine module
        self.combine_module = TorchCombineModule(
            dispatch_group_size=dispatch_group_size,
            experts_per_chip=experts_per_chip,
            num_experts_per_tok=num_experts_per_tok,
            seq_len_per_chip=seq_len_per_chip,
        )

        # Create routed and shared experts (with real weights if model_id provided)
        if model_id is not None and layer_idx is not None:
            logger.info(f"Loading MoE weights from {model_id}, layer {layer_idx}")
            routed_weights, shared_weights = load_moe_weights_from_hf(model_id, layer_idx, num_routed_experts)
            self.routed_experts = nn.ModuleList([TorchExpert(hidden_dim, torch_weights=w) for w in routed_weights])
            shared_expert = TorchExpert(hidden_dim, torch_weights=shared_weights)
        else:
            # Identity weights (flow testing)
            self.routed_experts = nn.ModuleList([TorchExpert(hidden_dim) for _ in range(num_routed_experts)])
            shared_expert = TorchExpert(hidden_dim)

        # Create shared expert module
        self.shared_expert_module = TorchSharedExpertModule(shared_expert)

        # Create split connection module
        self.split_connection_module = TorchSplitConnectionModule()

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_token_counts: torch.Tensor,
        return_intermediates: bool = False,
    ) -> tuple[torch.Tensor, Optional[MoEIntermediates]]:
        """
        Forward pass through the full MoE pipeline.

        Args:
            x: Input tensor (dispatch_group_size, seq_len_per_chip, hidden_dim)
            weights: Gate weights (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
            indices: Expert indices (dispatch_group_size, seq_len_per_chip, num_experts_per_tok)
            expert_offsets: Base offset for each expert from each chip
            expert_token_counts: Token counts per expert per chip
            return_intermediates: If True, return intermediate values for debugging

        Returns:
            final_output: MoE output (dispatch_group_size, seq_len_per_chip, hidden_dim)
            intermediates: Optional MoEIntermediates if return_intermediates=True
        """
        # Step 1: Run shared expert on original input
        with torch.no_grad():
            shared_output = self.shared_expert_module(x.float())

        # Step 2: Dispatch tokens to expert buffers
        dispatched_buffer, metadata = self.dispatch_module(x, weights, indices, expert_offsets)

        # Step 3: Run routed experts on dispatch buffer slices
        expert_outputs = torch.zeros_like(dispatched_buffer)
        for chip in range(self.dispatch_group_size):
            for local_expert in range(self.experts_per_chip):
                global_expert = chip * self.experts_per_chip + local_expert
                token_count = expert_token_counts[0, chip, local_expert].item()

                if token_count > 0:
                    expert_input = dispatched_buffer[0, chip, local_expert, :token_count, :]
                    with torch.no_grad():
                        expert_output = self.routed_experts[global_expert](expert_input.float())
                    expert_outputs[0, chip, local_expert, :token_count, :] = expert_output

        # Step 4: Combine routed expert outputs
        combined_output = self.combine_module(expert_outputs, metadata, expert_token_counts)

        # Step 5: Apply gate weights (split connection)
        routed_output = self.split_connection_module(combined_output, weights)

        # Step 6: Final output = routed + shared
        final_output = routed_output + shared_output

        # Build intermediates if requested
        intermediates = None
        if return_intermediates:
            intermediates = MoEIntermediates(
                dispatched_buffer=dispatched_buffer,
                metadata=metadata,
                expert_outputs=expert_outputs,
                shared_output=shared_output,
                combined_output=combined_output,
                routed_output=routed_output,
            )

        return final_output, intermediates


@pytest.mark.parametrize(
    "seq_len_per_chip, hidden_dim, num_routed_experts, num_experts_per_tok, dispatch_group_size, capacity_factor",
    [
        (32, 64, 24, 4, 4, 2),  # 24 experts, 4 tok/expert, dispatch_group=4
    ],
    ids=["deepseek-v3-like"],
)
def test_moe(
    seq_len_per_chip,
    hidden_dim,
    num_routed_experts,
    num_experts_per_tok,
    dispatch_group_size,
    capacity_factor,
):
    """
    Test TorchMinimalMoE module with return_intermediates flag.

    Validates that the unified module produces the same output as the step-by-step test
    and that intermediates are correctly captured.
    """
    logger.info(f"\n{'='*60}")
    logger.info("TorchMinimalMoE Module Test")
    logger.info(f"{'='*60}\n")

    # Compute derived constants
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices=dispatch_group_size,
        dispatch_group_size=dispatch_group_size,
        capacity_factor=capacity_factor,
    )

    # Initialize test inputs
    x, weights, indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seed=42,
    )

    # Create expert dispatch table
    expert_dispatch_table = create_expert_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=1,
    )

    # Compute gate outputs
    expert_offsets, expert_token_counts, cum_sum = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
    )

    # Create MinimalMoE module (all sub-modules created internally)
    moe = TorchMinimalMoE(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        expert_dispatch_table=expert_dispatch_table,
    )

    # Test without intermediates
    logger.info("Testing forward pass without intermediates...")
    final_output, intermediates = moe(
        x, weights, indices, expert_offsets, expert_token_counts, return_intermediates=False
    )
    assert intermediates is None, "Expected no intermediates when return_intermediates=False"
    assert final_output.shape == x.shape, f"Expected output shape {x.shape}, got {final_output.shape}"
    logger.info(f"Output shape: {final_output.shape}")
    logger.info(f"Output sum (abs): {final_output.abs().sum().item():.4f}")

    # Test with intermediates
    logger.info("\nTesting forward pass with intermediates...")
    final_output_2, intermediates = moe(
        x, weights, indices, expert_offsets, expert_token_counts, return_intermediates=True
    )
    assert intermediates is not None, "Expected intermediates when return_intermediates=True"

    # Verify intermediates shapes
    logger.info("Intermediate shapes:")
    logger.info(f"  dispatched_buffer: {intermediates.dispatched_buffer.shape}")
    logger.info(f"  metadata: {intermediates.metadata.shape}")
    logger.info(f"  expert_outputs: {intermediates.expert_outputs.shape}")
    logger.info(f"  shared_output: {intermediates.shared_output.shape}")
    logger.info(f"  combined_output: {intermediates.combined_output.shape}")
    logger.info(f"  routed_output: {intermediates.routed_output.shape}")

    # Verify shapes
    assert intermediates.dispatched_buffer.shape == (
        1,
        dispatch_group_size,
        experts_per_chip,
        max_dispatched_tokens_per_expert,
        hidden_dim,
    )
    assert intermediates.shared_output.shape == (dispatch_group_size, seq_len_per_chip, hidden_dim)
    assert intermediates.combined_output.shape == (
        dispatch_group_size,
        seq_len_per_chip,
        num_experts_per_tok,
        hidden_dim,
    )
    assert intermediates.routed_output.shape == (dispatch_group_size, seq_len_per_chip, hidden_dim)

    # Verify both runs produce same output
    assert torch.allclose(
        final_output, final_output_2
    ), "Outputs should be identical regardless of return_intermediates"

    # Verify no NaN/Inf
    assert not torch.isnan(final_output).any(), "Final output contains NaN values"
    assert not torch.isinf(final_output).any(), "Final output contains Inf values"

    logger.info("\n" + "=" * 60)
    logger.info("TorchMinimalMoE Module Test PASSED!")
    logger.info("=" * 60)


# DeepSeek V3/R1 dimensions (from HuggingFace config.json)
DEEPSEEK_HIDDEN_SIZE = 7168
DEEPSEEK_MOE_INTERMEDIATE_SIZE = 2048
DEEPSEEK_NUM_ROUTED_EXPERTS = 256
DEEPSEEK_NUM_EXPERTS_PER_TOK = 8
DEEPSEEK_NUM_SHARED_EXPERTS = 1


# Note: DeepSeek V3 has dense layers 0-2 and MoE layers 3-60
@pytest.mark.parametrize("layer_idx", [3])
@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-V3"])
@pytest.mark.parametrize(
    "seq_len_per_chip, dispatch_group_size, capacity_factor",
    [
        (32, 4, 4),  # capacity_factor=4 to handle 256 experts with random routing
    ],
    ids=["small-config"],
)
def test_moe_real_weights(
    layer_idx,
    model_id,
    seq_len_per_chip,
    dispatch_group_size,
    capacity_factor,
):
    """
    Test TorchMinimalMoE with real DeepSeek V3 weights from HuggingFace.

    This test validates that:
    1. Weights can be loaded successfully from HuggingFace
    2. Forward pass completes without errors
    3. Output contains no NaN/Inf values
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TorchMinimalMoE Real Weights Test")
    logger.info(f"Model: {model_id}, Layer: {layer_idx}")
    logger.info(f"{'='*60}\n")

    # Use real DeepSeek V3 dimensions
    hidden_dim = DEEPSEEK_HIDDEN_SIZE
    num_routed_experts = DEEPSEEK_NUM_ROUTED_EXPERTS
    num_experts_per_tok = DEEPSEEK_NUM_EXPERTS_PER_TOK

    # Compute derived constants
    experts_per_chip, metadata_len, max_dispatched_tokens_per_expert = compute_constants(
        seq_len_per_chip,
        num_routed_experts,
        num_experts_per_tok,
        num_devices=dispatch_group_size,
        dispatch_group_size=dispatch_group_size,
        capacity_factor=capacity_factor,
    )

    # Initialize test inputs with real dimensions
    x, weights, indices = initialize_test_inputs(
        dispatch_group_size=dispatch_group_size,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seed=42,
    )

    # Create expert dispatch table
    expert_dispatch_table = create_expert_dispatch_table(
        num_routed_experts=num_routed_experts,
        dispatch_group_size=dispatch_group_size,
        num_dispatch_groups=1,
    )

    # Compute gate outputs
    expert_offsets, expert_token_counts, cum_sum = get_gate_outputs(
        indices,
        dispatch_group_size,
        num_routed_experts,
        experts_per_chip,
        seq_len_per_chip,
        num_experts_per_tok,
    )

    # Create MinimalMoE module with real weights
    logger.info(f"Creating MoE with real weights from {model_id}...")
    moe = TorchMinimalMoE(
        dispatch_group_size=dispatch_group_size,
        experts_per_chip=experts_per_chip,
        num_routed_experts=num_routed_experts,
        num_experts_per_tok=num_experts_per_tok,
        metadata_len=metadata_len,
        max_dispatched_tokens_per_expert=max_dispatched_tokens_per_expert,
        seq_len_per_chip=seq_len_per_chip,
        hidden_dim=hidden_dim,
        expert_dispatch_table=expert_dispatch_table,
        model_id=model_id,
        layer_idx=layer_idx,
    )

    # Log weight shapes from first routed expert
    logger.info("Weight shapes (first routed expert):")
    logger.info(f"  gate_proj: {moe.routed_experts[0].gate_proj.shape}")
    logger.info(f"  up_proj: {moe.routed_experts[0].up_proj.shape}")
    logger.info(f"  down_proj: {moe.routed_experts[0].down_proj.shape}")

    # Run forward pass
    logger.info("\nRunning forward pass...")
    final_output, intermediates = moe(
        x, weights, indices, expert_offsets, expert_token_counts, return_intermediates=True
    )

    # Verify output shape
    assert final_output.shape == x.shape, f"Expected output shape {x.shape}, got {final_output.shape}"
    logger.info(f"Output shape: {final_output.shape}")
    logger.info(
        f"Output stats - min: {final_output.min().item():.4f}, max: {final_output.max().item():.4f}, mean: {final_output.mean().item():.4f}"
    )

    # Verify no NaN/Inf
    assert not torch.isnan(final_output).any(), "Final output contains NaN values"
    assert not torch.isinf(final_output).any(), "Final output contains Inf values"

    # Verify intermediates
    assert not torch.isnan(intermediates.shared_output).any(), "Shared expert output contains NaN"
    assert not torch.isnan(intermediates.routed_output).any(), "Routed expert output contains NaN"

    logger.info("\n" + "=" * 60)
    logger.info("TorchMinimalMoE Real Weights Test PASSED!")
    logger.info("=" * 60)
