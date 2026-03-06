# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Expert-only MLP computation without all_to_all operations.

This module extracts the expert MLP computation from the ThroughputExperts module,
allowing it to be used in a unified MoE forward pass where all_to_all operations
are handled externally for consistency across backends.
"""

import ttnn

from .config import ThroughputExpertConfig, ThroughputProgramConfig
from .weights import ThroughputExpertWeights


def _apply_swiglu(
    gate: ttnn.Tensor, up: ttnn.Tensor, alpha: float, limit: float, memory_config: ttnn.MemoryConfig
) -> ttnn.Tensor:
    """Apply SwiGLU activation: (up + 1) * (gate * sigmoid(gate * alpha)).

    This implements the GPT-OSS SwiGLU variant with clamping:
    1. Clamp gate (min=None, max=limit) and up (min=-limit, max=limit)
    2. glu = gate * sigmoid(gate * alpha)
    3. result = (up + 1) * glu

    Args:
        gate: Gate projection output
        up: Up projection output
        alpha: Scaling factor for sigmoid
        limit: Clamping limit for swiglu
        memory_config: Output memory configuration

    Returns:
        Activated tensor with same shape as inputs
    """
    # Clamp gate (max only)
    gate_clamped = ttnn.clamp(gate, min=None, max=limit)
    ttnn.deallocate(gate)

    # Clamp up (both min and max)
    up_clamped = ttnn.clamp(up, min=-limit, max=limit)
    ttnn.deallocate(up)

    # Compute gate_alpha = gate * alpha
    gate_alpha = ttnn.mul(gate_clamped, alpha)

    # Compute gate_sigmoid = sigmoid(gate_alpha)
    gate_sigmoid = ttnn.sigmoid(gate_alpha)
    ttnn.deallocate(gate_alpha)

    # Compute glu = gate * gate_sigmoid
    glu = ttnn.mul(gate_clamped, gate_sigmoid, memory_config=memory_config)
    ttnn.deallocate(gate_clamped)
    ttnn.deallocate(gate_sigmoid)

    # Add 1 to up: up = up + 1
    ttnn.add(up_clamped, 1.0, output_tensor=up_clamped)

    # Multiply: result = up * glu
    result = ttnn.mul(up_clamped, glu, memory_config=memory_config)
    ttnn.deallocate(up_clamped)
    ttnn.deallocate(glu)

    return result


def expert_mlp_compute_only(
    experts_input: ttnn.Tensor,
    weights: ThroughputExpertWeights,
    config: ThroughputExpertConfig,
    memory_config: ttnn.MemoryConfig,
    program_config: ThroughputProgramConfig,
) -> ttnn.Tensor:
    """Expert MLP computation without all_to_all operations.

    This function performs the expert MLP computation (gate/up/down projections with
    SwiGLU activation) without any all_to_all dispatch or combine operations.
    It's extracted from expert_mlp_forward in decode.py/prefill.py to enable
    unified MoE forward pass where all_to_all operations are handled externally.

    Args:
        experts_input: Input tensor [1, num_experts_per_device, total_tokens, hidden_size]
        weights: Expert weights containing w1, w2, w3 (or w1_w3_fused) and biases
        config: Expert configuration with dimensions and activation parameters
        memory_config: Memory configuration for intermediate tensors
        program_config: Program configuration for matmul operations

    Returns:
        Expert output tensor [1, num_experts_per_device, total_tokens, hidden_size]
    """
    # Get total tokens from input shape
    total_tokens = experts_input.shape[2]

    # Build 1D multicast program configs sized for total_tokens (M dimension)
    down_matmul_config = program_config.get_down_config(n=config.hidden_size, m=total_tokens)

    # Choose between fused and unfused gate/up projection
    if weights.w1_w3_fused is not None:
        # ======================================================================
        # FUSED PATH: Single matmul for gate+up projections
        # ======================================================================
        assert weights.w1_w3_bias_fused is not None, "Fused bias must be present when using fused weights"

        # Get fused-specific matmul config (output size is 2*intermediate_size)
        fused_gate_up_matmul_config = program_config.get_fused_gate_up_config(
            n=weights.w1_w3_fused.shape[-1], m=total_tokens
        )

        # Fused projection: [1, E, total_tokens, H] x [1, E, H, 2*I] -> [1, E, total_tokens, 2*I]
        w1_w3_out = ttnn.matmul(experts_input, weights.w1_w3_fused, memory_config=memory_config)
        ttnn.deallocate(experts_input)

        # Add fused bias: [1, num_experts_per_device, 1, 2*I] broadcasts across total_tokens
        ttnn.add(w1_w3_out, weights.w1_w3_bias_fused, output_tensor=w1_w3_out)

        # Split into gate and up projections
        # w1_w3_out: [1, num_experts_per_device, total_tokens, 2*intermediate_size]
        # Split along last dimension: first half is gate, second half is up
        shape = w1_w3_out.shape

        # Extract gate projection (first half of last dimension)
        w1_out = ttnn.slice(
            w1_w3_out,
            [0, 0, 0, 0],
            [shape[0], shape[1], shape[2], config.intermediate_size],
            [1, 1, 1, 1],
        )

        # Extract up projection (second half of last dimension)
        w3_out = ttnn.slice(
            w1_w3_out,
            [0, 0, 0, config.intermediate_size],
            [shape[0], shape[1], shape[2], 2 * config.intermediate_size],
            [1, 1, 1, 1],
        )

        ttnn.deallocate(w1_w3_out)

    else:
        # ======================================================================
        # UNFUSED PATH: Separate matmuls for gate and up projections
        # ======================================================================
        assert weights.w1 is not None, "Unfused weights (w1) must be present when not using fused mode"
        assert weights.w3 is not None, "Unfused weights (w3) must be present when not using fused mode"
        assert weights.w1_bias is not None, "Unfused bias (w1_bias) must be present when not using fused mode"
        assert weights.w3_bias is not None, "Unfused bias (w3_bias) must be present when not using fused mode"

        # Get unfused-specific matmul config (output size is intermediate_size)
        gate_up_matmul_config = program_config.get_gate_up_config(n=config.intermediate_size, m=total_tokens)

        # Gate projection (w1)
        w1_out = ttnn.matmul(experts_input, weights.w1, memory_config=memory_config)
        # Bias: [1, num_experts_per_device, 1, I] broadcasts across total_tokens
        ttnn.add(w1_out, weights.w1_bias, output_tensor=w1_out)

        # Up projection (w3)
        w3_out = ttnn.matmul(experts_input, weights.w3, memory_config=memory_config)
        ttnn.deallocate(experts_input)
        # Bias: [1, num_experts_per_device, 1, I] broadcasts across total_tokens
        ttnn.add(w3_out, weights.w3_bias, output_tensor=w3_out)

    # SwiGLU activation: (up + 1) * (gate * sigmoid(gate * alpha))
    activated = _apply_swiglu(w1_out, w3_out, config.alpha, config.swiglu_limit, memory_config)

    # Down projection (w2): [1, E, total_tokens, I] x [1, E, I, H] -> [1, E, total_tokens, H]
    expert_output = ttnn.matmul(activated, weights.w2, memory_config=memory_config)
    ttnn.deallocate(activated)

    # Bias: [1, num_experts_per_device, 1, H] broadcasts across total_tokens
    ttnn.add(expert_output, weights.w2_bias, output_tensor=expert_output)

    # Return output maintaining [1, num_experts_per_device, total_tokens, hidden_size] shape
    return expert_output
