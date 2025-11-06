# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Core expert operations - pure functions for composability."""

import ttnn

from .config import ExpertConfig


def apply_swiglu(gate, up, config: ExpertConfig):
    """
    Apply SwiGLU activation: gate * sigmoid(alpha * gate) * (up + 1)

    Args:
        gate: Gate projection output
        up: Up projection output
        config: Expert configuration with alpha and limits

    Returns:
        Activated tensor
    """
    # Clamp gate and up
    gate = ttnn.clamp(gate, min=None, max=config.swiglu_limit, output_tensor=gate)
    up = ttnn.clamp(up, min=-config.swiglu_limit, max=config.swiglu_limit, output_tensor=up)

    # SwiGLU: gate * sigmoid(alpha * gate) * (up + 1)
    gate_alpha = ttnn.mul(gate, config.alpha)
    gate_sigmoid = ttnn.sigmoid(gate_alpha)
    gate_alpha.deallocate(True)

    glu = ttnn.mul(gate, gate_sigmoid, output_tensor=gate)
    gate_sigmoid.deallocate(True)

    up = ttnn.add(up, 1, output_tensor=up)
    result = ttnn.mul(up, glu, output_tensor=up)
    ttnn.deallocate(glu)

    return result


def apply_routing_weights(expert_output, routing_weights):
    """
    Apply routing weights to expert outputs.

    Args:
        expert_output: Output from experts [batch, num_experts, seq_len, hidden]
        routing_weights: Routing weights [batch, num_experts, seq_len, 1]

    Returns:
        Weighted output
    """
    return ttnn.mul(expert_output, routing_weights, output_tensor=expert_output)


def reduce_experts(expert_output):
    """
    Reduce across expert dimension.

    Args:
        expert_output: [batch, num_experts, seq_len, hidden]

    Returns:
        Reduced output [batch, 1, seq_len, hidden]
    """
    return ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(expert_output, dims=[1]))


def apply_expert_parallel_allreduce(tensor, mesh_config, ccl_manager):
    """Apply expert parallel allreduce communication."""
    return mesh_config.allreduce(tensor, ccl_manager, axis=mesh_config.ep_axis)


def apply_tensor_parallel_allreduce(tensor, mesh_config, mesh_device, ccl_manager, activation_dtype, seq_len, tp):
    """
    Apply tensor parallel allreduce communication.

    Handles dtype conversion for allreduce and converts back if needed.

    Args:
        tensor: Input tensor to allreduce
        mesh_config: Mesh configuration
        mesh_device: TTNN mesh device
        ccl_manager: Communication manager
        activation_dtype: Target dtype after allreduce
        seq_len: Sequence length
        tp: Tensor parallel degree

    Returns:
        Allreduced tensor
    """
    # Convert to bfloat16 for allreduce if needed
    if tensor.dtype != ttnn.bfloat16:
        tensor_16 = ttnn.typecast(tensor, ttnn.bfloat16)
        ttnn.deallocate(tensor)
    else:
        tensor_16 = tensor

    # Synchronize for prefill
    if seq_len > 1:
        ttnn.synchronize_device(mesh_device)  # ✅ Use explicit mesh_device

    tensor_allreduced = mesh_config.allreduce(
        tensor_16,
        ccl_manager,
        pad_size=192 if tp == 8 else 0,  # Optimal padding for TP=8
        axis=mesh_config.tp_axis,
    )
    tensor_16.deallocate(True)

    # Convert back to original dtype if needed
    if tensor_allreduced.dtype == ttnn.bfloat16 and activation_dtype != ttnn.bfloat16:
        tensor_converted = ttnn.typecast(tensor_allreduced, activation_dtype)
        tensor_allreduced.deallocate(True)
        return tensor_converted

    return tensor_allreduced


def apply_sequence_parallel_allgather(tensor, mesh_config, ccl_manager):
    """Apply sequence parallel allgather communication."""
    return mesh_config.allgather(tensor, ccl_manager, axis=mesh_config.sp_axis, dim=-2)
