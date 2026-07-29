# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
    # SwiGLU with build-time alpha-fold: the gate weights are pre-scaled by alpha
    # (see weights.py), so `gate` here is already alpha*gate_raw. Thus
    # gate_raw*sigmoid(alpha*gate_raw) = silu(alpha*gate_raw)/alpha = silu(gate)/alpha,
    # and the 1/alpha is absorbed into the down weights. This lets us use a single
    # fused ttnn.silu instead of (mul-by-alpha + sigmoid + mul), removing ops/layer
    # with zero runtime correction. Clamp bound scales with alpha (clamp(alpha*g) at
    # alpha*limit == alpha*clamp(g at limit)).
    gate = ttnn.clamp(gate, min=None, max=config.alpha * config.swiglu_limit, output_tensor=gate)
    up = ttnn.clamp(up, min=-config.swiglu_limit, max=config.swiglu_limit, output_tensor=up)

    glu = ttnn.silu(gate, output_tensor=gate)  # = alpha * gate_raw*sigmoid(alpha*gate_raw)

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
    tensor_allreduced = ttnn.all_reduce(
        tensor, num_links=ccl_manager.num_links, topology=ttnn.Topology.Ring, cluster_axis=mesh_config.ep_axis
    )
    tensor.deallocate(True)
    return tensor_allreduced


def apply_tensor_parallel_allreduce(tensor, mesh_config, mesh_device, seq_len, ccl_manager):
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
    tensor_allreduced = ttnn.all_reduce(
        tensor, num_links=ccl_manager.num_links, topology=ttnn.Topology.Ring, cluster_axis=mesh_config.tp_axis
    )
    tensor.deallocate(True)

    return tensor_allreduced


def apply_sequence_parallel_allgather(tensor, mesh_config, ccl_manager):
    """Apply sequence parallel allgather communication."""
    tensor_gathered = ttnn.all_gather(
        tensor, dim=-2, num_links=ccl_manager.num_links, topology=ttnn.Topology.Ring, cluster_axis=mesh_config.sp_axis
    )
    tensor.deallocate(True)
    return tensor_gathered
