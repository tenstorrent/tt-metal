# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Core expert operations - pure functions for composability."""

import ttnn

from .config import ExpertConfig


def apply_swiglu(gate, up, config: ExpertConfig):
    """
    Apply MiniMax-M3 clamped "swigluoai" SwiGLU: (up + 1) * (gate * sigmoid(alpha * gate)),
    with gate clamped to max=swiglu_limit and up clamped to [-swiglu_limit, swiglu_limit].

    This is the gpt-oss SwiGLU variant (anchor: transformers gpt_oss modeling_gpt_oss.py
    lines 119-122; mirrors models/demos/gpt_oss/tt/experts/operations.py). The M3 deltas
    vs M2's plain SiLU SwiGLU are: the gate/up clamp, the `alpha` inside the sigmoid, and
    the `(up + 1)` linear term.

    Args:
        gate: Gate projection output
        up: Up projection output
        config: Expert configuration carrying `swiglu_limit` and `alpha`

    Returns:
        Activated tensor
    """
    # Clamp gate (upper only) and up (symmetric).
    gate = ttnn.clamp(gate, min=None, max=config.swiglu_limit, output_tensor=gate)
    up = ttnn.clamp(up, min=-config.swiglu_limit, max=config.swiglu_limit, output_tensor=up)

    # glu = gate * sigmoid(alpha * gate)
    gate_alpha = ttnn.mul(gate, config.alpha)
    gate_sigmoid = ttnn.sigmoid(gate_alpha)
    gate_alpha.deallocate(True)

    glu = ttnn.mul(gate, gate_sigmoid, output_tensor=gate)
    gate_sigmoid.deallocate(True)

    # out = (up + 1) * glu
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
        tensor, num_links=ccl_manager.num_links, topology=ccl_manager.topology, cluster_axis=mesh_config.ep_axis
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
        tensor, num_links=ccl_manager.num_links, topology=ccl_manager.topology, cluster_axis=mesh_config.tp_axis
    )
    tensor.deallocate(True)

    return tensor_allreduced


def apply_sequence_parallel_allgather(tensor, mesh_config, ccl_manager):
    """Apply sequence parallel allgather communication."""
    tensor_gathered = ttnn.all_gather(
        tensor, dim=-2, num_links=ccl_manager.num_links, topology=ccl_manager.topology, cluster_axis=mesh_config.sp_axis
    )
    tensor.deallocate(True)
    return tensor_gathered
