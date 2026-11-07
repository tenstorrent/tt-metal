# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Core expert operations - pure functions for composability."""

import ttnn

from .. import opt_switches
from .config import ExpertConfig


def use_decode_ar():
    """GPT_OSS_OPT_DECODE_AR: one-op decode TP all-reduce (decode_tp_all_reduce)."""
    return opt_switches.on("DECODE_AR")


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


def decode_tp_all_reduce(tensor, mesh_config, ccl_manager, hidden_size, slot):
    """GPT_OSS_OPT_DECODE_AR=1 (callers use it only when use_decode_ar()): decode
    TP all-reduce of a one-token [1, 1, 1, hidden] partial sum with the one-op all_reduce_async of the batched decode
    layer (batched_decode.DecodeAllReduce: width-sharded over 30 cores, persistent scratch buffer + global semaphore
    per call site) instead of ttnn.all_reduce; returned as BF8 L1 interleaved."""
    all_reduce = ccl_manager.get_batched_decode_all_reduce(hidden_size, mesh_config.tp_axis)
    reduced = all_reduce(tensor, slot)
    out = ttnn.to_memory_config(reduced, ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
    reduced.deallocate(True)
    return out


def apply_sequence_parallel_allgather(tensor, mesh_config, ccl_manager):
    """Apply sequence parallel allgather communication."""
    tensor_gathered = ttnn.all_gather(
        tensor, dim=-2, num_links=ccl_manager.num_links, topology=ttnn.Topology.Ring, cluster_axis=mesh_config.sp_axis
    )
    tensor.deallocate(True)
    return tensor_gathered
