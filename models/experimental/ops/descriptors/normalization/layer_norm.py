# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor


def layer_norm(
    input_tensor: "ttnn.Tensor",
    core_range_set: Optional["ttnn.CoreRangeSet"] = None,
    epsilon: float = 1e-12,
    weight: Optional["ttnn.Tensor"] = None,
    bias: Optional["ttnn.Tensor"] = None,
    residual_input_tensor: Optional["ttnn.Tensor"] = None,
    compute_kernel_config: Optional["ttnn.DeviceComputeKernelConfig"] = None,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
    program_config: Optional["ttnn.LayerNormProgramConfig"] = None,
) -> OpDescriptor:
    """
    Create an OpDescriptor for a layer norm operation.

    Uses the C++ layer_norm_descriptor() path which shares all decision logic
    (validation, factory selection, output allocation) with ttnn.layer_norm().

    Args:
        input_tensor: The input tensor (must be on device).
        core_range_set: Unused — kept for API compatibility.
        epsilon: Small constant for numerical stability (default: 1e-12).
        weight: Optional weight (gamma) tensor for scaling.
        bias: Optional bias (beta) tensor for shifting.
        residual_input_tensor: Optional residual tensor to add before normalization.
        compute_kernel_config: Optional compute kernel configuration.
        memory_config: Optional output memory configuration. Defaults to input's memory config.
        program_config: Optional program configuration. If not provided, one will be auto-generated.

    Returns:
        OpDescriptor containing the program descriptor, input tensors, and output tensors.
    """
    descriptor, output = ttnn.layer_norm_descriptor(
        input_tensor,
        epsilon=epsilon,
        weight=weight,
        bias=bias,
        residual_input_tensor=residual_input_tensor,
        memory_config=memory_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    inputs = [input_tensor]
    if residual_input_tensor is not None:
        inputs.append(residual_input_tensor)
    if weight is not None:
        inputs.append(weight)
    if bias is not None:
        inputs.append(bias)
    return OpDescriptor(descriptor, inputs, [output])


__all__ = ["layer_norm"]
