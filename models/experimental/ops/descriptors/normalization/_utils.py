# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Internal utilities for normalization descriptors.
"""

from typing import List, Optional, Tuple

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor


def _build_layernorm_io_tensors(
    input_tensor: "ttnn.Tensor",
    output_tensor: "ttnn.Tensor",
    weight: Optional["ttnn.Tensor"] = None,
    bias: Optional["ttnn.Tensor"] = None,
    residual_input_tensor: Optional["ttnn.Tensor"] = None,
    recip_tensor: Optional["ttnn.Tensor"] = None,
) -> Tuple[List["ttnn.Tensor"], List["ttnn.Tensor"]]:
    """Build the input and output tensors for a layernorm/rmsnorm operation."""
    inputs = [input_tensor]
    if residual_input_tensor is not None:
        inputs.append(residual_input_tensor)
    if weight is not None:
        inputs.append(weight)
    if bias is not None:
        inputs.append(bias)
    if recip_tensor is not None:
        inputs.append(recip_tensor)
    outputs = [output_tensor]
    return inputs, outputs


def _create_layernorm_op_descriptor(
    input_tensor: "ttnn.Tensor",
    compute_kernel_config: Optional["ttnn.DeviceComputeKernelConfig"],
    norm_type: "ttnn.LayerNormType",
    weight: Optional["ttnn.Tensor"] = None,
    bias: Optional["ttnn.Tensor"] = None,
    residual_input_tensor: Optional["ttnn.Tensor"] = None,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
    core_range_set: Optional["ttnn.CoreRangeSet"] = None,
    epsilon: float = 1e-12,
    program_config: Optional["ttnn.LayerNormProgramConfig"] = None,
) -> "OpDescriptor":
    """Create a layernorm/rmsnorm op descriptor."""

    # For non-sharded inputs, core_range_set is required
    if not input_tensor.is_sharded() and core_range_set is None:
        raise RuntimeError("core_range_set is required for non-sharded input tensors")

    if compute_kernel_config is None:
        raise ValueError("compute_kernel_config is required")

    # Use input's memory config if not provided
    output_mem_config = memory_config if memory_config is not None else input_tensor.memory_config()

    # Create appropriate program config based on input tensor if not provided
    if program_config is None:
        shard_spec = input_tensor.memory_config().shard_spec if input_tensor.is_sharded() else None
        program_config = ttnn.create_layernorm_program_config(shard_spec)

    operation_params = ttnn.LayerNormParams()
    operation_params.norm_type = norm_type
    operation_params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    operation_params.eps = epsilon
    operation_params.output_mem_config = output_mem_config
    operation_params.program_config = program_config
    operation_params.compute_kernel_config = compute_kernel_config

    # Create LayerNormInputs
    tensor_args = ttnn.LayerNormInputs()
    tensor_args.input = input_tensor
    if residual_input_tensor is not None:
        tensor_args.residual_input_tensor = residual_input_tensor
    if weight is not None:
        tensor_args.weight = weight
    if bias is not None:
        tensor_args.bias = bias

    # Create output tensor using the device operation's create_output_tensors
    output_tensor = ttnn.LayerNormDeviceOperation.create_output_tensors(operation_params, tensor_args)

    # Select the appropriate factory and create descriptor
    factory = ttnn.LayerNormDeviceOperation.select_program_factory(operation_params, tensor_args)
    program_descriptor = factory.create_descriptor(operation_params, tensor_args, output_tensor, core_range_set)

    # Build input and output tensors
    inputs, outputs = _build_layernorm_io_tensors(input_tensor, output_tensor, weight, bias, residual_input_tensor)

    return OpDescriptor(program_descriptor, inputs, outputs)
