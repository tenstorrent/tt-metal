# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors.normalization._utils import _create_layernorm_op_descriptor


def rms_norm(
    input_tensor: "ttnn.Tensor",
    core_range_set: Optional["ttnn.CoreRangeSet"] = None,
    epsilon: float = 1e-12,
    weight: Optional["ttnn.Tensor"] = None,
    bias: Optional["ttnn.Tensor"] = None,
    residual_input_tensor: Optional["ttnn.Tensor"] = None,
    compute_kernel_config: Optional["ttnn.DeviceComputeKernelConfig"] = None,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
) -> OpDescriptor:
    """
    Create an OpDescriptor for an RMS norm operation.

    Args:
        input_tensor: The input tensor (must be on device).
        core_range_set: The set of cores to run the operation on. Required for non-sharded inputs.
        epsilon: Small constant for numerical stability (default: 1e-12).
        weight: Optional weight (gamma) tensor for scaling.
        bias: Optional bias (beta) tensor for shifting.
        residual_input_tensor: Optional residual tensor to add before normalization.
        compute_kernel_config: Optional compute kernel configuration.
        memory_config: Optional output memory configuration. Defaults to input's memory config.

    Returns:
        OpDescriptor containing the program descriptor, input tensors, and output tensors.

    Example:
        >>> rms_desc_1 = models.experimental.ops.descriptors.normalization.rms_norm(input1, weight=w1, cores=cores1)
        >>> rms_desc_2 = models.experimental.ops.descriptors.normalization.rms_norm(input2, weight=w2, cores=cores2)
        >>> out_1, out_2 = models.experimental.ops.descriptors.composite.launch([rms_desc_1, rms_desc_2])
    """
    device = input_tensor.device()
    arch = device.arch()

    # Initialize compute kernel config if not provided
    if compute_kernel_config is None:
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

    return _create_layernorm_op_descriptor(
        input_tensor,
        compute_kernel_config,
        ttnn.LayerNormType.RMSNORM,
        weight,
        bias,
        residual_input_tensor,
        memory_config,
        core_range_set,
        epsilon,
    )


__all__ = ["rms_norm"]
