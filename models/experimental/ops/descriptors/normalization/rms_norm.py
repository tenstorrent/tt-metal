# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors.normalization._utils import _create_layernorm_op_descriptor


def rms_norm(
    input_tensor: "ttnn.Tensor" = None,
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
    Create an OpDescriptor for an RMS norm operation.

    ``input_tensor`` may be omitted for **persistent mode** — call
    :meth:`~OpDescriptor.update` with the activation before the first ``run()``.

    Args:
        input_tensor: The input tensor (must be on device).  Omit for persistent mode.
        core_range_set: The set of cores to run the operation on. Required for non-sharded inputs.
        epsilon: Small constant for numerical stability (default: 1e-12).
        weight: Optional weight (gamma) tensor for scaling.
        bias: Optional bias (beta) tensor for shifting.
        residual_input_tensor: Optional residual tensor to add before normalization.
        compute_kernel_config: Optional compute kernel configuration.
        memory_config: Optional output memory configuration. Defaults to input's memory config.
        program_config: Optional program configuration. If not provided, one will be auto-generated.

    Returns:
        OpDescriptor containing the program descriptor, input tensors, and output tensors.

    Example::

        # Inline mode:
        desc = rms_norm(input_tensor, weight=w, compute_kernel_config=cc, ...)
        desc.launch()

        # Persistent mode:
        desc = rms_norm(weight=w, compute_kernel_config=cc, ...)  # no activation
        desc.update(activation)  # bind activation before first run
    """
    if input_tensor is not None:
        device = input_tensor.device()
        arch = device.arch()

        if program_config is not None and program_config.use_welford:
            raise ValueError("Welford's algorithm is not supported for RMS norm")

        if compute_kernel_config is None:
            compute_kernel_config = ttnn.rmsnorm_default_compute_config(arch)

    return _create_layernorm_op_descriptor(
        input_tensor=input_tensor,
        compute_kernel_config=compute_kernel_config,
        norm_type=ttnn.LayerNormType.RMSNORM,
        weight=weight,
        bias=bias,
        residual_input_tensor=residual_input_tensor,
        memory_config=memory_config,
        core_range_set=core_range_set,
        epsilon=epsilon,
        program_config=program_config,
    )


__all__ = ["rms_norm"]
