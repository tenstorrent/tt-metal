# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors.normalization._utils import _create_layernorm_op_descriptor


@OpDescriptor.create(name="rms_norm")
def rms_norm(
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
    """Create an OpDescriptor for an RMS norm operation.

    ``input_tensor`` may be omitted for **persistent mode** — call
    :meth:`~OpDescriptor.update` with the activation before the first ``run()``.

    Example::

        # Inline:
        desc = rms_norm(tt_q, weight=qw, compute_kernel_config=cc, ...)

        # Persistent:
        desc = rms_norm(weight=qw, compute_kernel_config=cc, ...)
        desc.update(tt_q)
    """
    device = input_tensor.device()
    arch = device.arch()

    if program_config is not None and program_config.use_welford:
        raise ValueError("Welford's algorithm is not supported for RMS norm")

    if compute_kernel_config is None:
        compute_kernel_config = ttnn.rmsnorm_default_compute_config(arch)

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
        program_config,
    )


__all__ = ["rms_norm"]
