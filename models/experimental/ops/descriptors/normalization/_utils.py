# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Internal utilities for normalization descriptors.
"""

from typing import Optional

import ttnn

from models.experimental.ops.descriptors._generic import expose
from models.experimental.ops.descriptors.op_descriptor import core_range_set_fusion_key


def _layernorm_extra_cache_key(**kw):
    cr_arg = kw.get("_core_range_arg")
    if cr_arg is not None:
        return (core_range_set_fusion_key(kw["_core_range_set"]),)
    return ()


def _layernorm_factory_override(fct, params, inputs, out, **extra_kw):
    return fct.create_descriptor(params, inputs, out, extra_kw.get("_core_range_arg"))


_layernorm_descriptor = expose(
    ttnn.LayerNormDeviceOperation,
    name="layernorm_inner",
    required_inputs=["input"],
    extra_cache_key_fn=_layernorm_extra_cache_key,
    factory_call_override=_layernorm_factory_override,
)


def _create_layernorm_op_descriptor(
    input_tensor: "ttnn.Tensor",
    compute_kernel_config: "ttnn.DeviceComputeKernelConfig",
    norm_type: "ttnn.LayerNormType",
    weight: Optional["ttnn.Tensor"] = None,
    bias: Optional["ttnn.Tensor"] = None,
    residual_input_tensor: Optional["ttnn.Tensor"] = None,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
    core_range_set: Optional["ttnn.CoreRangeSet"] = None,
    epsilon: float = 1e-12,
    program_config: Optional["ttnn.LayerNormProgramConfig"] = None,
) -> "OpDescriptor":
    """Create a normalization branch descriptor with deferred ``ProgramDescriptor``.

    Always receives real tensors — the ``@OpDescriptor.create`` decorator on
    the public wrapper (``rms_norm``, ``layer_norm``) handles persistent mode.
    """
    if core_range_set is None:
        if input_tensor.is_sharded():
            core_range_set = input_tensor.memory_config().shard_spec.grid
        else:
            core_range_set = ttnn.LayerNormMultiCoreProgramFactory.default_core_range(input_tensor.device())

    if compute_kernel_config is None:
        raise ValueError("compute_kernel_config is required")

    output_mem_config = memory_config if memory_config is not None else input_tensor.memory_config()

    if program_config is None:
        shard_spec = input_tensor.memory_config().shard_spec if input_tensor.is_sharded() else None
        program_config = ttnn.create_layernorm_program_config(shard_spec)

    recip_tensor = None
    if program_config.use_welford:
        if input_tensor.is_sharded():
            W = input_tensor.memory_config().shard_spec.shape[1]
        else:
            W = input_tensor.shape[-1]
        recip_tensor = ttnn.create_layer_norm_reciprocals(input_tensor.device(), core_range_set, W)

    cr_arg = None if input_tensor.is_sharded() else core_range_set

    desc = _layernorm_descriptor(
        input=input_tensor,
        weight=weight,
        bias=bias,
        residual_input_tensor=residual_input_tensor,
        recip_tensor=recip_tensor,
        norm_type=norm_type,
        distributed_norm_stage=ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED,
        eps=epsilon,
        output_mem_config=output_mem_config,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        _core_range_arg=cr_arg,
        _core_range_set=core_range_set,
    )

    # Remap C++ field name 'input' → 'input_tensor' so the outer
    # @OpDescriptor.create wrapper (rms_norm / layer_norm) can match
    # keyword update() calls using the Python parameter name.
    # Rebuild the dict to preserve insertion order (pop+insert would
    # move the new key to the end, breaking _gather_inputs ordering).
    if desc._input_names and "input" in desc._input_names:
        desc._input_names = {("input_tensor" if k == "input" else k): v for k, v in desc._input_names.items()}

    return desc
