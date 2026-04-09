# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Internal utilities for normalization descriptors.
"""

from typing import Optional

import ttnn

from models.experimental.ops.descriptors.op_descriptor import (
    OpDescriptor,
    LazyOutputList,
    core_range_set_fusion_key,
    extend_branch_program_cache_key,
    make_partial_descriptor,
)


def _create_layernorm_op_descriptor(
    input_tensor: "ttnn.Tensor" = None,
    compute_kernel_config: Optional["ttnn.DeviceComputeKernelConfig"] = None,
    norm_type: "ttnn.LayerNormType" = None,
    weight: Optional["ttnn.Tensor"] = None,
    bias: Optional["ttnn.Tensor"] = None,
    residual_input_tensor: Optional["ttnn.Tensor"] = None,
    memory_config: Optional["ttnn.MemoryConfig"] = None,
    core_range_set: Optional["ttnn.CoreRangeSet"] = None,
    epsilon: float = 1e-12,
    program_config: Optional["ttnn.LayerNormProgramConfig"] = None,
) -> "OpDescriptor":
    """Create a normalization branch descriptor with deferred ``ProgramDescriptor``.

    ``input_tensor`` may be omitted for **persistent mode** — call
    :meth:`~OpDescriptor.update` with the activation before the first
    :meth:`~Sequential.run` / :meth:`~Parallel.run`.

    ``program_cache_key`` is based on :meth:`ttnn.LayerNormDeviceOperation.compute_program_hash`
    (device program cache). Computed without running the program factory. The factory runs
    only on ``.descriptor`` access (e.g. fusion cache miss).
    """
    op_name = "rms_norm" if norm_type == ttnn.LayerNormType.RMSNORM else "layer_norm"

    # ── Build input_names (always — needed for update()) ──
    input_names = [("input_tensor", 0)]
    inputs = [input_tensor]
    idx = 1
    if residual_input_tensor is not None:
        input_names.append(("residual_input_tensor", idx))
        inputs.append(residual_input_tensor)
        idx += 1
    if weight is not None:
        input_names.append(("weight", idx))
        inputs.append(weight)
        idx += 1
    if bias is not None:
        input_names.append(("bias", idx))
        inputs.append(bias)
        idx += 1

    # ── Partial path: input_tensor not yet available ──
    if input_tensor is None:
        return make_partial_descriptor(
            _create_layernorm_op_descriptor,
            dict(
                input_tensor=None,
                compute_kernel_config=compute_kernel_config,
                norm_type=norm_type,
                weight=weight,
                bias=bias,
                residual_input_tensor=residual_input_tensor,
                memory_config=memory_config,
                core_range_set=core_range_set,
                epsilon=epsilon,
                program_config=program_config,
            ),
            inputs,
            input_names,
            name=op_name,
        )

    # ── Full path (existing logic) ──
    device = input_tensor.device()

    if core_range_set is None:
        if input_tensor.is_sharded():
            core_range_set = input_tensor.memory_config().shard_spec.grid
        else:
            core_range_set = ttnn.LayerNormMultiCoreProgramFactory.default_core_range(device)

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
        recip_tensor = ttnn.create_layer_norm_reciprocals(device, core_range_set, W)

    # Build C++ params + tensor_args for the factory (captured by closure)
    operation_params = ttnn.LayerNormParams()
    operation_params.norm_type = norm_type
    operation_params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    operation_params.eps = epsilon
    operation_params.output_mem_config = output_mem_config
    operation_params.program_config = program_config
    operation_params.compute_kernel_config = compute_kernel_config

    tensor_args = ttnn.LayerNormInputs(input_tensor)
    if residual_input_tensor is not None:
        tensor_args.residual_input_tensor = residual_input_tensor
    if weight is not None:
        tensor_args.weight = weight
    if bias is not None:
        tensor_args.bias = bias
    if recip_tensor is not None:
        tensor_args.recip_tensor = recip_tensor
        input_names.append(("recip_tensor", idx))
        inputs.append(recip_tensor)

    h = ttnn.LayerNormDeviceOperation.compute_program_hash(operation_params, tensor_args)
    base_key = int(h) & ((1 << 64) - 1)
    if input_tensor.is_sharded():
        program_cache_key = base_key
    else:
        program_cache_key = extend_branch_program_cache_key(base_key, core_range_set_fusion_key(core_range_set))

    def _alloc_outputs(slots):
        slots[0] = ttnn.LayerNormDeviceOperation.create_output_tensors(operation_params, tensor_args)

    outputs = LazyOutputList([None], _alloc_outputs)
    cr_arg = None if input_tensor.is_sharded() else core_range_set

    def _run_factory():
        out = outputs[0]
        factory = ttnn.LayerNormDeviceOperation.select_program_factory(operation_params, tensor_args)
        return factory.create_descriptor(operation_params, tensor_args, out, cr_arg)

    return OpDescriptor(
        factory_fn=_run_factory,
        input_tensors=inputs,
        output_tensors=outputs,
        name=op_name,
        program_cache_key=program_cache_key,
        input_names=tuple(input_names),
    )
