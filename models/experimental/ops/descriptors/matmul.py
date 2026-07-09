# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Matmul operation descriptor.

Creates an OpDescriptor for matrix multiplication using any of the backend
matmul program factories.  The factory is selected automatically based on
the ``program_config`` type, matching the dispatch logic in
``MatmulDeviceOperation::select_program_factory``.

The descriptor sets ``allowed_worker_cores`` on the program config to
``core_range_set``, which is the single source of truth for core placement.
"""

from typing import Optional

import ttnn

from models.experimental.ops.descriptors.op_descriptor import (
    OpDescriptor,
    LazyOutputList,
    core_range_set_fusion_key,
    extend_branch_program_cache_key,
)


_UNSUPPORTED_FACTORY = getattr(ttnn, "MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory", None)


@OpDescriptor.create(name="matmul")
def matmul(
    input_a: "ttnn.Tensor",
    input_b: "ttnn.Tensor",
    *,
    core_range_set: "ttnn.CoreRangeSet",
    program_config: "ttnn.MatmulProgramConfig",
    compute_kernel_config: Optional["ttnn.DeviceComputeKernelConfig"] = None,
    output_mem_config: Optional["ttnn.MemoryConfig"] = None,
    output_dtype: Optional["ttnn.DataType"] = None,
    transpose_a: bool = False,
    transpose_b: bool = False,
) -> "OpDescriptor":
    """Create a matmul op descriptor.

    Args:
        input_a: First input tensor (on device, tiled).
        input_b: Second input tensor (on device, tiled).
        core_range_set: Core range set for this operation.  Will be written to
            ``program_config.allowed_worker_cores``.
        program_config: One of the matmul program config variants
            (e.g. MatmulMultiCoreReuseProgramConfig,
            MatmulMultiCoreReuseMultiCastProgramConfig, etc.).
        compute_kernel_config: Compute kernel configuration.
        output_mem_config: Output memory configuration.
        output_dtype: Output data type. Defaults to input_a dtype.
        transpose_a: Transpose first input.
        transpose_b: Transpose second input.

    Returns:
        OpDescriptor with matmul program descriptor and IO tensors.
    """
    device = input_a.device()

    # Detect gather_in0 early — the factory it selects (MeshWorkload) has no
    # Python binding, so matmul_select_program_factory would raise a TypeError.
    if getattr(program_config, "gather_in0", False):
        raise ValueError(
            "MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory (gather_in0) "
            "is not supported in the descriptor interface"
        )

    if hasattr(program_config, "allowed_worker_cores"):
        program_config.allowed_worker_cores = core_range_set

    # Build MatmulParams
    base_params = ttnn.MatmulParams()
    base_params.program_config = program_config
    base_params.transpose_a = transpose_a
    base_params.transpose_b = transpose_b
    if output_dtype is not None:
        base_params.output_dtype = output_dtype
    if output_mem_config is not None:
        base_params.output_mem_config = output_mem_config
    if compute_kernel_config is not None:
        base_params.compute_kernel_config = compute_kernel_config

    operation_params = ttnn.create_matmul_attributes(input_a, input_b, base_params, [])

    # Build MatmulInputs
    tensor_args = ttnn.MatmulInputs()
    tensor_args.input_tensors = [input_a, input_b]
    tensor_args.optional_input_tensors = [None]

    # Select the program factory based on program_config type
    factory = ttnn.matmul_select_program_factory(operation_params, tensor_args)
    if _UNSUPPORTED_FACTORY is not None and isinstance(factory, _UNSUPPORTED_FACTORY):
        raise ValueError(
            "MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory (gather_in0) "
            "is not supported in the descriptor interface"
        )

    # Compute program cache key
    h = ttnn.MatmulDeviceOperation.compute_program_hash(operation_params, tensor_args)
    program_cache_key = extend_branch_program_cache_key(h, core_range_set_fusion_key(core_range_set))

    # Build input dict
    inputs = {"input_a": input_a, "input_b": input_b}

    # Lazy output allocation
    def _alloc_outputs(slots):
        spec = ttnn.MatmulDeviceOperation.compute_output_specs(operation_params, tensor_args)[0]
        slots[0] = ttnn.allocate_tensor_on_device(spec, device)

    outputs = LazyOutputList([None], _alloc_outputs)

    def _run_factory():
        out = outputs[0]
        return factory.create_descriptor(operation_params, tensor_args, [out], core_range_set)

    return OpDescriptor(
        factory_fn=_run_factory,
        input_tensors=inputs,
        output_tensors=outputs,
        program_cache_key=program_cache_key,
    )
