# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Regression: `LayerNormDeviceOperation.compute_program_hash` uses default `compute_program_hash`, which hashes
`tensor_args` including `Tensor` fields. `LayerNormInputs()` leaves `input` as a C++ default `Tensor` (null
`tensor_attributes`). `Tensor::to_hash` must handle that without dereferencing null.
"""

import ttnn


def test_layer_norm_compute_program_hash_default_constructed_input_tensor(device):
    """`LayerNormInputs()` leaves `input` as a C++ default `Tensor` (null `tensor_attributes`).

    Default `compute_program_hash` hashes `tensor_args`; without `Tensor::to_hash` handling that case, this crashes in
    reflective hashing.
    """
    arch = device.arch()
    operation_params = ttnn.LayerNormParams()
    operation_params.norm_type = ttnn.LayerNormType.RMSNORM
    operation_params.distributed_norm_stage = ttnn.DistributedLayerNormStage.NOT_DISTRIBUTED
    operation_params.eps = 1e-12
    operation_params.output_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.DRAM
    )
    operation_params.program_config = ttnn.create_layernorm_program_config(None)
    operation_params.compute_kernel_config = ttnn.rmsnorm_default_compute_config(arch)

    tensor_args = ttnn.LayerNormInputs()

    h = ttnn.LayerNormDeviceOperation.compute_program_hash(operation_params, tensor_args)
    assert isinstance(h, int)
