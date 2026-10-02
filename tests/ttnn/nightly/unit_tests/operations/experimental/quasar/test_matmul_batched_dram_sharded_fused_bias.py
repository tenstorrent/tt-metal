# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fused-bias batched HEIGHT_SHARDED DRAM matmul hangs if bias is pushed per batch."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Program-config types are bound on the C++ submodule, not ttnn.experimental.quasar.
qsr = ttnn._ttnn.operations.experimental.quasar


@pytest.mark.parametrize(
    "n_tiles",
    [
        pytest.param(1, id="bias_one_tile"),
        pytest.param(None, id="bias_all_dram_banks"),
    ],
)
def test_quasar_batched_dram_sharded_matmul_fused_bias_multi_batch(device, n_tiles):
    """Descriptor path with FUSE_BIAS and batches_per_core > 1. Public linear post-processes on WH."""
    workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_banks = len(workers)
    tile = 32
    m = k = tile
    n = tile * (num_banks if n_tiles is None else n_tiles)
    batches_per_core = 2
    batch = batches_per_core * num_banks

    worker_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in workers]
    )
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    in0_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(worker_grid, (batches_per_core * m, k), ttnn.ShardOrientation.ROW_MAJOR),
    )
    in1_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, (batches_per_core * k, n), ttnn.ShardOrientation.ROW_MAJOR),
    )
    out_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(worker_grid, (batches_per_core * m, n), ttnn.ShardOrientation.ROW_MAJOR),
    )

    torch.manual_seed(12345)
    in0 = torch.rand([1, batch, m, k], dtype=torch.bfloat16)
    in1 = torch.rand([1, batch, k, n], dtype=torch.bfloat16)
    bias = torch.rand([1, 1, 1, n], dtype=torch.bfloat16)
    in0_t = ttnn.from_torch(in0, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=in0_mc)
    in1_t = ttnn.from_torch(in1, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=in1_mc)
    bias_t = ttnn.from_torch(
        bias, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    params = qsr.MatmulParams()
    params.program_config = qsr.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
        in0_block_w=k // tile,
        per_core_M=m // tile,
        per_core_N=n // tile,
        fused_activation=None,
    )
    params.output_mem_config = out_mc
    params.output_dtype = ttnn.bfloat16

    attributes = qsr.create_matmul_attributes(in0_t, in1_t, params, [])
    tensor_args = qsr.MatmulInputs()
    tensor_args.input_tensors = [in0_t, in1_t]
    tensor_args.optional_input_tensors = [bias_t]

    factory = qsr.matmul_select_program_factory(attributes, tensor_args)
    assert isinstance(
        factory, qsr.MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory
    ), f"Expected batched HS DRAM factory, got {type(factory)}"

    output = qsr.MatmulDeviceOperation.create_output_tensors(attributes, tensor_args)
    descriptor = qsr.MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory.create_descriptor(
        attributes, tensor_args, output
    )
    fused = {(name, val) for kernel in descriptor.kernels for name, val in kernel.defines}
    assert ("FUSE_BIAS", "1") in fused, f"Expected FUSE_BIAS=1, got {fused}"

    result = ttnn.generic_op([in0_t, in1_t, bias_t, output[0]], descriptor)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    ref = torch.matmul(in0, in1) + bias
    assert_with_pcc(ref, got, pcc=0.999)
