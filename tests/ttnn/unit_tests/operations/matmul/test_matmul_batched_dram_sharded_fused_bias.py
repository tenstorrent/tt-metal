# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fused [1, N] bias on batched HEIGHT_SHARDED DRAM matmul (batches_per_core > 1)."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "n_tiles",
    [
        pytest.param(1, id="bias_one_tile"),
        pytest.param(None, id="bias_all_dram_banks"),
    ],
)
# batches_per_core=1 is the control: the hoisted bias read must match the old single-push case.
@pytest.mark.parametrize("batches_per_core", [1, 2], ids=["one_batch_per_core", "multi_batch_per_core"])
def test_batched_dram_sharded_matmul_fused_bias_multi_batch(device, n_tiles, batches_per_core):
    """Compiles FUSE_BIAS via prim.invoke; ttnn.linear post-processes batched in1."""
    workers = device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    num_banks = len(workers)
    tile = 32
    m = k = tile
    n = tile * (num_banks if n_tiles is None else n_tiles)
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

    params = ttnn.MatmulParams()
    params.program_config = ttnn.MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig(
        in0_block_w=k // tile,
        per_core_M=m // tile,
        per_core_N=n // tile,
        fused_activation=None,
    )
    params.output_mem_config = out_mc
    params.output_dtype = ttnn.bfloat16
    attributes = ttnn.create_matmul_attributes(in0_t, in1_t, params, [])

    output = ttnn.MatmulDeviceOperation.invoke(in0_t, in1_t, bias_t, attributes)
    got = ttnn.to_torch(output[0])
    ref = torch.matmul(in0, in1) + bias
    assert_with_pcc(ref, got, pcc=0.999)
