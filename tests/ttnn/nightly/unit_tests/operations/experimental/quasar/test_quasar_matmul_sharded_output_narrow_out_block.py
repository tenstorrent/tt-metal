# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""#58046 in the Quasar copy of the 1D and 2D multicast factories: a sharded output with out_block_w < per_core_N
(allowed when out_block_h == 1) must not share the output shard with the partials when a core computes more than
one output block."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

# Every test in this module takes the function-scoped `device` fixture without
# parametrizing device_params, so the device is opened once per module rather than
# once per test.
pytestmark = pytest.mark.use_module_device

qsr = ttnn._ttnn.operations.experimental.quasar


@pytest.mark.parametrize("mcast_in0", [True, False], ids=["in0_mcast", "in1_mcast"])
@pytest.mark.parametrize("out_block_w", [4, 2, 1])
def test_quasar_matmul_1d_sharded_output_narrow_out_block(device, out_block_w, mcast_in0):
    torch.manual_seed(0)
    grid = device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    per_core_N = 4
    if mcast_in0:  # one row of tiles, N split across the cores
        M, K, N = 32, 1024, 32 * per_core_N * num_cores
        memory_config = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
    else:  # one row of tiles per core, the whole N on every core
        M, K, N = 32 * num_cores, 1024, 32 * per_core_N
        memory_config = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
    torch_input_a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    torch_input_b = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    torch_output = torch_input_a.float() @ torch_input_b.float()
    ttnn_input_a = ttnn.from_torch(torch_input_a, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input_b = ttnn.from_torch(torch_input_b, layout=ttnn.TILE_LAYOUT, device=device)
    program_config = qsr.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=out_block_w,
        per_core_M=1,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=mcast_in0,
    )
    output = ttnn.experimental.quasar.matmul(
        ttnn_input_a, ttnn_input_b, program_config=program_config, memory_config=memory_config
    )
    assert_with_pcc(torch_output, ttnn.to_torch(output).float(), 0.999)


@pytest.mark.parametrize("out_block_w", [4, 2, 1])
def test_quasar_matmul_2d_sharded_output_narrow_out_block(device, out_block_w):
    torch.manual_seed(0)
    grid_x, grid_y = 4, 4
    per_core_M, per_core_N = 1, 4  # per_core_M == out_block_h: only the N split separates the blocks
    M, K, N = 32 * per_core_M * grid_y, 512, 32 * per_core_N * grid_x
    torch_input_a = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    torch_input_b = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    torch_output = torch_input_a.float() @ torch_input_b.float()
    ttnn_input_a = ttnn.from_torch(torch_input_a, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input_b = ttnn.from_torch(torch_input_b, layout=ttnn.TILE_LAYOUT, device=device)
    program_config = qsr.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
        in0_block_w=2,
        out_subblock_h=1,
        out_subblock_w=1,
        out_block_h=1,
        out_block_w=out_block_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
    )
    output_memory_config = ttnn.create_sharded_memory_config(
        (M, N),
        core_grid=ttnn.CoreGrid(y=grid_y, x=grid_x),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    output = ttnn.experimental.quasar.matmul(
        ttnn_input_a, ttnn_input_b, program_config=program_config, memory_config=output_memory_config
    )
    assert_with_pcc(torch_output, ttnn.to_torch(output).float(), 0.999)
