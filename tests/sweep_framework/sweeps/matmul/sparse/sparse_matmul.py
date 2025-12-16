# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools

from loguru import logger
import pytest
import random
import torch
import math
import ttnn

from tests.ttnn.utils_for_testing import start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.utils import gen_pytest_parametrize_args
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return

TIMEOUT = 50

# Parameters provided to the test vector generator are defined here.
parameters = {
    "moe_traces1": {
        "mkn": [(128, 7168, 2048)],
        "num_experts": [8],
        "num_batches": [(1, 1), (1, 4), (1, 32)],
        "tile_h": [32],
        "tile_w": [32],
        "in1_dtype": [ttnn.bfloat8_b],
        "core_grid": [(8, 8)],
    },
    "moe_traces2": {
        "mkn": [(128, 2048, 7168)],
        "num_experts": [8],
        "num_batches": [(1, 1), (1, 4), (1, 32)],
        "tile_h": [32],
        "tile_w": [32],
        "in1_dtype": [ttnn.bfloat8_b],
        "core_grid": [(7, 8)],
    },
    "gpt_traces": {
        "mkn": [(32, 2880, 360)],
        "num_experts": [32, 128],
        "num_batches": [(1, 1 << x) for x in range(0, 13)],
        "tile_h": [32],
        "tile_w": [32],
        "in1_dtype": [ttnn.bfloat4_b],
        "core_grid": [(3, 4)],
    },
}


def run_sparse_matmul(device, mkn, num_experts, num_batches, tile_h, tile_w, in1_dtype, core_grid) -> list:
    m, k, n = mkn
    b, s = num_batches
    in0 = torch.randn((b, s, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    sparsity_shape = (1, b, s, num_experts)
    sparsity = torch.rand(sparsity_shape)

    # Mark some as 0 to test the sparsity
    sparsity[(sparsity == 0)] = 0.1  # First make sure there are no zeros
    number_of_zeros = random.randint(0, sparsity.numel() - 1)
    zero_indices = torch.randperm(sparsity.numel())[:number_of_zeros]
    sparsity.view(-1)[zero_indices] = 0.0

    sparsity = sparsity.to(dtype=torch.bfloat16)

    nnz = int((sparsity != 0).sum().item())
    logger.info(f"nnz: {nnz}")

    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, 32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((32, tile_w)),
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    sparsity_t = ttnn.from_torch(
        sparsity,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tile = ttnn.Tile([tile_h, tile_w])
    core_x, core_y = core_grid

    start_time = start_measuring_time()
    output_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        nnz=nnz,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(core_x, core_y),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=m // 32,
            per_core_N=int(math.ceil(n / 32)) // (core_x * core_y),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        ),
    )
    output_tensor = ttnn.to_torch(output_t)
    e2e_perf = stop_measuring_time(start_time)

    # Compute matmul using torch for each batch and concatenate the results
    for b_idx, s_idx, e_idx in itertools.product(range(b), range(s), range(num_experts)):
        if sparsity[0, b_idx, s_idx, e_idx] == 0.0:
            continue
        in0_batch = in0[b_idx, s_idx, :, :]
        in1_batch = in1[0, e_idx, :, :]
        pt_out = torch.matmul(in0_batch, in1_batch)
        ttnn_out = output_tensor[b_idx, s_idx, 0, e_idx, :, :]
        # We only check the first non-zero entry for PCC
        return get_run_return(pt_out, ttnn_out, 0.99, [in0_t, in1_t, sparsity_t], e2e_perf)


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_sparse_matmul(device, mkn, num_experts, num_batches, tile_h, tile_w, in1_dtype, core_grid):
    (result, msg), e2e_perf = run_sparse_matmul(
        device, mkn, num_experts, num_batches, tile_h, tile_w, in1_dtype, core_grid
    )
    assert result, msg
    logger.info(f"e2e_perf: {e2e_perf}")


def run(
    mkn,
    num_experts,
    num_batches,
    tile_h,
    tile_w,
    in1_dtype,
    core_grid,
    *,
    device,
) -> list:
    return run_sparse_matmul(device, mkn, num_experts, num_batches, tile_h, tile_w, in1_dtype, core_grid)
