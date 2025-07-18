# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os
from typing import Optional, Tuple

from loguru import logger
import pytest
import random
import torch
import math
import ttnn

from models.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc, start_measuring_time, stop_measuring_time
from tests.sweep_framework.sweep_utils.utils import gen_pytest_parametrize_args
from tests.sweep_framework.sweep_utils.roofline_utils import get_run_return

TIMEOUT = 50

# Parameters provided to the test vector generator are defined here.
parameters = {
    "moe_traces": {
        "mkn": [(128, 7168, 2048), (128, 2048, 7168)],
        "num_experts": [8],
        "num_tokens": [(1, 1), (1, 4), (1, 32)],
        "tile_h": [32, 16],
        "tile_w": [32, 16],
        "in1_dtype": [ttnn.bfloat8_b],
    },
}


def run_sparse_matmul(device, mkn, num_experts, num_tokens, tile_h, tile_w, in1_dtype) -> list:
    m, k, n = mkn
    b, s = num_tokens
    in0 = torch.randn((b, s, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    sparsity_shape = (1, b, s, num_experts)
    sparsity = torch.rand(sparsity_shape)

    # Mark some as 0 to test the sparsity
    sparsity[(sparsity == 0)] = 0.1  # First make sure there are no zeros
    number_of_zeros = random.randint(0, sparsity.numel() - 1)
    zero_indices = torch.randperm(sparsity.numel())[:number_of_zeros]
    sparsity.view(-1)[zero_indices] = 0.0

    sparsity = sparsity.to(dtype=torch.float32)

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
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tile = ttnn.Tile([tile_h, tile_w])

    start_time = start_measuring_time()
    output_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        nnz=nnz,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
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
        return get_run_return(pt_out, ttnn_out, 0.999, [in0_t, in1_t, sparsity_t], e2e_perf)


@pytest.mark.parametrize(**gen_pytest_parametrize_args(parameters))
def test_sparse_matmul(device, mkn, num_experts, num_tokens, tile_h, tile_w, in1_dtype):
    (result, msg), e2e_perf = run_sparse_matmul(device, mkn, num_experts, num_tokens, tile_h, tile_w, in1_dtype)
    assert result, msg
    logger.info(f"e2e_perf: {e2e_perf}")


def run(
    mkn,
    num_experts,
    num_tokens,
    tile_h,
    tile_w,
    in1_dtype,
    *,
    device,
) -> list:
    return run_sparse_matmul(device, mkn, num_experts, num_tokens, tile_h, tile_w, in1_dtype)
