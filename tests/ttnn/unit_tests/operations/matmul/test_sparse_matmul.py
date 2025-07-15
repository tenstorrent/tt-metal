# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import itertools
import os

from loguru import logger
import pytest
import random
import torch
import math
import ttnn

from models.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("mkn", [(128, 7168, 2048), (128, 2048, 7168)])
@pytest.mark.parametrize("num_experts", [2])
@pytest.mark.parametrize("num_tokens", [(2, 1)])  # , (64, 128), (64, 256)])
@pytest.mark.parametrize("tile_h", [32])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b])
def test_sparse_matmul(device, mkn, num_experts, num_tokens, tile_h, tile_w, in1_dtype):
    torch.manual_seed(0)

    m, k, n = mkn
    b, s = num_tokens
    in0 = torch.randn((b, s, m, k), dtype=torch.bfloat16)
    in1 = torch.randn((1, num_experts, k, n), dtype=torch.bfloat16)

    sparsity_density = random.random()  # random number between 0 and 1
    sparsity_shape = (1, b, s, num_experts)
    sparsity = torch.rand(sparsity_shape)
    mask = torch.rand(sparsity_shape) < sparsity_density
    sparsity[mask] = 0.0
    sparsity = sparsity.to(dtype=torch.bfloat16)

    nnz = int((sparsity != 0).sum().item())

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
    output_t = ttnn.sparse_matmul(
        in0_t,
        in1_t,
        sparsity=sparsity_t,
        nnz=nnz,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_tile=output_tile,
    )

    output_tensor = ttnn.to_torch(output_t)
    pt_out = torch.matmul(in0, in1)
    expected_pcc = 1.0 if in1_dtype == ttnn.bfloat16 else 0.99

    pcc_passed, pcc_message = comp_pcc(pt_out, output_tensor, expected_pcc)
    assert pcc_passed, pcc_message
    logger.info(pcc_message)


@pytest.mark.parametrize("mkn", [(128, 7168, 2048), (128, 2048, 7168)])
@pytest.mark.parametrize("num_experts", [2])
@pytest.mark.parametrize("num_tokens", [(2, 1)])  # , (64, 128), (64, 256)])
@pytest.mark.parametrize("tile_h", [32])
@pytest.mark.parametrize("tile_w", [32])
@pytest.mark.parametrize("in1_dtype", [ttnn.bfloat8_b])
def test_dense_matmul(device, mkn, num_experts, num_tokens, tile_h, tile_w, in1_dtype):
    torch.manual_seed(0)

    m, k, n = mkn
    b, s = num_tokens

    output_tile = ttnn.Tile([tile_h, tile_w])

    for e in range(num_experts):
        in1 = torch.randn((k, n), dtype=torch.bfloat16)
        in1_t = ttnn.from_torch(
            in1,
            tile=ttnn.Tile((32, tile_w)),
            dtype=in1_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        for b, s in itertools.product(range(b), range(s)):
            in0 = torch.randn(m, k, dtype=torch.bfloat16)
            in0_t = ttnn.from_torch(
                in0,
                tile=ttnn.Tile((tile_h, 32)),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            output_t = ttnn.matmul(
                in0_t,
                in1_t,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_tile=output_tile,
            )

            output_tensor = ttnn.to_torch(output_t)
            pt_out = torch.matmul(in0, in1)
            expected_pcc = 1.0 if in1_dtype == ttnn.bfloat16 else 0.99

            pcc_passed, pcc_message = comp_pcc(pt_out, output_tensor, expected_pcc)
            assert pcc_passed, pcc_message
            logger.info(pcc_message)
