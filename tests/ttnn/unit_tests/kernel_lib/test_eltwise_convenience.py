# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib
from tests.ttnn.utils_for_testing import comp_pcc

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/api/sum_of_squares.cpp"


@pytest.mark.parametrize("block_size", [1, 2, 8])
def test_sum_of_squares_reduces_each_row(device, block_size):
    ht = 3
    wt = 8
    input_tiles = ht * wt
    dtype = ttnn.bfloat16
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input([1, 1, 32 * ht, 32 * wt], dtype, device, seed=1701, scale=0.125)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32 * ht]),
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_in], input_tiles, core_grid),
            lib.build_writer_1out_kernel(tt_out, ht, core_grid),
            lib.build_compute_kernel(KERNEL, [ht, wt, block_size], core_grid),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dtype, 2, core_grid),
            lib.cb_descriptor(16, dtype, ht, core_grid),
        ],
    )

    output = ttnn.to_torch(ttnn.generic_op([tt_in, tt_out], program)).to(torch.float32)
    input_tiles_torch = torch_in.to(torch.float32).reshape(1, 1, ht, 32, wt, 32).permute(0, 1, 2, 4, 3, 5)
    golden_tiles = input_tiles_torch.square().sum(dim=3)
    golden = golden_tiles.permute(0, 1, 3, 2, 4).reshape(1, 1, 32, 32 * ht)

    pcc_ok, message = comp_pcc(golden, output, 0.999)
    logger.info(f"sum_of_squares block_size={block_size} | {message}")
    assert pcc_ok, message
