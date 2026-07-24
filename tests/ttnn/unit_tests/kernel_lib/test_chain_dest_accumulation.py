# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib
from tests.ttnn.utils_for_testing import comp_pcc


KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/dest_accumulation/dest_accumulation.cpp"


@pytest.mark.parametrize("block_size", [1, 2, 8])
@pytest.mark.parametrize("caller_managed", [False, True])
def test_dest_accumulation(device, block_size, caller_managed):
    n = 8
    num_outputs = 3
    total_input_tiles = n * num_outputs
    dtype = ttnn.bfloat16
    core_grid = lib.single_core_grid()

    torch_a, tt_a = lib.make_input([1, 1, 32, 32 * total_input_tiles], dtype, device, seed=1701)
    torch_b, tt_b = lib.make_input([1, 1, 32, 32 * total_input_tiles], dtype, device, seed=1702)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32 * num_outputs]),
        dtype,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_a, tt_b], total_input_tiles, core_grid),
            lib.build_writer_1out_kernel(tt_out, num_outputs, core_grid),
            lib.build_compute_kernel(KERNEL, [n, block_size, int(caller_managed), num_outputs], core_grid),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dtype, total_input_tiles, core_grid),
            lib.cb_descriptor(1, dtype, total_input_tiles, core_grid),
            lib.cb_descriptor(16, dtype, num_outputs, core_grid),
        ],
    )
    out = ttnn.to_torch(ttnn.generic_op([tt_a, tt_b, tt_out], program)).to(torch.float32)

    a_tiles = torch.stack(torch_a.to(torch.float32).split(32, dim=-1)).reshape(num_outputs, n, 1, 1, 32, 32)
    b_tiles = torch.stack(torch_b.to(torch.float32).split(32, dim=-1)).reshape(num_outputs, n, 1, 1, 32, 32)
    reduced = (a_tiles + b_tiles).sum(dim=1)
    golden = torch.cat([reduced[i] for i in range(num_outputs)], dim=-1)
    pcc_ok, message = comp_pcc(golden, out, lib.pcc_threshold([dtype]))
    logger.info(f"DEST accumulation block={block_size}, caller_managed={caller_managed} | {message}")
    assert pcc_ok, message
