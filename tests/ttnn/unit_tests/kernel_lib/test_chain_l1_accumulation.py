# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib
from tests.ttnn.utils_for_testing import comp_pcc


KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/l1_accumulation/l1_accumulation.cpp"


@pytest.mark.parametrize("caller_managed", [False, True], ids=["managed", "caller-managed"])
def test_l1_accumulation(device, caller_managed):
    n = 8
    dtype = ttnn.bfloat16
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input([1, 1, 32, 32 * n], dtype, device, seed=1701, scale=0.125)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_in], n, core_grid),
            lib.build_writer_1out_kernel(tt_out, 1, core_grid),
            lib.build_compute_kernel(KERNEL, [n, int(caller_managed)], core_grid),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dtype, 2, core_grid),
            lib.cb_descriptor(15, dtype, 1, core_grid),
            lib.cb_descriptor(16, dtype, 2, core_grid),
        ],
    )
    out = ttnn.to_torch(ttnn.generic_op([tt_in, tt_out], program)).to(torch.float32)

    golden = torch_in.to(torch.float32).reshape(1, 1, 32, n, 32).sum(dim=3)
    pcc_ok, message = comp_pcc(golden, out, 0.999)
    logger.info(f"L1 accumulation caller_managed={caller_managed} | {message}")
    assert pcc_ok, message
