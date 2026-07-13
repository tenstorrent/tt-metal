# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional tests for eltwise-chain packer L1 accumulation."""

import pytest
import torch
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/l1_accumulation/l1_accumulation.cpp"


@pytest.mark.parametrize("caller_managed", [False, True], ids=["reserve-one-push-one", "caller-managed"])
def test_l1_accumulation_output_lifecycle(device, caller_managed):
    n = 8
    dt = ttnn.bfloat16
    input_shape = [1, 1, 32, 32 * n]
    output_shape = [1, 1, 32, 32]
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(input_shape, dt, device, seed=1701, scale=0.125, bias=0.0)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    cbs = [
        lib.cb_descriptor(0, dt, 2, core_grid),
        lib.cb_descriptor(15, dt, 1, core_grid),
        lib.cb_descriptor(16, dt, 2, core_grid),
    ]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, 1, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [n, int(caller_managed)], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)

    golden = torch_in.to(torch.float32).reshape(1, 1, 32, n, 32).sum(dim=3)
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, 0.999)
    logger.info(f"L1 accumulation caller_managed={caller_managed} | {msg}")
    assert pcc_ok, msg
