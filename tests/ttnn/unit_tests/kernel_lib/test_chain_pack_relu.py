# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib
from tests.ttnn.utils_for_testing import comp_pcc


KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/pack_relu/pack_relu.cpp"


def test_pack_relu(device):
    n = 8
    dtype = ttnn.bfloat16
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input([1, 1, 32, 32 * n], dtype, device, seed=1801, scale=2.0, bias=-1.0)
    tt_relu = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32 * n]), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    tt_linear = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32 * n]), dtype, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_in], n, core_grid),
            lib.build_writer_2out_kernel([tt_relu, tt_linear], n, core_grid),
            lib.build_compute_kernel(KERNEL, [n], core_grid),
        ],
        semaphores=[],
        cbs=[
            lib.cb_descriptor(0, dtype, 2, core_grid),
            lib.cb_descriptor(16, dtype, 2, core_grid),
            lib.cb_descriptor(17, dtype, 2, core_grid),
        ],
    )
    ttnn.generic_op([tt_in, tt_relu, tt_linear], program)

    for output, golden in (
        (tt_relu, torch.clamp_min(torch_in.to(torch.float32), 0)),
        (tt_linear, torch_in.to(torch.float32)),
    ):
        pcc_ok, message = comp_pcc(golden, ttnn.to_torch(output).to(torch.float32), lib.pcc_threshold([dtype]))
        assert pcc_ok, message
