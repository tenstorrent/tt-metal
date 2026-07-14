# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
SetupOwner::Caller functional test for eltwise_chain.

SetupOwner::Caller means the caller emitted the chain's one-time setup (init + reconfig) itself
before the loop, so the chain emits none of it. Positive path: setup_owner_caller.cpp emits that
setup as raw LLK (copy_tile_init + exp_tile_init), then runs eltwise_chain<SetupOwner::Caller> over
N tiles; exp(x) comes out wrong if Caller skipped a needed init or the raw setup didn't match.
The two compile-time guards (must-be-boot-hoistable; no live reconfig knob) are covered by the
negative tests in test_chain_static_asserts.py.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/axes/setup_owner_caller.cpp"


@pytest.mark.parametrize("n", [4, 16])
def test_setup_owner_caller_exp(device, n):
    """Raw caller-owned setup + eltwise_chain<SetupOwner::Caller> must produce exp(x) over all N tiles."""
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(shape, dt, device, seed=777)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, core_grid), lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [n], core_grid)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    output = ttnn.generic_op([tt_in, tt_out], program)
    golden = torch.exp(torch_in.to(torch.float32))
    out = ttnn.to_torch(output).to(torch.float32)

    ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"SetupOwner::Caller exp n={n} | {msg}")
    assert ok, f"SetupOwner::Caller produced wrong exp(x) (caller-owned setup not reused correctly): {msg}"
