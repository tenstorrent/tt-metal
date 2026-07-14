# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Lifecycle & CB-synchronization (the hang suite). Run under --dev.

A lifecycle is the (WaitPolicy, PopPolicy) pair an input declares — whether the chain or the CALLER
emits cb_wait_front / cb_pop_front. A miscount deadlocks the device.

held_b.cpp computes out[i] = A[i] + B[0]: A streams, B is one held tile reused each iter on a
selectable lifecycle, with the kernel supplying whatever edge the chain doesn't. Each case asserts
BOTH no-hang (--dev timeout trips triage) AND correct values (a miscount reads a stale tile).
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/lifecycle/held_b.cpp"

# Selector -> lifecycle name (must match held_b.cpp).
LIFECYCLES = {
    0: "Bulk",
    1: "HeldBulk",
    2: "HeldStream",
    3: "CallerManaged",
    4: "DeferredPop",
}


@pytest.mark.parametrize("life,name", list(LIFECYCLES.items()), ids=list(LIFECYCLES.values()))
def test_held_b_lifecycle(device, life, name):
    n = 8
    dt = ttnn.bfloat16
    a_shape = [1, 1, 32, 32 * n]
    b_shape = [1, 1, 32, 32]  # single held tile
    core_grid = lib.single_core_grid()

    torch_a, tt_a = lib.make_input(a_shape, dt, device, seed=701)
    torch_b, tt_b = lib.make_input(b_shape, dt, device, seed=702)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(a_shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [
        lib.cb_descriptor(0, dt, 2, core_grid),
        lib.cb_descriptor(1, dt, 2, core_grid),
        lib.cb_descriptor(16, dt, 2, core_grid),
    ]
    reader = lib.build_reader_asym_kernel([tt_a, tt_b], [n, 1], core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [n, life], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_out], program)  # a hang here trips --dev triage

    golden = torch_a.to(torch.float32) + torch_b.to(torch.float32).repeat(1, 1, 1, n)
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"lifecycle={name} | no-hang + {msg}")
    assert pcc_ok, f"lifecycle {name}: {msg}"
