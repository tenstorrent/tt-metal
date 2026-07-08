# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Init-hoisting equality for same data (eltwise_chain G5 / hoisting).

A uniform chain over same-format data hoists its per-element init OUT of the tile loop and emits
it once at boot (chain_hoist_math_mop_v && chain_hoist_sfpu_v -> hoist_compute_init,
eltwise_chain.inl:2010-2042). A non-uniform chain re-inits per tile. Hoisting is a pure emission
optimization: for the same op on same-format data it must produce a BIT-IDENTICAL result to the
per-tile-init path.

This compares two kernels computing exp(x) over N tiles:
  - hoist_single_call.cpp : one eltwise_chain(N, ...) -> init hoisted once.
  - hoist_per_tile.cpp    : N x eltwise_chain(1, ...) -> init emitted per tile.
The outputs must be exactly equal (and both match torch.exp). If hoisting ever skipped a needed
re-init, later tiles in the hoisted kernel would diverge from the per-tile baseline.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

HOIST = "ttnn/cpp/ttnn/kernel_lib/tests/axes/hoist_single_call.cpp"
PERTILE = "ttnn/cpp/ttnn/kernel_lib/tests/axes/hoist_per_tile.cpp"


def _run(device, kernel, n):
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=501)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, core_grid), lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(kernel, [n], core_grid)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)
    return torch_in.to(torch.float32), ttnn.to_torch(output).to(torch.float32)


@pytest.mark.parametrize("n", [4, 16])
def test_init_hoisting_equality(device, n):
    """The hoisted (init-once) and per-tile (init-per-tile) kernels must agree bit-for-bit."""
    torch_in, out_hoist = _run(device, HOIST, n)
    _, out_pertile = _run(device, PERTILE, n)

    # Both must match the math golden ...
    golden = torch.exp(torch_in)
    ok_h, msg_h = comp_pcc(golden, out_hoist, lib.pcc_threshold([ttnn.bfloat16]))
    ok_p, msg_p = comp_pcc(golden, out_pertile, lib.pcc_threshold([ttnn.bfloat16]))
    assert ok_h, f"hoisted vs golden: {msg_h}"
    assert ok_p, f"per-tile vs golden: {msg_p}"

    # ... and be BIT-IDENTICAL to each other: hoisting init must change nothing.
    max_diff = (out_hoist - out_pertile).abs().max().item()
    logger.info(f"init-hoisting equality n={n} | hoisted~golden={msg_h} | max|hoist-pertile|={max_diff}")
    assert torch.equal(
        out_hoist, out_pertile
    ), f"init-hoisting changed the result: max abs diff {max_diff} between hoisted and per-tile init."
