# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Coverage for the `transform_in_place` convenience helper (eltwise_convenience.hpp).

`transform_in_place<Cb>(shape, ops...)` is the eltwise-chain replacement for
streaming_reduce_helpers' lambda-based transform_in_place. It forwards to
`eltwise_chain(shape, CopyTile<Cb>, ops..., PackTile<Cb>)`, i.e. an in-place SFPU
transform reading and writing ONE CB.

The kernel computes out = in * 2 + 1 in place, which exercises the surface that
plain `unary<Op, ...>` does not:
  - in-place CB (CopyTile and PackTile on the same buffer),
  - multiple SFPU ops in one chain (MulUnary then AddUnary), and
  - runtime-scalar op objects passed as constructed chain elements.
"""

import torch
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

TRANSFORM_IN_PLACE = "ttnn/cpp/ttnn/kernel_lib/tests/axes/transform_in_place.cpp"


def test_transform_in_place(device):
    """out = in * 2 + 1, applied in place via transform_in_place<scratch>(n, MulUnary{2}, AddUnary{1})."""
    n = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    t_in, tt_in = lib.make_input(shape, dt, device, seed=1701)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)

    # cb_in (streamed in, 2 pages), cb_scratch (whole window, n pages — holds the
    # full intermediate so the in-place transform has no concurrent producer),
    # cb_out (streamed out, 2 pages).
    cbs = [
        lib.cb_descriptor(0, dt, 2, core_grid),
        lib.cb_descriptor(1, dt, n, core_grid),
        lib.cb_descriptor(16, dt, 2, core_grid),
    ]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(TRANSFORM_IN_PLACE, [n], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)

    golden = t_in.to(torch.float32) * 2.0 + 1.0
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"transform_in_place out = in*2+1 | {msg}")
    assert pcc_ok, msg
