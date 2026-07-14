# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
In-place (output side): which (InputLifecycle, OutputLifecycle) pairs let a chain read AND write the
SAME CB, overwriting a resident buffer with no second CB. Run under --dev.

In-place is CB-self-deadlock-prone: the packer's reserve can't succeed while the reader's tiles
still occupy the buffer. Per-iter order is wait->read->POP then RESERVE->pack->push, so it is safe
only when BOTH the input pops and the output reserves incrementally (per-tile/chunk); an
upfront-reserve output would deadlock. See inplace_chain.cpp for the rule.

inplace_chain.cpp runs exp(x) in place on cb_x under a selectable lifecycle pair. Only the safe pairs
are parametrized (all must PASS): each asserts no-hang (--dev timeout trips triage) AND correct
values. The deadlocking pairs are intentionally not run.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/lifecycle/inplace_chain.cpp"

# Selector -> (InputLifecycle + OutputLifecycle) pair (must match inplace_chain.cpp).
INPLACE_LIFECYCLES = {
    0: "BulkDrain+Streaming",  # front rotation
    1: "Chunked+Chunked",  # chunk lockstep
    2: "Streaming+Streaming",  # per-tile rotation
}


@pytest.mark.parametrize("life,name", list(INPLACE_LIFECYCLES.items()), ids=list(INPLACE_LIFECYCLES.values()))
def test_inplace_chain_lifecycle(device, life, name):
    n = 8
    blk = 4  # block_size for the Chunked case; unused (clamped to 1) by the Scalar rotations
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(shape, dt, device, seed=1301)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    # cb_x (c_1) is the in-place buffer: stage 0 fills it with all n tiles upfront (Bulk), so it must
    # hold n. cb_src/cb_out are read/written Bulk too -> all three are sized n.
    cbs = [
        lib.cb_descriptor(0, dt, n, core_grid),  # cb_src: reader fills, stage 0 drains (Bulk)
        lib.cb_descriptor(1, dt, n, core_grid),  # cb_x:   the in-place buffer
        lib.cb_descriptor(16, dt, n, core_grid),  # cb_out: stage B fills, writer drains
    ]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [n, life, blk], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)  # a hang here trips --dev triage

    golden = torch.exp(torch_in.to(torch.float32))
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"in-place lifecycle={name} | no-hang + {msg}")
    assert pcc_ok, f"in-place lifecycle {name}: {msg}"
