# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
PackTile lifecycle / CB-synchronization (output side). Run under --dev.

The output lifecycle decides whether the chain or the CALLER emits cb_reserve_back / cb_push_back.
A miscount hangs the writer (BRISC) or overwrites an unpushed tile. out_lifecycle.cpp does an
identity copy with a selectable OutputLifecycle and supplies the caller-side edge where needed.

Covers the 4 well-defined cells (Streaming, Bulk, ReserveAllPushPerTile, CallerManaged);
ReserveNonePushEnd / Chunked are skipped (ambiguous n>1 reserve-without-push semantics).
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/lifecycle/out_lifecycle.cpp"

OUT_LIFECYCLES = {
    0: "Streaming",
    1: "Bulk",
    2: "ReserveAllPushPerTile",
    3: "CallerManaged",
}


@pytest.mark.parametrize("life,name", list(OUT_LIFECYCLES.items()), ids=list(OUT_LIFECYCLES.values()))
def test_output_lifecycle(device, life, name):
    n = 8
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(shape, dt, device, seed=901)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    # Non-streaming output lifecycles reserve the whole window upfront -> size cb_out for n tiles.
    cbs = [lib.cb_descriptor(0, dt, 2, core_grid), lib.cb_descriptor(16, dt, n, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [n, life], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)  # hang here trips --dev triage

    golden = torch_in.to(torch.float32)
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"output lifecycle={name} | no-hang + {msg}")
    assert pcc_ok, f"output lifecycle {name}: {msg}"
