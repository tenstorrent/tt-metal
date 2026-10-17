# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""noc_async_read_barrier_with_trid must not return before the read has landed.

A transaction-id read followed directly by its barrier, as kernels write it. Each iteration stores a marker into
the destination word, issues a 4 B read of a DRAM word, waits on the barrier and checks the destination. If the
barrier returns early, the destination still holds the marker ("stale") and the read-response counter has not
moved ("early"). On Blackhole, while the barrier polled the outstanding counter right after the issue, this
happened in every iteration on BRISC (all three read forms) and on NCRISC with skip_ptr_update.

On Blackhole BRISC the test also runs a diagnostic control: the same pair with the unordered counter poll. It is
never a pass/fail condition, because whether that race fires depends on the card and the compiled code. If it
fires, the detector is shown to work on this card; if not, a warning says so.
"""

import os
import warnings

import pytest
import torch
import ttnn
from loguru import logger

KERNEL = "tests/ttnn/unit_tests/operations/debug/kernels/noc_read_barrier_with_trid/trid_read_barrier.cpp"

# The race depends on NIU counter timing, which the simulator does not model.
pytestmark = pytest.mark.skipif(
    os.getenv("TT_METAL_SIMULATOR") is not None, reason="NIU counter timing is a hardware property"
)

ITERATIONS = 100_000
CONTROL_ITERATIONS = 10_000
PATTERN = 0x5A5A5A5A
MAGIC, DONE = 0xC0DE7B1D, 0xD0E5
R_MAGIC, R_ITERATIONS, R_EARLY, R_STALE, R_TIMEOUTS, R_WRONG, R_NOC, R_DONE = range(8)
MODE_BARRIER, MODE_CONTROL = 0, 1
FORMS = {"default": 0, "skip_ptr_update": 1, "skip_cmdbuf_chk": 2}
# Data movement processor and NoC, as in the default reader/writer configs on Blackhole.
RISCS = {
    "brisc": (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
    "ncrisc": (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
}


def run_trid_read_barrier(device, risc, form, mode, iterations):
    """Runs the kernel on one core and returns its counts."""
    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    # Two 2 KB DRAM pages: page 0 is the source word (PATTERN), page 1 receives the result words.
    buffer = ttnn.from_torch(
        torch.full((1, 1, 2, 512), PATTERN, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # generic_op needs an input and an output tensor; the buffer is the output.
    dummy = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cbs = [
        ttnn.CBDescriptor(
            total_size=4096,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=4096)],
        )
    ]
    runtime_args = ttnn.RuntimeArgs()
    runtime_args[core.x][core.y] = [buffer.buffer_address()]
    processor, noc = RISCS[risc]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[mode, FORMS[form], iterations]
        + list(ttnn.TensorAccessorArgs(buffer).get_compile_time_args()),
        runtime_args=runtime_args,
        config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc),
    )
    output = ttnn.generic_op([dummy, buffer], ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs, semaphores=[]))

    words = (ttnn.to_torch(output).reshape(-1).to(torch.int64) & 0xFFFFFFFF).tolist()
    r = words[512:520]
    assert r[R_MAGIC] == MAGIC and r[R_DONE] == DONE, f"kernel did not finish: {[hex(w) for w in r]}"
    return {
        "iterations": r[R_ITERATIONS],
        "early": r[R_EARLY],
        "stale": r[R_STALE],
        "timeouts": r[R_TIMEOUTS],
        "wrong": r[R_WRONG],
        "noc": r[R_NOC],
    }


@pytest.mark.timeout(600)
@pytest.mark.parametrize("form", list(FORMS))
@pytest.mark.parametrize("risc", list(RISCS))
def test_noc_read_barrier_with_trid_waits_for_data(device, risc, form):
    control = None
    if ttnn.device.is_blackhole(device) and risc == "brisc":
        # Diagnostic only: polling the outstanding counter directly after the issue returned early on the
        # Blackhole BRISC that was measured. Whether it fires depends on the card and the compiled code, so it
        # never fails the test.
        control = run_trid_read_barrier(device, risc, form, MODE_CONTROL, CONTROL_ITERATIONS)
        if control["early"] > 0 and control["stale"] > 0:
            logger.info(
                f"{risc} {form}: the unordered counter poll returned early, so the detector works here: {control}"
            )
        else:
            warnings.warn(
                f"{risc} {form}: the unordered counter poll did not return early on this card ({control}); "
                "this run cannot show that the detector would catch an early barrier return here"
            )

    result = run_trid_read_barrier(device, risc, form, MODE_BARRIER, ITERATIONS)
    assert result["iterations"] == ITERATIONS and result["timeouts"] == 0 and result["wrong"] == 0, result
    assert result["early"] == 0 and result["stale"] == 0, (
        f"{risc} {form}: noc_async_read_barrier_with_trid returned before the read landed: {result} "
        f"(diagnostic control: {control})"
    )
