# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Transaction-id write waits must complete before their buffers are reused."""

import os

import pytest
import torch
import ttnn
from loguru import logger

KERNEL = "tests/ttnn/unit_tests/operations/debug/kernels/noc_write_barrier_with_trid/trid_write_barrier.cpp"
ITERATIONS = 10_000
MAGIC, DONE = 0xC0DE7B1D, 0xD0E5
RISCS = {
    "brisc": (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.NOC_1),
    "ncrisc": (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.NOC_0),
}
pytestmark = pytest.mark.skipif(
    os.getenv("TT_METAL_SIMULATOR") is not None, reason="NIU counter timing is a hardware property"
)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("noc_mode", [ttnn.NOC_MODE.DM_DEDICATED_NOC, ttnn.NOC_MODE.DM_DYNAMIC_NOC])
@pytest.mark.parametrize("risc", list(RISCS))
@pytest.mark.parametrize("stateful", [False, True])
@pytest.mark.parametrize("wait_for_completion", [False, True], ids=["flushed", "barrier"])
@pytest.mark.parametrize("size,use_register_buffer", [(4, False), (2048, False), (4, True)])
def test_noc_write_barrier_with_trid_waits_for_data(
    device, risc, stateful, wait_for_completion, size, use_register_buffer, noc_mode
):
    if not ttnn.device.is_blackhole(device):
        pytest.skip("NIU counter-ordering test, validated on Blackhole only")
    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    # Page 0 receives writes; page 1 receives the kernel's result words.
    buffer = ttnn.from_torch(
        torch.zeros((1, 1, 2, 512), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    dummy = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cbs = [
        ttnn.CBDescriptor(
            total_size=8192,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=8192)],
        )
    ]
    runtime_args = ttnn.RuntimeArgs()
    runtime_args[core.x][core.y] = [buffer.buffer_address()]
    processor, noc = RISCS[risc]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[int(wait_for_completion), int(stateful), size, ITERATIONS, int(use_register_buffer)]
        + list(ttnn.TensorAccessorArgs(buffer).get_compile_time_args()),
        runtime_args=runtime_args,
        config=ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc, noc_mode=noc_mode),
    )
    output = ttnn.generic_op([dummy, buffer], ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs, semaphores=[]))
    words = (ttnn.to_torch(output).reshape(-1).to(torch.int64) & 0xFFFFFFFF).tolist()[512:520]
    magic, iterations, early, wrong, timeouts, observed_noc, done, _ = words
    result = dict(iterations=iterations, early=early, wrong=wrong, timeouts=timeouts, noc=observed_noc)
    logger.info(f"{risc=} {noc_mode=} {stateful=} {wait_for_completion=} {size=} {use_register_buffer=}: {result}")
    assert magic == MAGIC and done == DONE, f"kernel did not finish: {words}"
    assert iterations == ITERATIONS and timeouts == 0, result
    assert early == 0 and wrong == 0, result
