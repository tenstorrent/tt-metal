# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check ordinary payload/relay ordering independently of the multicast host ABI."""

import pytest
import torch
import ttnn


@pytest.mark.parametrize("noc", [0, 1])
@pytest.mark.parametrize("flush_between", [False, True], ids=["back-to-back", "sdpa-flush"])
def test_chain_write_ordering(device, noc, flush_between):
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 0))])
    output = ttnn.allocate_tensor_on_device(
        ttnn.Shape([3, 1, 32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    coords = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)) for x in range(3)]
    rt = ttnn.RuntimeArgs()
    for rank in range(3):
        prev, nxt = coords[max(0, rank - 1)], coords[min(2, rank + 1)]
        rt[rank][0] = [output.buffer_address(), rank, prev.x, prev.y, nxt.x, nxt.y]
    kernel = ttnn.KernelDescriptor(
        kernel_source="tests/ttnn/unit_tests/kernel_lib/kernels/chain_write_ordering.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cores,
        compile_time_args=[int(flush_between)] + list(ttnn.TensorAccessorArgs(output).get_compile_time_args()),
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor() if noc else ttnn.ReaderConfigDescriptor(),
    )
    cbs = [
        ttnn.CBDescriptor(
            total_size=size,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=2048)],
        )
        for i, size in enumerate([65536, 2048])
    ]
    semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=cores, initial_value=int(i == 2)) for i in range(3)]
    actual = ttnn.generic_op([output, output], ttnn.ProgramDescriptor(kernels=[kernel], cbs=cbs, semaphores=semaphores))
    assert torch.count_nonzero(ttnn.to_torch(actual).contiguous().view(torch.int32)) == 0
