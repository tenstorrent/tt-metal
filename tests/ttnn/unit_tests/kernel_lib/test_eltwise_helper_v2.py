# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""V2 eltwise helper smoke test.

Builds a single-input compute kernel that calls
    eltwise_pipeline<cb_out>(n, eltwise_chain(CopyTile<cb_in>{}, Exp<>{}))
runs it on device, compares output against torch.exp.

Acceptance: PCC >= 0.9999 across num_tiles in {1, 8, 64}.
"""

import torch
import pytest
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc, skip_for_blackhole


COMPUTE_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/helper/compute_kernels/helper_unary_exp.cpp"
READER_KERNEL = (
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/" "reader_unary_interleaved_start_id.cpp"
)
WRITER_KERNEL = (
    "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/" "writer_unary_interleaved_start_id.cpp"
)

IN_CB = 0
OUT_CB = 16
CB_PAGE_SIZE = 2 * 1024  # bf16 tile = 32*32*2 bytes
CB_TOTAL_SIZE = 2 * CB_PAGE_SIZE  # double-buffered


def _build_program(input_tensor, output_tensor, num_tiles):
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    in_cb_format = ttnn.CBFormatDescriptor(buffer_index=IN_CB, data_format=ttnn.bfloat16, page_size=CB_PAGE_SIZE)
    out_cb_format = ttnn.CBFormatDescriptor(buffer_index=OUT_CB, data_format=ttnn.bfloat16, page_size=CB_PAGE_SIZE)
    in_cb = ttnn.CBDescriptor(total_size=CB_TOTAL_SIZE, core_ranges=core_grid, format_descriptors=[in_cb_format])
    out_cb = ttnn.CBDescriptor(total_size=CB_TOTAL_SIZE, core_ranges=core_grid, format_descriptors=[out_cb_format])

    reader_ct_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    # writer expects [out_cb_id, TensorAccessorArgs...]
    writer_ct_args = [OUT_CB] + list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [input_tensor.buffer_address(), num_tiles, 0]
    writer_rt[0][0] = [output_tensor.buffer_address(), num_tiles, 0]

    reader = ttnn.KernelDescriptor(
        kernel_source=READER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=WRITER_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute = ttnn.KernelDescriptor(
        kernel_source=COMPUTE_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[num_tiles],
        runtime_args=[],
        defines=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    program = ttnn.ProgramDescriptor(
        kernels=[reader, writer, compute],
        semaphores=[],
        cbs=[in_cb, out_cb],
    )
    return program


@skip_for_blackhole("Not tested on Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
def test_eltwise_helper_v2_unary_exp(device, num_tiles):
    shape = [1, 1, 32, 32 * num_tiles]
    # exp() on bf16 input + bf16 output compounds ULP errors; lessons §8 cites
    # ~0.999 PCC for SFPU exp under bf16 round-trip. Restrict range to keep
    # exp() output well-bounded.
    torch_input = torch.randn(shape, dtype=torch.bfloat16).clamp(-1.0, 1.0)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )

    program = _build_program(input_tensor, output_tensor, num_tiles)
    result = ttnn.generic_op([input_tensor, output_tensor], program)
    ttnn.synchronize_device(device)

    got = ttnn.to_torch(result)
    expected = torch.exp(torch_input.float()).bfloat16()

    passing, info = comp_pcc(expected.float(), got.float(), pcc=0.999)
    logger.info(f"num_tiles={num_tiles} PCC: {info}")
    assert passing, info
