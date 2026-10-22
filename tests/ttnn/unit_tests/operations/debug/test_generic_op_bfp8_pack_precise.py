# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import ttnn


def _copy_to_bfp8(device, input_tensor, bfp8_pack_precise, key_salt=None):
    core = ttnn.CoreCoord(0, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    output_tensor = ttnn.allocate_tensor_on_device(
        input_tensor.shape, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    num_tiles = input_tensor.logical_volume() // 1024
    in_cb, out_cb = 0, 16
    in_page = 32 * 32 * (4 if input_tensor.dtype == ttnn.float32 else 2)
    cbs = [
        ttnn.CBDescriptor(
            total_size=2 * page,
            core_ranges=cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)],
        )
        for index, dtype, page in ((in_cb, input_tensor.dtype, in_page), (out_cb, ttnn.bfloat8_b, 1088))
    ]
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[0][0] = [input_tensor.buffer_address(), num_tiles, 0]
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[0][0] = [output_tensor.buffer_address(), num_tiles, 0]
    program = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args(),
                runtime_args=reader_rt_args,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=[out_cb, *ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()],
                runtime_args=writer_rt_args,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source="tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy.cpp",
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=cores,
                compile_time_args=[num_tiles] + ([key_salt] if key_salt is not None else []),
                runtime_args=[],
                config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True, bfp8_pack_precise=bfp8_pack_precise),
            ),
        ],
        semaphores=[],
        cbs=cbs,
    )
    return ttnn.to_torch(ttnn.generic_op([input_tensor, output_tensor], program))


# bfp8_pack_precise changes the generated pack format, so it has to be part of the JIT kernel key: a precise
# and an approximate compute kernel built in the same process must not share a binary.
@pytest.mark.skipif(bool(os.getenv("TT_METAL_SIMULATOR")), reason="compares hardware packer rounding modes")
@pytest.mark.parametrize("first", [False, True], ids=["approximate_first", "precise_first"])
def test_bfp8_pack_precise_builds_its_own_kernel(device, first):
    torch.manual_seed(0)
    data = torch.randn((1, 1, 32, 32), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    results = {flag: _copy_to_bfp8(device, input_tensor, flag) for flag in (first, not first)}
    for flag in (False, True):
        expected = _copy_to_bfp8(device, input_tensor, flag, key_salt=int(flag))
        assert torch.equal(results[flag], expected), f"bfp8_pack_precise={flag} output differs from its own build"
