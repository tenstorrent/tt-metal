# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""An Fp8_e4m3 pack untilize in one-tile blocks of a wider row is refused when the kernel is built (tt-metal#59140).

Each row of a one-tile block is its own 32-datum L1 stream, and an Fp8_e4m3 stream reaches L1 in 64-byte units, so every
row would be followed by 32 bytes of zeros over the next block's data. The compute API widths are the ones the DeepSeek
prefill dispatch and combine kernels split into one-tile blocks (no divisor of the row between 2 and 8); the helper
widths are the ones the untilize helper splits into one-tile blocks with a 32-bit DEST (no divisor between 2 and 4).
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole

CB_IN = 0
CB_OUT = 16

# The init sequence of the DeepSeek prefill dispatch and combine compute kernels.
API_KERNEL_SOURCE = """
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t block_ct_dim = get_compile_time_arg_val(2);
    constexpr uint32_t full_ct_dim = get_compile_time_arg_val(3);
    compute_kernel_hw_startup(cb_in, cb_out);
    pack_untilize_init<block_ct_dim, full_ct_dim>(cb_in, cb_out);
    pack_untilize_uninit(cb_out);
}
"""

HELPER_KERNEL_SOURCE = """
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t width = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(cb_in, cb_out);
    compute_kernel_lib::untilize_init<width, cb_in, cb_out>();
    compute_kernel_lib::untilize_uninit<width, cb_in, cb_out>();
}
"""


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _cb(cb_id, dtype, num_tiles):
    page_size = ttnn.tile_size(dtype)
    return ttnn.CBDescriptor(
        total_size=page_size * num_tiles,
        core_ranges=_core(),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_size)],
    )


def _run_init(device, kernel_source, extra_args, out_dtype, in_tiles, out_tiles):
    # generic_op takes its inputs and output as tensors; the kernel reads and writes none of them.
    tensors = [
        ttnn.from_torch(
            torch.zeros(32, 32, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        for _ in range(2)
    ]
    kernel = ttnn.KernelDescriptor(
        kernel_source=kernel_source,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=_core(),
        compile_time_args=[CB_IN, CB_OUT, *extra_args],
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
    )
    program = ttnn.ProgramDescriptor(
        kernels=[kernel],
        semaphores=[],
        cbs=[_cb(CB_IN, ttnn.bfloat16, in_tiles), _cb(CB_OUT, out_dtype, out_tiles)],
    )
    ttnn.generic_op(tensors, program)
    ttnn.synchronize_device(device)


def _run_api(device, out_dtype, block_ct_dim, full_ct_dim):
    _run_init(device, API_KERNEL_SOURCE, [block_ct_dim, full_ct_dim], out_dtype, block_ct_dim, full_ct_dim)


def _run_helper(device, out_dtype, width):
    _run_init(device, HELPER_KERNEL_SOURCE, [width], out_dtype, width, width)


@pytest.mark.skipif(not is_blackhole(), reason="Fp8_e4m3 pack untilize is Blackhole only")
@pytest.mark.parametrize("full_ct_dim", [11, 13, 17])
def test_fp8_one_tile_blocks_refused(device, expect_error, full_ct_dim):
    with expect_error(RuntimeError, "one-tile blocks of a wider row"):
        _run_api(device, ttnn.fp8_e4m3, 1, full_ct_dim)


@pytest.mark.skipif(not is_blackhole(), reason="Fp8_e4m3 pack untilize is Blackhole only")
@pytest.mark.parametrize(
    "out_dtype, block_ct_dim, full_ct_dim",
    [
        (ttnn.bfloat16, 1, 11),  # other formats keep one-tile blocks
        (ttnn.fp8_e4m3, 2, 12),  # blocks of two or more tiles
        (ttnn.fp8_e4m3, 3, 9),  # odd blocks of three or more tiles, fixed by the first-tile stream
        (ttnn.fp8_e4m3, 1, 1),  # a one-tile row is one block
    ],
)
def test_fp8_untilize_blocks_allowed(device, out_dtype, block_ct_dim, full_ct_dim):
    _run_api(device, out_dtype, block_ct_dim, full_ct_dim)


@pytest.mark.skipif(not is_blackhole(), reason="Fp8_e4m3 pack untilize is Blackhole only")
@pytest.mark.parametrize("width", [5, 7, 11, 13])
def test_fp8_helper_one_tile_blocks_refused(device, expect_error, width):
    with expect_error(RuntimeError, "one-tile blocks of a wider row"):
        _run_helper(device, ttnn.fp8_e4m3, width)


@pytest.mark.skipif(not is_blackhole(), reason="Fp8_e4m3 pack untilize is Blackhole only")
@pytest.mark.parametrize(
    "out_dtype, width",
    [
        (ttnn.float32, 7),  # other formats keep one-tile blocks
        (ttnn.fp8_e4m3, 6),  # blocks of three tiles
        (ttnn.fp8_e4m3, 8),  # blocks of four tiles
        (ttnn.fp8_e4m3, 3),  # one block
    ],
)
def test_fp8_helper_blocks_allowed(device, out_dtype, width):
    _run_helper(device, out_dtype, width)
