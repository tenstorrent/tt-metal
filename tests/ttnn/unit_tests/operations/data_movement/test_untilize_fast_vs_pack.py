# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The fast untilize and the plain pack untilize give the same bytes on Blackhole, so the untilize helper may pick
either for a row (tt-metal#58736). Every bf16 bit pattern, special values included, and every bfp8_b mantissa code at
every shared exponent with normal bf16 values, untilized to bf16 at the widths the helper moves to the plain pack
untilize with a 16-bit DEST, and a few it keeps on the fast untilize.
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="The fast untilize is Blackhole only")

TILE = 32
CB_IN = 0
CB_OUT = 16

READER = """
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();
    const auto src = TensorAccessor(src_args, src_addr);
    const uint32_t tile_bytes = get_tile_size(cb_in);
    for (uint32_t t = 0; t < num_tiles; ++t) {
        cb_reserve_back(cb_in, 1);
        noc_async_read(src.get_noc_addr(t), get_write_ptr(cb_in), tile_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_in, 1);
    }
}
"""

WRITER = """
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t wait_pages = get_compile_time_arg_val(1);
    constexpr uint32_t num_rows = get_compile_time_arg_val(2);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(3);
    constexpr auto dst_args = TensorAccessorArgs<4>();
    const auto dst = TensorAccessor(dst_args, dst_addr);
    cb_wait_front(cb_out, wait_pages);
    const uint32_t l1_addr = get_read_ptr(cb_out);
    for (uint32_t r = 0; r < num_rows; ++r) {
        noc_async_write(l1_addr + r * row_bytes, dst.get_noc_addr(r), row_bytes);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, wait_pages);
}
"""

FAST_COMPUTE = """
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/experimental/fast_untilize.h"

void kernel_main() {
    constexpr uint32_t width = get_compile_time_arg_val(0);
    constexpr uint32_t num_rows = get_compile_time_arg_val(1);
    compute_kernel_hw_startup(0, 16);
    fast_untilize_init<width>(0, 16);
    for (uint32_t r = 0; r < num_rows; ++r) {
        cb_wait_front(0, width);
        cb_reserve_back(16, width);
        fast_untilize_block<width>(0, 16);
        cb_push_back(16, width);
        cb_pop_front(0, width);
    }
    fast_untilize_uninit<width>(16);
}
"""

# The untilize helper's pack untilize path: one block for rows of up to 8 tiles, else blocks of the largest divisor up
# to 8.
PACK_COMPUTE = """
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack_untilize.h"

void kernel_main() {
    constexpr uint32_t width = get_compile_time_arg_val(0);
    constexpr uint32_t num_rows = get_compile_time_arg_val(1);
    constexpr uint32_t block = get_compile_time_arg_val(2);
    compute_kernel_hw_startup(0, 16);
    pack_untilize_init<block, width>(0, 16);
    for (uint32_t r = 0; r < num_rows; ++r) {
        cb_reserve_back(16, width);
        for (uint32_t b = 0; b < width / block; ++b) {
            cb_wait_front(0, block);
            pack_untilize_block<block, width>(0, 1, 16, b);
            cb_pop_front(0, block);
        }
        cb_push_back(16, width);
    }
    pack_untilize_uninit(16);
}
"""


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _cb(cb_id, dtype, num_pages):
    page_size = ttnn.tile_size(dtype)
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=_core(),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_size)],
    )


def _untilize(device, values, in_dtype, compute_source, extra_args):
    num_rows, row_datums = values.shape
    width = row_datums // TILE
    num_tiles = (num_rows // TILE) * width
    tt_in = ttnn.from_torch(values, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out = ttnn.from_torch(
        torch.zeros(num_rows, row_datums, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    reader_rt = ttnn.RuntimeArgs()
    reader_rt[0][0] = [tt_in.buffer_address()]
    writer_rt = ttnn.RuntimeArgs()
    writer_rt[0][0] = [tt_out.buffer_address()]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=READER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_core(),
            compile_time_args=[CB_IN, num_tiles, *ttnn.TensorAccessorArgs(tt_in).get_compile_time_args()],
            runtime_args=reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=compute_source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_core(),
            compile_time_args=[width, num_rows // TILE, *extra_args],
            config=ttnn.ComputeConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=WRITER,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=_core(),
            compile_time_args=[
                CB_OUT,
                num_tiles,
                num_rows,
                row_datums * 2,
                *ttnn.TensorAccessorArgs(tt_out).get_compile_time_args(),
            ],
            runtime_args=writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        ),
    ]
    program = ttnn.ProgramDescriptor(
        kernels=kernels, semaphores=[], cbs=[_cb(CB_IN, in_dtype, num_tiles), _cb(CB_OUT, ttnn.bfloat16, num_tiles)]
    )
    ttnn.generic_op([tt_in, tt_out], program)
    return ttnn.to_torch(tt_out).view(torch.int16)


def _block(width):
    return max(b for b in range(1, 9) if width % b == 0)


def _compare(device, values, in_dtype):
    width = values.shape[1] // TILE
    fast = _untilize(device, values, in_dtype, FAST_COMPUTE, [])
    pack = _untilize(device, values, in_dtype, PACK_COMPUTE, [_block(width)])
    mismatched = (fast != pack).nonzero()
    assert mismatched.numel() == 0, f"{mismatched.shape[0]} datums differ, first {mismatched[:4]}"


def _bf16_patterns(width):
    tile_rows = -(-65536 // (width * TILE * TILE))
    bits = torch.arange(tile_rows * TILE * width * TILE, dtype=torch.int32) % 65536
    return bits.to(torch.int16).view(torch.bfloat16).reshape(tile_rows * TILE, width * TILE)


def _bfp8_codes(width):
    # One shared exponent per 16-datum group (a face row): the first datum pins the exponent with the largest mantissa,
    # the other 15 take the signed 7-bit mantissa codes in turn, at every exponent whose smallest step is a normal bf16.
    groups = []
    for exponent in range(-120, 128):
        step = 2.0 ** (exponent - 6)
        signed = [(-1.0 if c >= 128 else 1.0) * (c % 128) * step for c in range(256)]
        for i in range(0, 256, 15):
            chunk = signed[i : i + 15]
            groups.append([127 * step] + chunk + [0.0] * (15 - len(chunk)))
    codes = torch.tensor(groups, dtype=torch.float32).flatten()
    tile_rows = -(-codes.numel() // (width * TILE * TILE))
    values = torch.zeros(tile_rows * TILE * width * TILE)
    values[: codes.numel()] = codes
    return values.reshape(tile_rows * TILE, width * TILE)


@pytest.mark.parametrize("width", [5, 6, 7, 8, 16])
def test_fast_and_pack_untilize_match_bf16(device, width):
    _compare(device, _bf16_patterns(width), ttnn.bfloat16)


@pytest.mark.parametrize("width", [5, 6, 7, 8, 9, 12, 16, 24])
def test_fast_and_pack_untilize_match_bfp8(device, width):
    _compare(device, _bfp8_codes(width), ttnn.bfloat8_b)
