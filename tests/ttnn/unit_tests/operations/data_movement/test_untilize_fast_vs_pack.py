# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The fast untilize and the plain pack untilize give the same bytes on Blackhole, so the untilize helper may pick
either for a row (tt-metal#58736). Every bf16 bit pattern, special values included, every bfp8_b tile byte pair (each of
the 256 shared exponents with each of the 256 sign and mantissa bytes, written as raw tile bytes) and every bfp4_b
exponent with every 4-bit code, untilized to bf16 with a 16-bit DEST: rows of one fast untilize chunk (5 to 8 tiles)
and of several (9 tiles and more, with tails of 2 to 7 tiles).
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


def _untilize(device, tt_in, in_dtype, num_rows, width, compute_source, extra_args):
    # tt_in holds the input tiles as consecutive pages, in row-major tile order.
    row_datums = width * TILE
    num_tiles = (num_rows // TILE) * width
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


def _compare(device, tt_in, in_dtype, num_rows, width):
    fast = _untilize(device, tt_in, in_dtype, num_rows, width, FAST_COMPUTE, [])
    pack = _untilize(device, tt_in, in_dtype, num_rows, width, PACK_COMPUTE, [_block(width)])
    mismatched = (fast != pack).nonzero()
    assert mismatched.numel() == 0, f"{mismatched.shape[0]} datums differ, first {mismatched[:4]}"


def _bf16_patterns(width):
    tile_rows = -(-65536 // (width * TILE * TILE))
    bits = torch.arange(tile_rows * TILE * width * TILE, dtype=torch.int32) % 65536
    return bits.to(torch.int16).view(torch.bfloat16).reshape(tile_rows * TILE, width * TILE)


def _bfp8_raw_tiles(width):
    # Raw Bfp8_b tiles: 64 shared exponent bytes (one per 16-datum face row), then 1024 sign and mantissa bytes. Face
    # row g of the input takes exponent g // 16 and the bytes (g % 16) * 16 to (g % 16) * 16 + 15, so the 4096 face rows
    # of 64 tiles carry every exponent with every sign and mantissa byte.
    tile_rows = -(-64 // width)
    num_tiles = tile_rows * width
    face_row = torch.arange(num_tiles * 64) % 4096
    exponents = (face_row // 16).to(torch.uint8).reshape(num_tiles, 64)
    datums = ((face_row % 16).unsqueeze(1) * 16 + torch.arange(16)).to(torch.uint8).reshape(num_tiles, 1024)
    return torch.cat([exponents, datums], dim=1), tile_rows * TILE


def _bfp4_raw_tiles(width):
    # Raw Bfp4_b tiles: 64 shared exponent bytes, then 512 bytes of two 4-bit sign and mantissa codes each. Every face
    # row holds all 16 codes, and face row g takes exponent g % 256, so 4 tiles carry every exponent with every code.
    tile_rows = -(-4 // width)
    num_tiles = tile_rows * width
    face_row = torch.arange(num_tiles * 64)
    exponents = (face_row % 256).to(torch.uint8).reshape(num_tiles, 64)
    codes = torch.arange(8) * 2
    datums = ((codes + 1) * 16 + codes).to(torch.uint8).repeat(num_tiles * 64).reshape(num_tiles, 512)
    return torch.cat([exponents, datums], dim=1), tile_rows * TILE


@pytest.mark.parametrize("width", [5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 24])
def test_fast_and_pack_untilize_match_bf16(device, width):
    values = _bf16_patterns(width)
    tt_in = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    _compare(device, tt_in, ttnn.bfloat16, values.shape[0], width)


@pytest.mark.parametrize("width", [5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 24])
def test_fast_and_pack_untilize_match_bfp8(device, width):
    raw, num_rows = _bfp8_raw_tiles(width)
    # One page per tile: a row-major uint8 tensor of the raw tile bytes, read into a Bfp8_b input CB.
    tt_in = ttnn.from_torch(raw, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _compare(device, tt_in, ttnn.bfloat8_b, num_rows, width)


@pytest.mark.parametrize("width", [5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 24])
def test_fast_and_pack_untilize_match_bfp4(device, width):
    raw, num_rows = _bfp4_raw_tiles(width)
    tt_in = ttnn.from_torch(raw, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    _compare(device, tt_in, ttnn.bfloat4_b, num_rows, width)
