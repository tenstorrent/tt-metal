// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row scatter of the DeepSeek-V4.1 decode pool (tt/v41/pool.py scatter_rows): for i in [row_start, row_start +
// row_count), copy row src_base + i of ``src`` to row dst_base + i * stride + idx[i] of ``dst`` unless idx[i] is
// SENTINEL. ``src`` is row-major, DRAM-interleaved, one page per row; ``idx`` is one uint32 row-major page. The
// destination row comes from a device tensor, so a captured trace writes wherever the step's inputs say (the ring
// slot, a user's compressed row), while the per-tensor bases are the capture's constants. Data movement only.
//
// The destination row index is always a logical row of ``dst``; DST_MODE says how that row is stored:
//   0 row-major, one page per row of ROW_BYTES (as ``src``): one write of ROW_BYTES.
//   1 tiled bf16 (32 x 32 tiles of four 16 x 16 faces, 2 KB pages in row-major tile order): logical row r is
//      in-tile row ri = r % 32 of the tiles of tile row r / 32, TILES_PER_ROW = ROW_BYTES / 64 tiles across;
//      in tile column c the row's 32 columns are the 16 of face (ri / 16) * 2 (bytes [c * 64, c * 64 + 32) of
//      the source row) and the 16 of the face after it; each face is 512 B, row-major, 32 B per row
//      (tt_metal tilize_utils: face (fy, fx) at element fy * 512 + fx * 256 of the tile). Eight 32-byte writes;
//      Blackhole DRAM writes need 16-byte alignment (NOC_DRAM_WRITE_ALIGNMENT_BYTES), reads 64.
//   2 block pages: ROWS_PER_PAGE consecutive logical rows per page (page r / ROWS_PER_PAGE, byte offset
//      (r % ROWS_PER_PAGE) * ROW_BYTES), the layout ``candidate_scores`` reads a block from: one write.
//
// SRC_MODE says how ``src`` stores its rows: 0 one page per row (as before); 2 block pages of ROWS_PER_PAGE rows
// (the restore of a parked prefix: the FP8 rows travel over the SP axis in 64-byte-aligned
// pages of eight rows, and come out of them here). A block-page source is read a whole page at a time (a DRAM
// read must start 64-byte aligned, and a row inside a page is not), and the rows are written from that L1 page.
//
// compile_time_args = [cb_scratch, ROW_BYTES, DST_MODE, ROWS_PER_PAGE, SRC_MODE, TensorAccessorArgs(src)...,
//                      TensorAccessorArgs(dst)..., TensorAccessorArgs(idx)...]
// runtime args = [src_addr, dst_addr, idx_addr, src_base, dst_base, stride, row_start, row_count, rows]
// (``rows``: the length of idx. Each core reads its own window of the indices, from the 64-byte line holding
// row_start, so an index of any length fits: a DRAM read must start at an aligned address)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t src_base = get_arg_val<uint32_t>(3);
    const uint32_t dst_base = get_arg_val<uint32_t>(4);
    const uint32_t stride = get_arg_val<uint32_t>(5);
    const uint32_t row_start = get_arg_val<uint32_t>(6);
    const uint32_t row_count = get_arg_val<uint32_t>(7);
    const uint32_t rows = get_arg_val<uint32_t>(8);

    constexpr uint32_t cb_scratch = get_compile_time_arg_val(0);
    constexpr uint32_t ROW_BYTES = get_compile_time_arg_val(1);
    constexpr uint32_t DST_MODE = get_compile_time_arg_val(2);
    constexpr uint32_t ROWS_PER_PAGE = get_compile_time_arg_val(3);
    constexpr uint32_t SRC_MODE = get_compile_time_arg_val(4);
    constexpr auto src_args = TensorAccessorArgs<5>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<dst_args.next_compile_time_args_offset()>();
    constexpr uint32_t SENTINEL = 0xFFFFFFFFu;
    constexpr uint32_t ROW_L1 = (ROW_BYTES + 63) & ~63u;
    constexpr uint32_t TILES_PER_ROW = ROW_BYTES / 64;  // bf16: 32 columns of 2 bytes per tile
    constexpr uint32_t FACE_BYTES = 512;
    constexpr uint32_t FACE_ROW_BYTES = 32;
    constexpr uint32_t PAGE_BYTES = ROWS_PER_PAGE * ROW_BYTES;  // SRC_MODE 2: the block page (64-byte aligned)
    constexpr uint32_t PAGE_L1 = (PAGE_BYTES + 63) & ~63u;

    if (row_count == 0) {
        return;
    }
    const auto src = TensorAccessor(src_args, src_addr);
    const auto dst = TensorAccessor(dst_args, dst_addr);
    const auto idx = TensorAccessor(idx_args, idx_addr);
    DataflowBuffer scratch(cb_scratch);
    const uint32_t base_l1 = (scratch.get_write_ptr() + 63) & ~63u;
    // this core's window of the indices (from a 64-byte aligned entry, whole 64-byte lines), then two row buffers
    const uint32_t idx_first = row_start & ~15u;
    const uint32_t idx_read = (row_start + row_count - idx_first) * 4;  // never past the page's end
    const uint32_t idx_bytes = (idx_read + 63) & ~63u;
    const uint32_t idx_l1 = base_l1;
    const uint32_t row_l1[2] = {base_l1 + idx_bytes, base_l1 + idx_bytes + ROW_L1};
    const uint32_t page_l1 = base_l1 + idx_bytes;  // SRC_MODE 2: one page buffer in place of the two row buffers
    volatile tt_l1_ptr uint32_t* indices = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);
    (void)rows;
    (void)PAGE_L1;

    noc_async_read(idx.get_noc_addr(0, idx_first * 4), idx_l1, idx_read);
    noc_async_read_barrier();
    uint32_t buf = 0;
    uint32_t cur_page = 0xFFFFFFFFu;
    for (uint32_t n = 0; n < row_count; ++n) {
        const uint32_t target = indices[row_start + n - idx_first];
        if (target == SENTINEL) {
            continue;
        }
        const uint32_t i = row_start + n;
        const uint32_t r = dst_base + i * stride + target;
        uint32_t src_l1 = row_l1[buf];
        if constexpr (SRC_MODE == 2) {
            const uint32_t s = src_base + i;
            const uint32_t page = s / ROWS_PER_PAGE;
            if (page != cur_page) {
                noc_async_write_barrier();  // the previous page's rows have left L1 before it is replaced
                noc_async_read(src.get_noc_addr(page, 0), page_l1, PAGE_BYTES);
                noc_async_read_barrier();
                cur_page = page;
            }
            src_l1 = page_l1 + (s % ROWS_PER_PAGE) * ROW_BYTES;
        } else {
            noc_async_read(src.get_noc_addr(src_base + i, 0), src_l1, ROW_BYTES);
            noc_async_read_barrier();
        }
        if constexpr (DST_MODE == 1) {
            const uint32_t ri = r & 31u;
            const uint32_t face_off = (ri >> 4) * 2 * FACE_BYTES + (ri & 15u) * FACE_ROW_BYTES;
            const uint32_t page0 = (r >> 5) * TILES_PER_ROW;
            for (uint32_t c = 0; c < TILES_PER_ROW; ++c) {
                noc_async_write(src_l1 + c * 64, dst.get_noc_addr(page0 + c, face_off), FACE_ROW_BYTES);
                noc_async_write(
                    src_l1 + c * 64 + 32, dst.get_noc_addr(page0 + c, face_off + FACE_BYTES), FACE_ROW_BYTES);
            }
        } else if constexpr (DST_MODE == 2) {
            noc_async_write(src_l1, dst.get_noc_addr(r / ROWS_PER_PAGE, (r % ROWS_PER_PAGE) * ROW_BYTES), ROW_BYTES);
        } else {
            noc_async_write(src_l1, dst.get_noc_addr(r, 0), ROW_BYTES);
        }
        if constexpr (SRC_MODE != 2) {
            buf ^= 1;
            if (buf == 0) {
                noc_async_write_barrier();  // both buffers' writes have left L1 before they are reused
            }
        }
    }
    noc_async_write_barrier();
}
