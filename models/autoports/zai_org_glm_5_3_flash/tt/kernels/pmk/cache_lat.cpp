// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK cache-write phase, MLA latent pages (sparse_dsa layers; tt/pmk_cache.py latent_parts): the layer's latent rows
// [T, KX x 32] bf16 TILE (the normalised kv_lora rows, every packed row, DRAM or L1 interleaved: row-tile r = tiles
// [r KX, (r + 1) KX)) -> the slot's physical pages of the paged latent cache [num_blocks, 1, BLOCK, KX x 32] bf16
// (page p = tiles [p TPB, (p + 1) TPB), TPB = (BLOCK / 32) KX: ttnn's paged_fill_cache layout). Whole pages: segments
// start on 128-row boundaries (prefill_tp.py packed_align), so a BLOCK-row page lies inside one segment, and the
// op path writes its pad rows too (chunk_pages) -- no read-modify-write (the decode kernel kernels/mla/lat_dm1.cpp
// patches one row; this one copies pages).
//
// Page i (packed rows [i BLOCK, (i + 1) BLOCK)) of this core's range [first, first + count):
//   k = segment of row i BLOCK (SEGMENT TABLE, cache_common.hpp); none, or the page holds no valid row -> a PAD page:
//       the NULL-block rule (chunk_pages' spare page): skipped (default) or written to NULL_BLOCK 0 (WRITE_NULL);
//   else dst = page_table[seg.page_row][(seg.pos0 + i BLOCK - seg.row0) / BLOCK] (< 0: skipped), copied src -> dst
//   through DEPTH tile slots of scratch (read batch, barrier, write batch, barrier).
// Pages outside the source rows [src_row0, src_row0 + src_rows) are not this chip's (a local-rows source) and skipped.
// WAIT_MODE 2: page i waits for its last row-tile's 32-row block counter (>= 2 (i + 1) - src_row0 / 32).
//
// SRC_MODE 0: src is a tensor (DRAM / L1 interleaved, TP2's DRAM stores). SRC_MODE 1 (prefill_megakernel.md §4b
// H2 / NOTE_16_GEOMETRY): the rows are 32-row L1 blocks, block j = KX contiguous bf16 tiles at the SAME L1 address
// blk_addr on MM core j (its (x, y) NoC pair is runtime arg 8 + 2 j; j counts from src_row0 / 32) -- the attention
// phase's hand-off CB; the src accessor is unused.
// Compile-time (PH_CT0 +): CB_SCR, KX, BLOCK, SEM_WAIT, WAIT_MODE, WAIT_COUNT, NSEG_MAX, PT_BYTES (page-table row
//   bytes), DEPTH, SRC_MODE, then the TensorAccessorArgs of src, cache, page table, segtab.
// Runtime (ph_rt0 +): src (SRC_MODE 1: blk_addr), cache, page table, segtab addresses, src_row0, src_rows, first page,
//   page count, then (SRC_MODE 1) src_rows / 32 (x, y) pairs.
// Scratch CB_SCR: DEPTH x 2048 + align64(table) + NSEG_MAX x align64(PT_BYTES).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "cache_common.hpp"

void kernel_main() {
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t KX = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr uint32_t SEM_WAIT = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t WAIT_MODE = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t WAIT_COUNT = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t NSEG_MAX = get_compile_time_arg_val(PH_CT0 + 6);
    constexpr uint32_t PT_BYTES = get_compile_time_arg_val(PH_CT0 + 7);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(PH_CT0 + 8);
    constexpr uint32_t SRC_MODE = get_compile_time_arg_val(PH_CT0 + 9);
    constexpr auto s_args = TensorAccessorArgs<PH_CT0 + 10>();
    constexpr auto c_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto p_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr auto t_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    const auto src = TensorAccessor(s_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const auto cache = TensorAccessor(c_args, get_arg_val<uint32_t>(ph_rt0 + 1));
    const auto pt = TensorAccessor(p_args, get_arg_val<uint32_t>(ph_rt0 + 2));
    const auto tab = TensorAccessor(t_args, get_arg_val<uint32_t>(ph_rt0 + 3));
    const uint32_t src_row0 = get_arg_val<uint32_t>(ph_rt0 + 4);
    const uint32_t src_rows = get_arg_val<uint32_t>(ph_rt0 + 5);
    const uint32_t first = get_arg_val<uint32_t>(ph_rt0 + 6);
    const uint32_t count = get_arg_val<uint32_t>(ph_rt0 + 7);
    constexpr uint32_t TB = pmk::TILE_BF16;
    constexpr uint32_t RT = BLOCK / 32;  // row-tiles per page
    constexpr uint32_t TPB = RT * KX;    // tiles per page
    constexpr uint32_t PT_AL = pmk::align64(PT_BYTES);
    static_assert(DEPTH >= 1 && TPB % DEPTH == 0, "a page is a whole number of batches");
    if (count == 0) {
        return;
    }

    const uint32_t scr = get_write_ptr(CB_SCR);
    const uint32_t tab_l1 = scr + DEPTH * TB;
    const uint32_t pt_l1 = tab_l1 + pmk::align64(pmk::table_bytes(NSEG_MAX));
    if constexpr (WAIT_MODE == pmk::WAIT_ALL) {
        pmk::wait_min(SEM_WAIT, WAIT_COUNT);
    }
    const pmk::SegTable st{pmk::load_segtab(tab, tab_l1)};
    // the page-table rows of every segment, once
    const uint32_t n = st.n();
    for (uint32_t k = 0; k < n; ++k) {
        if (st.seg(k, pmk::S_SLOT) != pmk::SEG_NONE) {
            noc_async_read_page(st.seg(k, pmk::S_PTROW), pt, pt_l1 + k * PT_AL);
        }
    }
    noc_async_read_barrier();
    invalidate_l1_cache();

    for (uint32_t i = first; i < first + count; ++i) {
        const uint32_t r = i * BLOCK;
        if (r < src_row0 || r - src_row0 >= src_rows) {
            continue;  // not this chip's rows
        }
        const uint32_t k = st.seg_of_row(r);
        uint32_t dst;
        if (k == pmk::SEG_NONE || !st.valid(k, r)) {
#if defined(WRITE_NULL)
            dst = pmk::NULL_BLOCK;  // a pad page: the op path's spare page
#else
            continue;
#endif
        } else {
            const int32_t p = reinterpret_cast<volatile tt_l1_ptr int32_t*>(pt_l1 + k * PT_AL)[st.pos(k, r) / BLOCK];
            if (p < 0) {
                continue;  // vLLM's -1 tail: no page
            }
            dst = static_cast<uint32_t>(p);
        }
        if constexpr (WAIT_MODE == pmk::WAIT_BLOCKS) {
            pmk::wait_min(SEM_WAIT, (r - src_row0) / 32 + RT);  // the page's last row-tile has landed
        }
        const uint32_t s0 = ((r - src_row0) / 32) * KX, d0 = dst * TPB;
        for (uint32_t b = 0; b < TPB; b += DEPTH) {
            for (uint32_t j = 0; j < DEPTH; ++j) {
                if constexpr (SRC_MODE == 1) {  // tile (b + j) of the page = tile (b + j) % KX of row block (s0 + b +
                                                // j) / KX
                    const uint32_t blk = (s0 + b + j) / KX, tl = (s0 + b + j) % KX;
                    const uint32_t x = get_arg_val<uint32_t>(ph_rt0 + 8 + 2 * blk),
                                   y = get_arg_val<uint32_t>(ph_rt0 + 9 + 2 * blk);
                    noc_async_read(get_noc_addr(x, y, get_arg_val<uint32_t>(ph_rt0 + 0) + tl * TB), scr + j * TB, TB);
                } else {
                    noc_async_read_page(s0 + b + j, src, scr + j * TB);
                }
            }
            noc_async_read_barrier();
            for (uint32_t j = 0; j < DEPTH; ++j) {
                noc_async_write_page(d0 + b + j, cache, scr + j * TB);
            }
            noc_async_write_barrier();
        }
    }
}
