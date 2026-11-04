// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Routed-expert down-projection weight streamer (NCRISC, NOC0) for the gpt-oss decode MoE (fused_decode/stream.py).
//
// Per DRAM bank the down weights are stored expert-major, then output column, then K tile (I_pad / 32 weight tiles
// + 1 bias tile), so one (expert, column) segment is contiguous. For every output column this core owns, the
// segments of the k routed experts are read back to back into one K = k * seg_tiles block: the compute kernel
// reduces the score-weighted expert sum as a single matmul over that block.
//
// runtime args: [w_addr, idx_addr, bank_id, vc, reader_offset_bytes]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t reader_offset = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(1);
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t cols = get_compile_time_arg_val(3);
    constexpr uint32_t num_sel = get_compile_time_arg_val(4);
    constexpr uint32_t expert_stride = get_compile_time_arg_val(5);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(7);
    constexpr auto idx_args = TensorAccessorArgs<8>();

    constexpr uint32_t seg_bytes = seg_tiles * tile_bytes;
    constexpr uint32_t block_tiles = num_sel * seg_tiles;
    constexpr uint32_t block_bytes = num_sel * seg_bytes;
    // A segment is read in packets of page_bytes (whole tiles): one packet in BFP4, several in BFP8.
    constexpr uint32_t pages_per_seg = seg_bytes / page_bytes;
    static_assert(pages_per_seg * page_bytes == seg_bytes, "page size must divide the segment");
    static_assert(page_bytes <= NOC_MAX_BURST_SIZE, "one page must fit one NOC packet");

    cb_reserve_back(cb_idx, 1);
    const uint32_t idx_l1 = get_write_ptr(cb_idx);
    const auto s_idx = TensorAccessor(idx_args, idx_addr, 64);
    noc_async_read(s_idx.get_noc_addr(0), idx_l1, 64);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);
    uint32_t offsets[num_sel];
    for (uint32_t e = 0; e < num_sel; ++e) {
        offsets[e] = static_cast<uint32_t>(ids[e]) * expert_stride + reader_offset;
    }

    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    noc_async_read_one_packet_set_state<true>(w_base, page_bytes, vc);

    constexpr uint32_t num_buffers = 3;
    constexpr uint32_t extra_in_flight = 1;
    cb_reserve_back(cb_w, block_tiles * (extra_in_flight + 1));
    uint32_t l1_write = get_write_ptr(cb_w);
    const uint32_t cb_base = l1_write;
    const uint32_t cb_end = cb_base + num_buffers * block_bytes;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;

#ifdef EXP_TAIL_SUBS
    // TEMP experiment (fused_decode/config.py DECODE_EXPERT_TAIL_SPLIT): the last column's num_sel * pages_per_seg
    // pages are read as EXP_TAIL_SUBS sub-blocks of EXP_TAIL_TILES tiles (in page order), each under its own
    // transaction id (past the ring's 1..num_buffers), and pushed sub-block by sub-block below, so compute multiplies
    // it while the rest lands (stream_matmul_compute.cpp: same tiles, same order into the same DST tile).
    constexpr uint32_t tail_subs = EXP_TAIL_SUBS;
    constexpr uint32_t tail_trid0 = num_buffers + 1;
    constexpr uint32_t pages_per_sub = num_sel * pages_per_seg / tail_subs;
    static_assert(pages_per_sub * tail_subs == num_sel * pages_per_seg, "the sub-blocks are whole pages");
    static_assert(pages_per_sub * page_bytes == EXP_TAIL_TILES * tile_bytes, "EXP_TAIL_TILES tiles per sub-block");
    static_assert(tail_trid0 + tail_subs - 1 <= NOC_MAX_TRANSACTION_ID, "transaction ids");
#endif
    for (uint32_t c = 0; c < cols; ++c) {
#ifdef EXP_TAIL_SUBS
        if (c + 1 == cols) {
            uint32_t q = 0;  // page index within the column
            for (uint32_t e = 0; e < num_sel; ++e) {
                uint32_t src = offsets[e] + c * seg_bytes;
                for (uint32_t p = 0; p < pages_per_seg; ++p, ++q) {
                    const uint32_t t = tail_trid0 + q / pages_per_sub;
                    noc_async_read_set_trid(t);
                    noc_async_read_one_packet_with_state_with_trid(w_base, src, l1_write, t);
                    src += page_bytes;
                    l1_write += page_bytes;
                }
            }
        } else
#endif
        {
            noc_async_read_set_trid(trid);
            for (uint32_t e = 0; e < num_sel; ++e) {
                uint32_t src = offsets[e] + c * seg_bytes;
                for (uint32_t p = 0; p < pages_per_seg; ++p) {
                    noc_async_read_one_packet_with_state_with_trid(w_base, src, l1_write, trid);
                    src += page_bytes;
                    l1_write += page_bytes;
                }
            }
        }
        if (free_blocks == num_buffers - extra_in_flight) {
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb_w, block_tiles);
            trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
            cb_reserve_back(cb_w, block_tiles * (extra_in_flight + 1));
        } else {
            free_blocks -= 1;
        }
        trid = trid == num_buffers ? 1 : trid + 1;
        if (l1_write >= cb_end) {
            l1_write = cb_base;
        }
    }
#ifdef EXP_TAIL_SUBS
    // the column left in flight is the last one (extra_in_flight = 1): push it sub-block by sub-block
    static_assert(extra_in_flight == 1, "the tail is one column");
    for (uint32_t t = 0; t < tail_subs; ++t) {
        noc_async_read_barrier_with_trid(tail_trid0 + t);
        cb_push_back(cb_w, EXP_TAIL_TILES);
    }
#else
    for (uint32_t i = 0; i < extra_in_flight; ++i) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w, block_tiles);
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
#endif
}
