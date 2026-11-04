// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// TEMP experiment (GPT_OSS_DECODE_EXPERTS_SPLIT=1, fused_decode/mxfp4_split.py): routed-expert weight streamer
// (NCRISC, NOC0) for the exact MXFP4 split streams. Per DRAM bank a weight segment is [BFP4 part][BFP2 part]
// (part4 / part2 bytes) and segments are laid out like the BFP4 streams (stream_gate_up_reader.cpp /
// stream_down_reader.cpp). A block is what the compute consumes at once:
//   mode 0 (gate|up): one column of one routed expert = 1 segment at offsets[e] + c * (part4 + part2);
//   mode 1 (down):    one output column of all routed experts = num_sel segments at offsets[e] + c * (part4 + part2).
// The BFP4 parts of a block go to cb_w4 (one page), the BFP2 parts to cb_w2 (one page). Reads use transaction ids so
// one block is in flight while compute consumes the previous one.
// SPLIT_RANKS=k: each segment is [BFP4 part][BFP2 part][rounded BFP4 part]; only the first k routed experts (ranks)
// read the exact parts, the others the rounded part into cb_w4 and nothing into cb_w2 (mode 0: no cb_w2 page for
// their blocks; mode 1: a cb_w2 page holds the k exact experts' BFP2 parts).
//
// SPLIT_SEPARATE (with 0 < SPLIT_RANKS): per bank all [BFP4][BFP2] segments, then from SPLIT_SEP_BASE all rounded
// parts (expert stride SPLIT_SEP_STRIDE_R, this reader's offset in runtime arg 5), so both read contiguous columns.
//
// runtime args: [w_addr, idx_addr, bank_id, vc, reader_offset_bytes(, rounded reader_offset_bytes)]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_arg_val<uint32_t>(1);
    const uint32_t bank_id = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    const uint32_t reader_offset = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_w4 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w2 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(2);
    constexpr uint32_t cols = get_compile_time_arg_val(3);
    constexpr uint32_t num_sel = get_compile_time_arg_val(4);
    constexpr uint32_t expert_stride = get_compile_time_arg_val(5);
    constexpr uint32_t part4 = get_compile_time_arg_val(6);
    constexpr uint32_t part2 = get_compile_time_arg_val(7);
    constexpr uint32_t page4 = get_compile_time_arg_val(8);
    constexpr uint32_t page2 = get_compile_time_arg_val(9);
    constexpr uint32_t mode = get_compile_time_arg_val(10);
    constexpr uint32_t idx_page_bytes = get_compile_time_arg_val(11);
    constexpr auto idx_args = TensorAccessorArgs<12>();

#ifdef SPLIT_RANKS
    constexpr uint32_t ranks = SPLIT_RANKS;
    constexpr uint32_t partr = part4;
#else
    constexpr uint32_t ranks = num_sel;
    constexpr uint32_t partr = 0;
#endif
#ifdef SPLIT_ROUND_ONLY
    // SPLIT_RANKS=0 with only the rounded parts stored (GPT_OSS_DECODE_SPLIT_ROUND_ONLY): a segment is one BFP4 part
    static_assert(ranks == 0, "a round-only stream has no exact rank");
    constexpr uint32_t seg_bytes = part4;
    constexpr uint32_t round_skip = 0;
#elif defined(SPLIT_SEPARATE)
    static_assert(ranks > 0 && partr > 0, "a separate stream has exact and rounded ranks");
    constexpr uint32_t seg_bytes = part4 + part2;  // the exact region's segments
    constexpr uint32_t round_skip = 0;
#else
    constexpr uint32_t seg_bytes = part4 + part2 + partr;
    constexpr uint32_t round_skip = part4 + part2;
#endif
    constexpr uint32_t segs = mode == 0 ? 1 : num_sel;  // segments per block
    constexpr uint32_t blocks = mode == 0 ? num_sel * cols : cols;
#ifdef SPLIT_W5
    // mode 1 (down) with the package tiles for the inexact ranks: their parts go to their own CB (SPLIT_W5), so the
    // exact BFP4 call and the package-tile call each read a CB from its start
    constexpr uint32_t cb_w5 = SPLIT_W5;
    constexpr bool w5 = mode == 1;
#else
    constexpr uint32_t cb_w5 = cb_w4;
    constexpr bool w5 = false;
#endif
    constexpr uint32_t block4 = (w5 ? ranks : segs) * part4;
    constexpr uint32_t block5 = (num_sel - ranks) * part4;
    constexpr uint32_t block2 = (mode == 0 ? 1 : (ranks > 0 ? ranks : 1)) * part2;  // (unused when no rank is exact)
    static_assert(part4 % page4 == 0 && part2 % page2 == 0, "pages must divide the parts");
    static_assert(page4 <= NOC_MAX_BURST_SIZE && page2 <= NOC_MAX_BURST_SIZE, "one page must fit one NOC packet");

    cb_reserve_back(cb_idx, 1);
    const uint32_t idx_l1 = get_write_ptr(cb_idx);
    const auto s_idx = TensorAccessor(idx_args, idx_addr, idx_page_bytes);
    noc_async_read(s_idx.get_noc_addr(0), idx_l1, idx_page_bytes);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);
    uint32_t offsets[num_sel];
    for (uint32_t e = 0; e < num_sel; ++e) {
        offsets[e] = static_cast<uint32_t>(ids[e]) * expert_stride + reader_offset;
    }
#ifdef SPLIT_SEPARATE
    const uint32_t reader_offset_r = get_arg_val<uint32_t>(5);
    uint32_t offsets_r[num_sel];
    for (uint32_t e = 0; e < num_sel; ++e) {
        offsets_r[e] = SPLIT_SEP_BASE + static_cast<uint32_t>(ids[e]) * SPLIT_SEP_STRIDE_R + reader_offset_r;
    }
#endif

    const uint64_t w_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);

    // Triple-buffered blocks in each CB, one extra block in flight (transaction ids 1..3).
    constexpr uint32_t num_buffers = 3;
    constexpr uint32_t extra_in_flight = 1;
    cb_reserve_back(cb_w4, extra_in_flight + 1);
    cb_reserve_back(cb_w2, extra_in_flight + 1);
    if constexpr (w5) {
        cb_reserve_back(cb_w5, extra_in_flight + 1);
    }
    uint32_t w4 = get_write_ptr(cb_w4);
    uint32_t w2 = get_write_ptr(cb_w2);
    uint32_t w5p = get_write_ptr(cb_w5);
    const uint32_t base4 = w4, end4 = w4 + num_buffers * block4;
    const uint32_t base2 = w2, end2 = w2 + num_buffers * block2;
    const uint32_t base5 = w5p, end5 = w5p + num_buffers * block5;
    uint32_t free_blocks = num_buffers;
    uint32_t trid = 1;
    uint32_t trid_to_wait = 1;
    bool has2[num_buffers + 1];  // per transaction id: whether its block filled a cb_w2 page

    for (uint32_t blk = 0; blk < blocks; ++blk) {
        noc_async_read_set_trid(trid);
        has2[trid] = mode == 1 ? ranks > 0 : blk / cols < ranks;
        for (uint32_t s = 0; s < segs; ++s) {
            const uint32_t e = mode == 0 ? blk / cols : s;
            const uint32_t c = mode == 0 ? blk % cols : blk;
            const bool exact = e < ranks;
#ifdef SPLIT_SEPARATE
            uint32_t src = exact ? offsets[e] + c * seg_bytes : offsets_r[e] + c * part4;
#else
            uint32_t src = offsets[e] + c * seg_bytes + (exact ? 0 : round_skip);
#endif
            uint32_t& dst4 = (w5 && !exact) ? w5p : w4;
            noc_async_read_one_packet_set_state<true>(w_base, page4, vc);
            for (uint32_t p = 0; p < part4 / page4; ++p) {
                noc_async_read_one_packet_with_state_with_trid(w_base, src, dst4, trid);
                src += page4;
                dst4 += page4;
            }
            if (exact) {
                noc_async_read_one_packet_set_state<true>(w_base, page2, vc);
                for (uint32_t p = 0; p < part2 / page2; ++p) {
                    noc_async_read_one_packet_with_state_with_trid(w_base, src, w2, trid);
                    src += page2;
                    w2 += page2;
                }
            }
        }
        if (free_blocks == num_buffers - extra_in_flight) {
            noc_async_read_barrier_with_trid(trid_to_wait);
            cb_push_back(cb_w4, 1);
            if (has2[trid_to_wait]) {
                cb_push_back(cb_w2, 1);
            }
            if constexpr (w5) {
                cb_push_back(cb_w5, 1);
            }
            trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
            cb_reserve_back(cb_w4, extra_in_flight + 1);
            cb_reserve_back(cb_w2, extra_in_flight + 1);
            if constexpr (w5) {
                cb_reserve_back(cb_w5, extra_in_flight + 1);
            }
        } else {
            free_blocks -= 1;
        }
        trid = trid == num_buffers ? 1 : trid + 1;
        if (w4 >= end4) {
            w4 = base4;
        }
        if (w2 >= end2) {
            w2 = base2;
        }
        if (w5p >= end5) {
            w5p = base5;
        }
    }
    for (uint32_t i = 0; i < extra_in_flight; ++i) {
        noc_async_read_barrier_with_trid(trid_to_wait);
        cb_push_back(cb_w4, 1);
        if (has2[trid_to_wait]) {
            cb_push_back(cb_w2, 1);
        }
        if constexpr (w5) {
            cb_push_back(cb_w5, 1);
        }
        trid_to_wait = trid_to_wait == num_buffers ? 1 : trid_to_wait + 1;
    }
}
