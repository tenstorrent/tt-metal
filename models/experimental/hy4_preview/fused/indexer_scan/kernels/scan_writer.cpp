// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer of one scan core: the survivor hand-offs of the binary tree over a user's banks, and the root's result.
//
// A survivor (2048 fused words, 2 raw 32-bit tiles) goes from a sender's cb_send straight into its partner's
// cb_landing. Every scan core has the same circular buffers, so cb_landing has the same L1 address on all of
// them, and it holds exactly one survivor, so that address never moves. Flow control per hand-off, as in
// ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/writer.cpp:
//     receiver: reserve cb_landing -> credit to that level's partner -> wait for its data count -> push
//     sender:   wait for the credit -> wait for its own survivor -> write -> barrier -> data count to the partner
// A receiver takes its partners level by level; cb_landing is reserved again only after the compute kernel has
// consumed the previous survivor.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "scan_common.hpp"

namespace {

using namespace hy4_scan;

constexpr bool scores_only = get_compile_time_arg_val(0) != 0;
constexpr uint32_t score_tiles_max = get_compile_time_arg_val(1);
constexpr uint32_t credit_id = get_compile_time_arg_val(2) + sem_credit;  // program semaphore ids
constexpr uint32_t data_id = get_compile_time_arg_val(2) + sem_data;
constexpr uint32_t max_levels = 3;
constexpr uint32_t survivor_bytes = survivor_tiles * word_tile_bytes;

FORCE_INLINE uint64_t interleaved_page(uint32_t base, uint32_t page, uint32_t page_bytes) {
    return get_noc_addr_from_bank_id<true>(page % NUM_DRAM_BANKS, base + (page / NUM_DRAM_BANKS) * page_bytes);
}

}  // namespace

void kernel_main() {
    const uint32_t num_receives = get_arg_val<uint32_t>(0);
    const bool sends = get_arg_val<uint32_t>(1) != 0;
    // args 2..7: (noc x, noc y) of the partner whose survivor lands here at level 0, 1, 2
    const uint32_t send_x = get_arg_val<uint32_t>(2 + 2 * max_levels);
    const uint32_t send_y = get_arg_val<uint32_t>(3 + 2 * max_levels);
    const uint32_t out_addr = get_arg_val<uint32_t>(4 + 2 * max_levels);
    const uint32_t out_page = get_arg_val<uint32_t>(5 + 2 * max_levels);

    if constexpr (scores_only) {
        for (uint32_t tile = 0; tile < score_tiles_max; ++tile) {
            cb_wait_front(cb_scores, 1);
            noc_async_write(
                get_read_ptr(cb_scores), interleaved_page(out_addr, out_page + tile, bf16_tile_bytes), bf16_tile_bytes);
            noc_async_write_barrier();
            cb_pop_front(cb_scores, 1);
        }
        return;
    }

    volatile tt_l1_ptr uint32_t* credit = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_id));
    volatile tt_l1_ptr uint32_t* data = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(data_id));
    const uint32_t landing_address = get_write_ptr(cb_landing);

    for (uint32_t level = 0; level < num_receives; ++level) {
        const uint32_t partner_x = get_arg_val<uint32_t>(2 + 2 * level);
        const uint32_t partner_y = get_arg_val<uint32_t>(3 + 2 * level);
        cb_reserve_back(cb_landing, survivor_tiles);
        noc_semaphore_inc(get_noc_addr(partner_x, partner_y, get_semaphore(credit_id)), 1);
        // The partner raises the count only after its write has drained, so the survivor is complete here.
        noc_semaphore_wait_min(data, level + 1);
        cb_push_back(cb_landing, survivor_tiles);
    }

    if (sends) {
        noc_semaphore_wait_min(credit, 1);
        cb_wait_front(cb_send, survivor_tiles);
        noc_async_write(get_read_ptr(cb_send), get_noc_addr(send_x, send_y, landing_address), survivor_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_send, survivor_tiles);
        noc_semaphore_inc(get_noc_addr(send_x, send_y, get_semaphore(data_id)), 1);
    } else {
        cb_wait_front(cb_out, survivor_tiles);
        const uint32_t out_ptr = get_read_ptr(cb_out);
        for (uint32_t tile = 0; tile < survivor_tiles; ++tile) {
            noc_async_write(
                out_ptr + tile * word_tile_bytes,
                interleaved_page(out_addr, out_page + tile, word_tile_bytes),
                word_tile_bytes);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, survivor_tiles);
    }
    noc_async_atomic_barrier();
}
