// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC (NOC 0): the distinct experts from the uint16 stick [count, ids...] (the deduplicated gate|up stream's),
// their count for compute (cb_cnt), then every expert's w_down slice streamed from this core's DRAM bank, one group
// of ct output columns of every expert at a time (the order compute accumulates in).
//
// The DRAM-interleaved w_down [1, E, kt * 32, 8 * cols * 32] puts tile column n in bank n % 8, so bank b holds expert
// e as kt rows of cols tiles (output columns b, b + 8, ...), contiguous per expert; a block is one pair's group:
// kt reads of ct tiles, landing as [kt][ct] for custom_mm.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "moe_stream_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ids = get_compile_time_arg_val(1);
    constexpr uint32_t num_pairs = get_compile_time_arg_val(2);
    constexpr uint32_t groups = get_compile_time_arg_val(3);
    constexpr uint32_t kt = get_compile_time_arg_val(4);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t block_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t group_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t expert_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(10);
    constexpr uint32_t in_flight = get_compile_time_arg_val(11);
    constexpr uint32_t ids_bytes = get_compile_time_arg_val(12);
    constexpr uint32_t cb_cnt = get_compile_time_arg_val(13);

    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t ids_addr = get_arg_val<uint32_t>(3);

    MOE_MARK(40);
    // Experts: row-major uint16 [1, 1, 1, >= pairs + 1] = [count, ids...], one page in bank 0.
    const uint32_t ids_l1 = get_write_ptr(cb_ids);
    noc_async_read(get_noc_addr_from_bank_id<true>(0, ids_addr), ids_l1, ids_bytes);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(ids_l1);
    const uint32_t n = ids[0];
    uint32_t offsets[num_pairs];
    for (uint32_t p = 0; p < n; p++) {
        offsets[p] = static_cast<uint32_t>(ids[1 + p]) * expert_bytes;
    }
    cb_reserve_back(cb_cnt, 1);
    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_cnt)) = n;
    cb_push_back(cb_cnt, 1);
    MOE_MARK(41);

    // Stream: block i goes to CB slot i % num_buffers with trid (i % num_buffers) + 1; at most in_flight blocks
    // are issued but not pushed, and each reserve covers them plus the new block.
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    const uint64_t w_noc_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    noc_async_read_one_packet_set_state<true>(w_noc_base, group_bytes, vc);
    const uint32_t cb_base = get_write_ptr(cb_w);

    const uint32_t total_blocks = groups * n;
    uint32_t issued = 0;
    uint32_t oldest = 0;
    for (uint32_t i = 0; i < total_blocks; i++) {
        if (issued == in_flight) {
            noc_async_read_barrier_with_trid((oldest % num_buffers) + 1);
            cb_push_back(cb_w, block_tiles);
            oldest++;
            issued--;
        }
        cb_reserve_back(cb_w, block_tiles * (issued + 1));
        const uint32_t trid = (i % num_buffers) + 1;
        noc_async_read_set_trid(trid);
        const uint32_t g = i / n;
        const uint32_t p = i - g * n;
        uint32_t src = offsets[p] + g * group_bytes;
        uint32_t dst = cb_base + (i % num_buffers) * block_bytes;
        for (uint32_t k = 0; k < kt; k++) {
            noc_async_read_one_packet_with_state_with_trid(w_noc_base, src, dst, trid);
            src += row_bytes;
            dst += group_bytes;
        }
        issued++;
    }
    while (issued > 0) {
        noc_async_read_barrier_with_trid((oldest % num_buffers) + 1);
        cb_push_back(cb_w, block_tiles);
        oldest++;
        issued--;
    }
    MOE_MARK(42);
}
