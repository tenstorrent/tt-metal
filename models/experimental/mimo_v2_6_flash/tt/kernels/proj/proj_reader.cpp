// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// NCRISC (NOC 0): this core's columns of its DRAM bank's weight tiles, streamed into cb_w block by block.
//
// The DRAM-interleaved w [K, 8 * cpb * 32] puts tile column n in bank n % 8, so bank b holds kt rows of cpb tiles
// (columns b, b + 8, ...), contiguous per row. The cores of bank b split the cpb slots: this core reads its row_bytes
// at col_offset of every row as one packet, a block of block_rows rows landing as [rows][cols].
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_w = get_compile_time_arg_val(0);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(1);
    constexpr uint32_t block_rows = get_compile_time_arg_val(2);
    constexpr uint32_t block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t bank_row_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t num_buffers = get_compile_time_arg_val(6);
    constexpr uint32_t in_flight = get_compile_time_arg_val(7);
    constexpr uint32_t block_bytes = block_rows * row_bytes;

    const uint32_t bank_id = get_arg_val<uint32_t>(0);
    const uint32_t vc = get_arg_val<uint32_t>(1);
    const uint32_t w_addr = get_arg_val<uint32_t>(2);
    const uint32_t col_offset = get_arg_val<uint32_t>(3);

    // Block i goes to CB slot i % num_buffers with trid (i % num_buffers) + 1; at most in_flight blocks are issued
    // but not pushed, and each reserve covers them plus the new block.
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    const uint64_t w_noc_base = get_noc_addr_from_bank_id<true>(bank_id, w_addr);
    noc_async_read_one_packet_set_state<true>(w_noc_base, row_bytes, vc);
    const uint32_t cb_base = get_write_ptr(cb_w);

    uint32_t issued = 0;
    uint32_t oldest = 0;
    uint32_t src = col_offset;
    for (uint32_t i = 0; i < num_blocks; i++) {
        if (issued == in_flight) {
            noc_async_read_barrier_with_trid((oldest % num_buffers) + 1);
            cb_push_back(cb_w, block_tiles);
            oldest++;
            issued--;
        }
        cb_reserve_back(cb_w, block_tiles * (issued + 1));
        const uint32_t trid = (i % num_buffers) + 1;
        noc_async_read_set_trid(trid);
        uint32_t dst = cb_base + (i % num_buffers) * block_bytes;
        for (uint32_t r = 0; r < block_rows; r++) {
            noc_async_read_one_packet_with_state_with_trid(w_noc_base, src, dst, trid);
            src += bank_row_bytes;
            dst += row_bytes;
        }
        issued++;
    }
    while (issued > 0) {
        noc_async_read_barrier_with_trid((oldest % num_buffers) + 1);
        cb_push_back(cb_w, block_tiles);
        oldest++;
        issued--;
    }
}
