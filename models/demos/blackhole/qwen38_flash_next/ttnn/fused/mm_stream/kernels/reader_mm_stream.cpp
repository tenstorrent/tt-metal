// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// x52 streamed decode matmul, weight reader (NOC 0, one per compute core): this core owns `ncols` adjacent tile columns
// of one DRAM bank's width shard (row-major tiles, `bank_tiles` per row): for every K row one read of its ncols tiles
// (contiguous), into the weight CB in K order, `blk` rows per CB block (trid pipelined, slots - 1 blocks in flight).
// Runtime args: weight address, bank id, first column in the bank, ncols, vc.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
    constexpr uint32_t kt = get_named_compile_time_arg_val("kt");
    constexpr uint32_t blk = get_named_compile_time_arg_val("blk");
    constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    constexpr uint32_t bank_tiles = get_named_compile_time_arg_val("bank_tiles");
    constexpr uint32_t tile_bytes = get_named_compile_time_arg_val("w_tile_bytes");
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t col0 = get_arg_val<uint32_t>(2);
    const uint32_t ncols = get_arg_val<uint32_t>(3);
    const uint32_t vc = get_arg_val<uint32_t>(4);
    constexpr uint32_t blocks = kt / blk;
    constexpr uint32_t in_flight = slots - 1;

    CircularBuffer w(cb_w);
    const uint32_t base = w.get_write_ptr();
    const uint32_t row_bytes = ncols * tile_bytes;
    const uint32_t block_bytes = blk * row_bytes;
    const uint64_t bank_base = get_noc_addr_from_bank_id<true>(bank, 0);
    noc_async_read_one_packet_set_state<true>(bank_base, row_bytes, vc);
    uint32_t trid_issue = 1, trid_wait = 1, slot = 0, pending = 0;
    FUSED_ZONE("fz_mm_r");
    w.reserve_back(blk * ncols * (in_flight - 1));
    for (uint32_t b = 0; b < blocks; ++b) {
        noc_async_read_set_trid(trid_issue);
        for (uint32_t r = 0; r < blk; ++r) {
            const uint32_t k = b * blk + r;
            noc_async_read_one_packet_with_state_with_trid<false, false>(
                bank_base,
                w_addr + (k * bank_tiles + col0) * tile_bytes,
                base + slot * block_bytes + r * row_bytes,
                trid_issue);
        }
        if (++slot == slots) {
            slot = 0;
        }
        if (++trid_issue > slots) {
            trid_issue = 1;
        }
        if (++pending == in_flight) {
            noc_async_read_barrier_with_trid(trid_wait);
            w.push_back(blk * ncols);
            if (++trid_wait > slots) {
                trid_wait = 1;
            }
            --pending;
            w.reserve_back(blk * ncols * in_flight);
        }
    }
    for (; pending > 0; --pending) {
        noc_async_read_barrier_with_trid(trid_wait);
        w.push_back(blk * ncols);
        if (++trid_wait > slots) {
            trid_wait = 1;
        }
    }
}
