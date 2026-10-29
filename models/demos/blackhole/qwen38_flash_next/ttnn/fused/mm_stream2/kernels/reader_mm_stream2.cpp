// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// x60 streamed decode matmul (re-laid weights), weight reader (NOC 0, one per compute core): this core's whole weight
// region is ONE contiguous byte range of its DRAM bank's shard (K rows of `ncols` tiles, K order), read as `blocks`
// blocks of `block_tiles` tiles in `page_tiles`-tile packets (<= 16 KB), trid pipelined with `slots - 1` blocks in
// flight, into the weight CB ring (the compute's CB, or with BENCH a free-running ring: no CB handshake). Runtime args:
// weight address, bank id, region offset (bytes in the bank shard), vc.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t cb_w = get_named_compile_time_arg_val("cb_w");
    constexpr uint32_t blocks = get_named_compile_time_arg_val("blocks");
    constexpr uint32_t block_tiles = get_named_compile_time_arg_val("block_tiles");
    constexpr uint32_t page_tiles = get_named_compile_time_arg_val("page_tiles");
    constexpr uint32_t slots = get_named_compile_time_arg_val("slots");
    constexpr uint32_t tile_bytes = get_named_compile_time_arg_val("w_tile_bytes");
    const uint32_t w_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank = get_arg_val<uint32_t>(1);
    const uint32_t offset = get_arg_val<uint32_t>(2);
    const uint32_t vc = get_arg_val<uint32_t>(3);
    constexpr uint32_t in_flight = slots - 1;
    constexpr uint32_t page_bytes = page_tiles * tile_bytes;
    constexpr uint32_t pages = block_tiles / page_tiles;
    constexpr uint32_t block_bytes = block_tiles * tile_bytes;
    static_assert(pages * page_tiles == block_tiles, "page must divide the block");

    CircularBuffer w(cb_w);
    const uint32_t base = w.get_write_ptr();
    const uint64_t bank_base = get_noc_addr_from_bank_id<true>(bank, 0);
    noc_async_read_one_packet_set_state<true>(bank_base, page_bytes, vc);
    uint32_t src = w_addr + offset;
    FUSED_ZONE("fz_mm_r");
    uint32_t trid_issue = 1, trid_wait = 1, slot = 0, pending = 0;
#ifndef BENCH
    w.reserve_back(block_tiles * (in_flight - 1));
#endif
    for (uint32_t b = 0; b < blocks; ++b) {
        noc_async_read_set_trid(trid_issue);
        uint32_t dst = base + slot * block_bytes;
        for (uint32_t p = 0; p < pages; ++p) {
            noc_async_read_one_packet_with_state_with_trid<false, false>(bank_base, src, dst, trid_issue);
            src += page_bytes;
            dst += page_bytes;
        }
        if (++slot == slots) {
            slot = 0;
        }
        if (++trid_issue > slots) {
            trid_issue = 1;
        }
        if (++pending == in_flight) {
            noc_async_read_barrier_with_trid(trid_wait);
#ifndef BENCH
            w.push_back(block_tiles);
#endif
            if (++trid_wait > slots) {
                trid_wait = 1;
            }
            --pending;
#ifndef BENCH
            w.reserve_back(block_tiles * in_flight);
#endif
        }
    }
    for (; pending > 0; --pending) {
        noc_async_read_barrier_with_trid(trid_wait);
#ifndef BENCH
        w.push_back(block_tiles);
#endif
        if (++trid_wait > slots) {
            trid_wait = 1;
        }
    }
}
