// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DRAM read + NoC forward probe. One reader RISC streams a contiguous range of one DRAM bank in CHUNK-byte reads
// into a 2-half L1 ring (HALF bytes each) and forwards every chunk to compute cores over its own NoC while the next
// half is being read:
//   iteration n: wait reads of half n  ->  wait writes of half n-1 flushed  ->  issue reads n+1  ->  issue writes n
// Receivers just take the bytes into an L1 scratch region (no consumer, no flow control): this measures what a
// dedicated reader can push out, not a full producer/consumer protocol.
//
// Compile-time args: 0 CHUNK, 1 HALF (multiple of CHUNK), 2 SCRATCH_CB (reader ring), 3 FWD (0 none, 1 unicast
//   round-robin over the receiver list, 2 multicast to one rectangle), 4 DST_SLOTS (chunks per receiver scratch)
// Runtime args: bank_base, bank_id, bank_offset, num_bytes, dst_addr,
//   FWD 1: num_dst, then num_dst packed (x << 16 | y) NoC coords
//   FWD 2: mcast x_start, y_start, x_end, y_end (already ordered for this RISC's NoC), num_dests

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t chunk = get_compile_time_arg_val(0);
    constexpr uint32_t half = get_compile_time_arg_val(1);
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(2);
    constexpr uint32_t fwd = get_compile_time_arg_val(3);
    constexpr uint32_t dst_slots = get_compile_time_arg_val(4);
    constexpr uint32_t chunks_per_half = half / chunk;

    const uint32_t bank_base = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t bank_offset = get_arg_val<uint32_t>(2);
    const uint32_t num_bytes = get_arg_val<uint32_t>(3);
    const uint32_t dst_addr = get_arg_val<uint32_t>(4);
    const uint32_t ring = get_write_ptr(scratch_cb);
    const uint64_t bank_noc = get_noc_addr_from_bank_id<true>(bank_id, bank_base + bank_offset);
    const uint32_t num_halves = num_bytes / half;

    uint32_t num_dst = 0;
    uint64_t mcast_base = 0;
    uint32_t mcast_dests = 0;
    if constexpr (fwd == 1) {
        num_dst = get_arg_val<uint32_t>(5);
    } else if constexpr (fwd == 2) {
        mcast_base = get_noc_multicast_addr(
            get_arg_val<uint32_t>(5), get_arg_val<uint32_t>(6), get_arg_val<uint32_t>(7), get_arg_val<uint32_t>(8), 0);
        mcast_dests = get_arg_val<uint32_t>(9);
    }

    uint32_t dst_rr = 0, dst_slot = 0;
    auto read_half = [&](uint32_t h) {
        const uint32_t l1 = ring + (h & 1) * half;
        const uint64_t src = bank_noc + h * half;
        for (uint32_t c = 0; c < chunks_per_half; ++c) {
            noc_async_read(src + c * chunk, l1 + c * chunk, chunk);
        }
    };
    auto write_half = [&](uint32_t h) {
        const uint32_t l1 = ring + (h & 1) * half;
        for (uint32_t c = 0; c < chunks_per_half; ++c) {
            const uint32_t dst = dst_addr + dst_slot * chunk;
            if constexpr (fwd == 1) {
                const uint32_t xy = get_arg_val<uint32_t>(6 + dst_rr);
                noc_async_write(l1 + c * chunk, get_noc_addr(xy >> 16, xy & 0xFFFF, dst), chunk);
                if (++dst_rr == num_dst) {
                    dst_rr = 0;
                    dst_slot = (dst_slot + 1) % dst_slots;
                }
            } else {
                noc_async_write_multicast(l1 + c * chunk, mcast_base | dst, chunk, mcast_dests);
                dst_slot = (dst_slot + 1) % dst_slots;
            }
        }
    };

    read_half(0);
    for (uint32_t h = 0; h < num_halves; ++h) {
        noc_async_read_barrier();  // half h landed (nothing newer is in flight yet)
        if constexpr (fwd != 0) {
            noc_async_writes_flushed();  // half h-1 (the buffer read_half(h+1) reuses) has left L1
        }
        if (h + 1 < num_halves) {
            read_half(h + 1);
        }
        if constexpr (fwd != 0) {
            write_half(h);
        }
    }
    noc_async_read_barrier();
    if constexpr (fwd != 0) {
        noc_async_write_barrier();
    }
}
