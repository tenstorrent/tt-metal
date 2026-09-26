// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Split-role DRAM read + forward probe, one kernel file for both RISCs of a reader core:
//   ROLE 0 (reader, BRISC/NOC0): stream [offset, offset + num_bytes) of one bank into CB `cb`, BATCH chunks per
//          reserve/barrier/push.
//   ROLE 1 (forwarder, NCRISC/NOC1): drain the CB BATCH chunks at a time and write every chunk to the receivers
//          (FWD 1: unicast round-robin over a list; FWD 2: multicast rectangle), then pop once the writes left L1.
// Compile-time args: 0 ROLE, 1 CHUNK, 2 BATCH, 3 CB, 4 FWD, 5 DST_SLOTS
// Runtime args: ROLE 0: bank_base, bank_id, bank_offset, num_bytes
//               ROLE 1: num_bytes, dst_addr, then FWD 1: num_dst, packed (x << 16 | y)...; FWD 2: x0, y0, x1, y1, n

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t chunk = get_compile_time_arg_val(1);
    constexpr uint32_t batch = get_compile_time_arg_val(2);
    constexpr uint32_t cb = get_compile_time_arg_val(3);
    constexpr uint32_t fwd = get_compile_time_arg_val(4);
    constexpr uint32_t dst_slots = get_compile_time_arg_val(5);

    if constexpr (role == 0) {
        const uint32_t base = get_arg_val<uint32_t>(0);
        const uint32_t bank_id = get_arg_val<uint32_t>(1);
        const uint32_t offset = get_arg_val<uint32_t>(2);
        const uint32_t num_bytes = get_arg_val<uint32_t>(3);
        const uint64_t src = get_noc_addr_from_bank_id<true>(bank_id, base + offset);
        for (uint32_t done = 0; done < num_bytes; done += batch * chunk) {
            cb_reserve_back(cb, batch);
            const uint32_t l1 = get_write_ptr(cb);
            for (uint32_t c = 0; c < batch; ++c) {
                noc_async_read(src + done + c * chunk, l1 + c * chunk, chunk);
            }
            noc_async_read_barrier();
            cb_push_back(cb, batch);
        }
    } else {
        const uint32_t num_bytes = get_arg_val<uint32_t>(0);
        const uint32_t dst_addr = get_arg_val<uint32_t>(1);
        uint32_t num_dst = 0, mcast_dests = 0;
        uint64_t mcast_base = 0;
        if constexpr (fwd == 1) {
            num_dst = get_arg_val<uint32_t>(2);
        } else {
            mcast_base = get_noc_multicast_addr(
                get_arg_val<uint32_t>(2),
                get_arg_val<uint32_t>(3),
                get_arg_val<uint32_t>(4),
                get_arg_val<uint32_t>(5),
                0);
            mcast_dests = get_arg_val<uint32_t>(6);
        }
        uint32_t rr = 0, slot = 0;
        for (uint32_t done = 0; done < num_bytes; done += batch * chunk) {
            cb_wait_front(cb, batch);
            const uint32_t l1 = get_read_ptr(cb);
            for (uint32_t c = 0; c < batch; ++c) {
                const uint32_t dst = dst_addr + slot * chunk;
                if constexpr (fwd == 1) {
                    const uint32_t xy = get_arg_val<uint32_t>(3 + rr);
                    noc_async_write(l1 + c * chunk, get_noc_addr(xy >> 16, xy & 0xFFFF, dst), chunk);
                    if (++rr == num_dst) {
                        rr = 0;
                        slot = (slot + 1) % dst_slots;
                    }
                } else {
                    noc_async_write_multicast(l1 + c * chunk, mcast_base | dst, chunk, mcast_dests);
                    slot = (slot + 1) % dst_slots;
                }
            }
            noc_async_writes_flushed();
            cb_pop_front(cb, batch);
        }
        noc_async_write_barrier();
    }
}
