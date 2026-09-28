// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Linked-multicast rule probe (test_mcast_linked_rule.py): ITERS chains of CHAIN linked multicasts (the last one
// unlinked) of BYTES each into a receiver rectangle; MODE puts a non-multicast transaction on the same NoC into every
// open chain (after its first piece):
//   0 legal     nothing
//   1 write     unicast write (the multicast's own command buffer)
//   2 atomic    unicast semaphore increment (the atomic command buffer)
//   3 read      unicast read (the read command buffer)
// RT: src, dst, mcast start xy, end xy, num dests, unicast xy, unicast addr, read landing addr.
// CT: MODE, ITERS, CHAIN, BYTES.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0), iters = get_compile_time_arg_val(1);
    constexpr uint32_t chain = get_compile_time_arg_val(2), bytes = get_compile_time_arg_val(3);
    const uint32_t src = get_arg_val<uint32_t>(0), dst = get_arg_val<uint32_t>(1);
    const uint32_t s = get_arg_val<uint32_t>(2), e = get_arg_val<uint32_t>(3), ndest = get_arg_val<uint32_t>(4);
    const uint32_t uxy = get_arg_val<uint32_t>(5), uaddr = get_arg_val<uint32_t>(6), rd_l1 = get_arg_val<uint32_t>(7);
    const uint64_t mc = get_noc_multicast_addr(s >> 16, s & 0xFFFF, e >> 16, e & 0xFFFF, 0);
    const uint64_t uni = get_noc_addr(uxy >> 16, uxy & 0xFFFF, uaddr);
    for (uint32_t it = 0; it < iters; ++it) {
        for (uint32_t k = 0; k < chain; ++k) {
            noc_async_write_multicast(src, mc | (dst + k * bytes), bytes, ndest, k + 1 < chain);
            if (k == 0 && k + 1 < chain) {  // inside the open chain
                if constexpr (mode == 1) {
                    noc_async_write(src, uni, bytes);
                } else if constexpr (mode == 2) {
                    noc_semaphore_inc(uni, 1);
                } else if constexpr (mode == 3) {
                    noc_async_read(uni, rd_l1, bytes);
                }
            }
        }
        if constexpr (mode == 3) {
            noc_async_read_barrier();
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
    noc_async_full_barrier();
}
