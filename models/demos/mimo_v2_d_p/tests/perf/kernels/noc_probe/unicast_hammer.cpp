// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The sender core's other RISC on the multicaster's NoC (test_mcast_linked_rule.py modes write_other / read_other):
// ITERS unicast writes (CT MODE 1) or reads (2) of BYTES to / from a core outside the rectangle, while the other RISC
// runs linked multicast chains. RT: src / landing L1, unicast xy, unicast addr.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0), iters = get_compile_time_arg_val(1);
    constexpr uint32_t bytes = get_compile_time_arg_val(2);
    const uint32_t l1 = get_arg_val<uint32_t>(0), uxy = get_arg_val<uint32_t>(1), uaddr = get_arg_val<uint32_t>(2);
    const uint64_t uni = get_noc_addr(uxy >> 16, uxy & 0xFFFF, uaddr);
    for (uint32_t it = 0; it < iters; ++it) {
        if constexpr (mode == 1) {
            noc_async_write(l1, uni, bytes);
            if (it % 8 == 7) {
                noc_async_writes_flushed();
            }
        } else {
            noc_async_read(uni, l1, bytes);
            if (it % 8 == 7) {
                noc_async_read_barrier();
            }
        }
    }
    noc_async_write_barrier();
    noc_async_read_barrier();
    noc_async_full_barrier();
}
