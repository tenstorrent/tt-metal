// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Test consumer of the weight stream (stream.py): take n slots from this receiver's ring, copy each into a local CB
// (as a linear's reader would), give it back to the sender.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t SLOT = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    // runtime args: n, ring, filled sem, consumed word (persistent), sender noc x, y, freed entry addr (on the sender)
    const uint32_t n = get_arg_val<uint32_t>(0);
    const uint32_t ring = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* filled = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(2));
    volatile tt_l1_ptr uint32_t* consumed =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(3));  // persistent
    const uint64_t freed = get_noc_addr(get_arg_val<uint32_t>(4), get_arg_val<uint32_t>(5), get_arg_val<uint32_t>(6));
    const uint32_t buf = get_write_ptr(0);
    uint32_t c = *consumed;
    for (uint32_t i = 0; i < n; ++i, ++c) {
        while ((int32_t)(*filled - c) <= 0) {
            invalidate_l1_cache();
        }
        noc_async_read(get_noc_addr(ring + (c % S) * SLOT), buf, SLOT);
        noc_async_read_barrier();
        noc_semaphore_inc(freed, 1);
    }
    *consumed = c;
    noc_async_atomic_barrier();
}
