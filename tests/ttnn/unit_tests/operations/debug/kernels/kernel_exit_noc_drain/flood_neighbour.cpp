// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Flooder of test_kernel_exit_noc_drain.py, first program, core F (BRISC): keeps the path to core B busy with 4 KiB
// non-posted writes so that the producer's atomic responses come back late. The final wait is capped.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t target_x = get_arg_val<uint32_t>(0);
    const uint32_t target_y = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t flood_offset = get_arg_val<uint32_t>(3);
    const uint32_t cap = get_arg_val<uint32_t>(4);

    const uint32_t page = get_write_ptr(tt::CBIndex::c_0);
    const uint64_t destination = get_noc_addr(target_x, target_y, page + flood_offset);
    for (uint32_t i = 0; i < count; ++i) {
        noc_async_write(page, destination, 4096);
    }
    for (uint32_t p = 0; p < cap; ++p) {
        if (ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            break;
        }
    }
}
