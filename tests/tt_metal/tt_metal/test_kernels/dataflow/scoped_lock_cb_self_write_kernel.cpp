// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Locks a CircularBuffer and then NOC-writes into the CB's own L1 region, so the NOC debug tool must
// report WRITE_TO_LOCKED_CB. The write target is get_write_ptr() (a real byte L1 address), so it only
// overlaps the reported lock region if that region is expressed in bytes. This is a targeted guard for
// CircularBuffer::scoped_lock: a 16x-inflated (<<4) region would sit at cb_base*16 (outside L1) and the
// self-write would NOT be flagged. The exact write target is published to scratch_addr so the host can
// assert issue_address == that address.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    uint32_t cb_id = get_arg_val<uint32_t>(0);
    uint32_t src_buffer_addr = get_arg_val<uint32_t>(1);
    uint32_t write_size = get_arg_val<uint32_t>(2);
    uint32_t self_noc_x = get_arg_val<uint32_t>(3);
    uint32_t self_noc_y = get_arg_val<uint32_t>(4);
    uint32_t scratch_addr = get_arg_val<uint32_t>(5);

    Noc noc;
    UnicastEndpoint unicast_endpoint;
    CoreLocalMem<uint32_t> src_buffer(src_buffer_addr);
    CircularBuffer cb(cb_id);

    {
        auto lock = cb.scoped_lock();
        uint32_t write_target_addr = cb.get_write_ptr();  // real byte address inside the CB region
        noc.async_write(
            src_buffer,
            unicast_endpoint,
            write_size,
            {},
            {.noc_x = self_noc_x, .noc_y = self_noc_y, .addr = write_target_addr});
        noc.async_write_barrier();
        // Publish the write target AFTER the write completes, so if scratch_addr happens to land inside
        // the CB region it is not clobbered by the write above.
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_addr)[0] = write_target_addr;
    }
}
