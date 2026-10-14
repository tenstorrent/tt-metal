// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t src_buffer_addr = get_arg(args::src_buffer_addr);
    const uint32_t write_size = get_arg(args::write_size);
    const uint32_t self_noc_x = get_arg(args::self_noc_x);
    const uint32_t self_noc_y = get_arg(args::self_noc_y);
    const uint32_t target_entry_offset = get_arg(args::target_entry_offset);
    const uint32_t write_after_unlock = get_arg(args::write_after_unlock);
    const uint32_t skip_lock = get_arg(args::skip_lock);
    // L1 word to stage the ring base into, so the host can read it back and hand it to a later,
    // DFB-free program.
    const uint32_t publish_ring_base_addr = get_arg(args::publish_ring_base_addr);

    Noc noc;
    UnicastEndpoint unicast_endpoint;
    DataflowBuffer dfb(dfb::out);
    CoreLocalMem<uint32_t> src_buffer(src_buffer_addr);

    dfb.reserve_back(1);

    // The write lands at ring_base + target_entry_offset: 0 = the one entry scoped_write_lock() locks,
    // 0<offset<ring_size = another in-region (unlocked) entry, ==ring_size = past the region.
    uint32_t ring_base = 0;
    auto do_write = [&]() {
        noc.async_write(
            src_buffer,
            unicast_endpoint,
            write_size,
            {},
            {.noc_x = self_noc_x, .noc_y = self_noc_y, .addr = ring_base + target_entry_offset});
        noc.async_write_barrier();
    };

    if (skip_lock) {
        // Never take the lock: a NOC write into the DFB ring with no lock held -> WRITE_TO_UNLOCKED_DFB.
        ring_base = dfb.get_write_ptr();
        do_write();
    } else if (write_after_unlock) {
        {
            auto lock = dfb.scoped_write_lock();
            ring_base = static_cast<uint32_t>(lock.get_ptr().get_address());
        }
        do_write();  // lock released before the write -> also WRITE_TO_UNLOCKED_DFB
    } else {
        auto lock = dfb.scoped_write_lock();
        ring_base = static_cast<uint32_t>(lock.get_ptr().get_address());
        do_write();  // held lock covers only its entry: offset 0 -> no issue, in-region ->
                     // WRITE_TO_UNLOCKED_DFB, past -> no issue
    }

    if (publish_ring_base_addr != 0) {
        *(volatile tt_l1_ptr uint32_t*)(uintptr_t)publish_ring_base_addr = ring_base;
    }

    dfb.push_back(1);
}
