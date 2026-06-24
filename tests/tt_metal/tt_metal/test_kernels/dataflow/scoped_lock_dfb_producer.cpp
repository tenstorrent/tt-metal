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

    Noc noc;
    UnicastEndpoint unicast_endpoint;
    DataflowBuffer dfb(dfb::out);
    CoreLocalMem<uint32_t> src_buffer(src_buffer_addr);

    dfb.reserve_back(1);

    // On WH/BH scoped_lock() locks the ENTIRE ring; get_write_ptr() (==ring base on a fresh buffer, no shift:
    // cb_addr_shift==0) is the first locked byte. target_entry_offset < ring_size lands inside the locked ring;
    // == ring_size lands at fifo_limit, the first byte outside it.
    uint32_t target_addr = dfb.get_write_ptr() + target_entry_offset;

    if (write_after_unlock) {
        {
            auto lock = dfb.scoped_lock();
        }
        noc.async_write(
            src_buffer,
            unicast_endpoint,
            write_size,
            {},
            {.noc_x = self_noc_x, .noc_y = self_noc_y, .addr = target_addr});
        noc.async_write_barrier();
    } else {
        auto lock = dfb.scoped_lock();
        noc.async_write(
            src_buffer,
            unicast_endpoint,
            write_size,
            {},
            {.noc_x = self_noc_x, .noc_y = self_noc_y, .addr = target_addr});
        noc.async_write_barrier();
    }

    dfb.push_back(1);
}
