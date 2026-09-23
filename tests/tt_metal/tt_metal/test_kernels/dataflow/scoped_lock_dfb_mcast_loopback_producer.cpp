// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

// Loopback multicast into our own DFB ring with no lock held. The mcast rectangle spans a row of cores with
// this producer in the interior (neither the start nor the end corner), and MCAST_INCL_SRC delivers the
// write to the producer too.
void kernel_main() {
    const uint32_t src_buffer_addr = get_arg(args::src_buffer_addr);
    const uint32_t write_size = get_arg(args::write_size);
    const uint32_t mcast_noc_x_start = get_arg(args::mcast_noc_x_start);
    const uint32_t mcast_noc_y_start = get_arg(args::mcast_noc_y_start);
    const uint32_t mcast_noc_x_end = get_arg(args::mcast_noc_x_end);
    const uint32_t mcast_noc_y_end = get_arg(args::mcast_noc_y_end);
    const uint32_t num_dests = get_arg(args::num_dests);

    Noc noc;
    MulticastEndpoint mcast_endpoint;
    DataflowBuffer dfb(dfb::out);
    CoreLocalMem<uint32_t> src_buffer(src_buffer_addr);

    dfb.reserve_back(1);
    const uint32_t target_addr = dfb.get_write_ptr();
    noc.async_write_multicast<NocOptions::MCAST_INCL_SRC>(
        src_buffer,
        mcast_endpoint,
        write_size,
        num_dests,
        {},
        {.noc_x_start = mcast_noc_x_start,
         .noc_y_start = mcast_noc_y_start,
         .noc_x_end = mcast_noc_x_end,
         .noc_y_end = mcast_noc_y_end,
         .addr = target_addr});
    noc.async_write_barrier();
    dfb.push_back(1);
}
