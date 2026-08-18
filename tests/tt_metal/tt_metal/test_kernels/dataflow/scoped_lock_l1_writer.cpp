// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// A NOC write into this core's own L1 at a host-supplied address, with no DataflowBuffer binding.
//
// Used as the second launch of the region-cleared-between-launches test: it targets the L1 that a
// previous program's DFB occupied.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t src_buffer_addr = get_arg(args::src_buffer_addr);
    const uint32_t write_size = get_arg(args::write_size);
    const uint32_t self_noc_x = get_arg(args::self_noc_x);
    const uint32_t self_noc_y = get_arg(args::self_noc_y);
    const uint32_t target_addr = get_arg(args::target_addr);

    Noc noc;
    UnicastEndpoint unicast_endpoint;
    CoreLocalMem<uint32_t> src_buffer(src_buffer_addr);

    noc.async_write(
        src_buffer, unicast_endpoint, write_size, {}, {.noc_x = self_noc_x, .noc_y = self_noc_y, .addr = target_addr});
    noc.async_write_barrier();
}
