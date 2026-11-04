// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads this core's own L1 past its populated end over the NoC, which makes the L1 ECC report false errors.
// arg 0: source L1 address, arg 1: destination L1 address, arg 2: bytes.

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"

void kernel_main() {
    uint32_t src = get_arg_val<uint32_t>(0);
    uint32_t dst = get_arg_val<uint32_t>(1);
    uint32_t bytes = get_arg_val<uint32_t>(2);

    Noc noc;
    UnicastEndpoint self;
    CoreLocalMem<uint32_t> dst_mem(dst);
    noc.async_read(
        self, dst_mem, bytes, {.noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = src}, {});
    noc.async_read_barrier();
}
