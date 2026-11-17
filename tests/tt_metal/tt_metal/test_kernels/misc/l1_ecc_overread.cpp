// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reads this core's own L1 past its populated end and sends it to another core's L1 over the NoC, which makes this
// core's L1 ECC report false errors.
// arg 0: source L1 address, arg 1: destination core NOC x, arg 2: destination core NOC y, arg 3: destination L1
// address, arg 4: bytes.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src = get_arg_val<uint32_t>(0);
    uint32_t dst_x = get_arg_val<uint32_t>(1);
    uint32_t dst_y = get_arg_val<uint32_t>(2);
    uint32_t dst = get_arg_val<uint32_t>(3);
    uint32_t bytes = get_arg_val<uint32_t>(4);

    noc_async_write(src, get_noc_addr(dst_x, dst_y, dst), bytes);
    noc_async_write_barrier();
}
