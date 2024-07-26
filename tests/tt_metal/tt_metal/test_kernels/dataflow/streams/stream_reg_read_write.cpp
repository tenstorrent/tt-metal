// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t target_noc_x = get_arg_val<uint32_t>(0);
    uint32_t target_noc_y = get_arg_val<uint32_t>(1);
    uint32_t stream_id = get_arg_val<uint32_t>(2);
    uint32_t stream_reg = get_arg_val<uint32_t>(3);
    uint32_t value_to_write = get_arg_val<uint32_t>(4);
    uint32_t l1_write_addr = get_arg_val<uint32_t>(5);

    // Write to stream register at `reg_addr` on core [target_noc_x, target_noc_y]
    uint32_t reg_addr = STREAM_REG_ADDR(stream_id, stream_reg);
    uint64_t dest_addr = NOC_XY_ADDR(target_noc_x, target_noc_y, reg_addr);
    noc_inline_dw_write(dest_addr, value_to_write);
    noc_async_write_barrier();

    // Read back value that was written, store in L1 of this core for host to validate
    noc_async_read(dest_addr, l1_write_addr, 4);
    noc_async_read_barrier();
}
