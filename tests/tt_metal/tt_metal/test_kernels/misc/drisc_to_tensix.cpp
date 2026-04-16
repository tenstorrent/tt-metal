// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "noc_nonblocking_api.h"
#include "drisc_mode.h"

void kernel_main() {
    constexpr uint32_t tensix_l1_addr = get_compile_time_arg_val(0);
    constexpr uint32_t dram_l1_addr = get_compile_time_arg_val(1);
    constexpr uint32_t data = get_compile_time_arg_val(2);
    constexpr uint32_t tensix_noc_x = get_compile_time_arg_val(3);
    constexpr uint32_t tensix_noc_y = get_compile_time_arg_val(4);

    // To enable DRISC to read initiate NOC transactions to Tensix L1,
    // we need to configure DRISC's NIU into stream mode from default NOC2AXI mode
    drisc_set_stream_mode();
    uint64_t src_noc_addr = get_noc_addr(tensix_noc_x, tensix_noc_y, tensix_l1_addr);
    noc_async_read(src_noc_addr, dram_l1_addr, sizeof(uint32_t));
    noc_async_read_barrier();
}
