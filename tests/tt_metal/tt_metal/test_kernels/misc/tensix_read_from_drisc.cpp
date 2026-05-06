// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Reads from DRISC L1 into Tensix L1

#include "api/dataflow/dataflow_api.h"
#include "noc_nonblocking_api.h"

void kernel_main() {
    constexpr uint32_t tensix_dst_addr = get_compile_time_arg_val(0);
    constexpr uint32_t drisc_l1_src_addr_low = get_compile_time_arg_val(1);
    constexpr uint32_t drisc_l1_src_addr_high = get_compile_time_arg_val(2);
    constexpr uint32_t drisc_noc_x = get_compile_time_arg_val(3);
    constexpr uint32_t drisc_noc_y = get_compile_time_arg_val(4);

    uint64_t drisc_l1_src_addr = (((uint64_t)drisc_l1_src_addr_high << 32) | drisc_l1_src_addr_low);
    uint32_t drisc_src_coord = NOC_XY_COORD(drisc_noc_x, drisc_noc_y);
    noc_read_init_state<BRISC_RD_CMD_BUF>(NOC_INDEX);
    noc_read_with_state<DM_DEDICATED_NOC, BRISC_RD_CMD_BUF, CQ_NOC_SNDL>(
        NOC_INDEX, drisc_src_coord, drisc_l1_src_addr, tensix_dst_addr, sizeof(uint32_t));
    noc_async_read_barrier();
}
