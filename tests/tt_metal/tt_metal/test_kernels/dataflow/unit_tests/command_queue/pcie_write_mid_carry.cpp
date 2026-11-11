// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "noc_nonblocking_api.h"
#include "tt_metal/impl/dispatch/kernels/cq_common.hpp"

// Walks cq_noc_async_write_with_state_any_len across a destination range that crosses a 4GB boundary and reports
// the address registers it left behind. Sending is disabled, so the synthetic destination never reaches the NOC
// and no mapping has to exist for it.
//
// NOC_RET_ADDR_LO holds the low 32 bits of the destination and NOC_RET_ADDR_MID the high 32. If the loop reprograms
// only LO, the two registers disagree once the address carries past 2^32.
void kernel_main() {
    const uint32_t dst_lo = get_compile_time_arg_val(0);
    const uint32_t dst_hi = get_compile_time_arg_val(1);
    const uint32_t pcie_xy_enc = get_compile_time_arg_val(2);
    const uint32_t total_bytes = get_compile_time_arg_val(3);
    const uint32_t result_l1_addr = get_compile_time_arg_val(4);
    const uint32_t src_l1_addr = get_compile_time_arg_val(5);

    constexpr uint32_t cmd_buf = NCRISC_WR_CMD_BUF;

    const uint64_t dst_base = (static_cast<uint64_t>(dst_hi) << 32) | static_cast<uint64_t>(dst_lo);

    // Same setup process_write_linear does: the separate-coordinate issuer programs COORDINATE, RET_ADDR_LO and
    // RET_ADDR_MID from the starting address.
    cq_noc_async_wwrite_init_state<CQ_NOC_sNDl, false, false, cmd_buf>(0, pcie_xy_enc, dst_base, 0, NOC_0);

    // set_ret_mid matches what process_write_linear passes, which is the path under test. Without it the
    // walk would leave RET_ADDR_MID at whatever init_state programmed and the check below would be vacuous.
    cq_noc_async_write_with_state_any_len<
        /*write_last_packet=*/true,
        /*update_counters=*/false,
        CQ_NOC_WAIT,
        cmd_buf,
        /*flush_last_transfer=*/false,
        CQ_NOC_send,
        /*set_ret_mid=*/true>(src_l1_addr, dst_base, total_bytes, 1, NOC_0);

    // The registers should describe the last burst the loop issued, so walk the same bursts to get its address.
    uint64_t last_burst = dst_base;
    uint32_t remaining = total_bytes;
    while (remaining > NOC_MAX_BURST_SIZE) {
        last_burst += NOC_MAX_BURST_SIZE;
        remaining -= NOC_MAX_BURST_SIZE;
    }

    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(result_l1_addr);
    result[0] = NOC_CMD_BUF_READ_REG(NOC_0, cmd_buf, NOC_RET_ADDR_MID);
    result[1] = NOC_CMD_BUF_READ_REG(NOC_0, cmd_buf, NOC_RET_ADDR_LO);
    result[2] = static_cast<uint32_t>(last_burst >> 32);
    result[3] = static_cast<uint32_t>(last_burst);

    noc_async_write_clear_pcie_state(NOC_0, cmd_buf);
}
