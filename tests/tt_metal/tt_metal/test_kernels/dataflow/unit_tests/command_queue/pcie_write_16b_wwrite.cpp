// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "noc_nonblocking_api.h"

void kernel_main() {
    // Compile-time arguments
    const uint32_t base_l1_src_address = get_compile_time_arg_val(0);
    const uint32_t dst_pcie_lo = get_compile_time_arg_val(1);
    const uint32_t dst_pcie_hi = get_compile_time_arg_val(2);
    const uint32_t num_16b_writes = get_compile_time_arg_val(3);
    const uint32_t pcie_xy_enc = get_compile_time_arg_val(4);

    // 64-bit PCIe destination address
    uint64_t dst_base = (static_cast<uint64_t>(dst_pcie_hi) << 32) | static_cast<uint64_t>(dst_pcie_lo);

    // Initialize write state on a known command buffer and VC
    // Use NCRISC write cmd buffer index 0 and VC0
    noc_write_init_state<0 /* cmd_buf */>(NOC_0, 0 /* vc */);

    uint32_t l1_src_address = base_l1_src_address;
    uint64_t pcie_dst_address = dst_base;
    for (uint32_t i = 0; i < num_16b_writes; i++) {
        // Stateful write with separate NOC XY coordinate encoding and 64b PCIe address
        noc_wwrite_with_state<
            DM_DEDICATED_NOC,
            0 /* cmd_buf */,
            CQ_NOC_SNDL,
            CQ_NOC_SEND,
            CQ_NOC_WAIT,
            true /* update_counter */,
            false /* posted */>(
            NOC_0, l1_src_address, pcie_xy_enc, pcie_dst_address, 16 /* bytes per write */, 1 /* ndests */);

        l1_src_address += 16;
        pcie_dst_address += 16;
    }

    // Ensure all outstanding transactions are flushed
    noc_async_write_barrier();
}
