// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    constexpr uint32_t base_l1_src_address = get_compile_time_arg_val(0);
    constexpr uint32_t base_pcie_dst_address = get_compile_time_arg_val(1);
    constexpr uint32_t num_16b_writes = get_compile_time_arg_val(2);

    uint64_t pcie_core_noc_encoding = uint64_t(NOC_XY_PCIE_ENCODING(PCIE_NOC_X, PCIE_NOC_Y, NOC_INDEX));

    uint32_t l1_src_address = base_l1_src_address;
    uint32_t pcie_dst_address = base_pcie_dst_address;
    for (uint32_t i = 0; i < num_16b_writes; i++) {
        uint64_t dst_noc_addr = pcie_core_noc_encoding | pcie_dst_address;
        noc_async_write(l1_src_address, dst_noc_addr, L1_ALIGNMENT);
        l1_src_address += L1_ALIGNMENT;
        pcie_dst_address += L1_ALIGNMENT;
    }
    noc_async_write_barrier();
}
