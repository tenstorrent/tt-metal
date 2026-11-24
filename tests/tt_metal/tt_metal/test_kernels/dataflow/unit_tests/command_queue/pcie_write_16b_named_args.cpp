// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 version of pcie_write_16b.cpp (named compile-time args instead of positional ones).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t base_l1_src_address = get_arg(args::base_l1_src_address);
    constexpr uint32_t base_pcie_dst_address = get_arg(args::base_pcie_dst_address);
    constexpr uint32_t num_16b_writes = get_arg(args::num_16b_writes);

    uint64_t pcie_core_noc_encoding = uint64_t(NOC_XY_PCIE_ENCODING(PCIE_NOC_X, PCIE_NOC_Y));

    uint32_t l1_src_address = base_l1_src_address;
    uint32_t pcie_dst_address = base_pcie_dst_address;

    noc_async_write_one_packet_set_state(pcie_core_noc_encoding | pcie_dst_address, L1_ALIGNMENT);
    for (uint32_t i = 0; i < num_16b_writes; i++) {
        noc_async_write_one_packet_with_state(l1_src_address, pcie_dst_address);
        l1_src_address += L1_ALIGNMENT;
        pcie_dst_address += L1_ALIGNMENT;
    }
    noc_async_write_barrier();
    noc_async_write_clear_pcie_state();
}
