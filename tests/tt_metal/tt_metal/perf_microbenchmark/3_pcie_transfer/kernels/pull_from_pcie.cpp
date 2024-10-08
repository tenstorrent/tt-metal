// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    constexpr uint32_t pcie_base = get_compile_time_arg_val(0);
    constexpr uint32_t pcie_sizeB = get_compile_time_arg_val(1);
    constexpr uint32_t read_sizeB = get_compile_time_arg_val(2);
    constexpr uint32_t done_address = get_compile_time_arg_val(3);

    uint32_t pcie_read_ptr = pcie_base;

    volatile tt_l1_ptr uint32_t* done_address_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(done_address);

    uint64_t pcie_noc_xy_encoding = (uint64_t)NOC_XY_PCIE_ENCODING(PCIE_NOC_X, PCIE_NOC_Y, NOC_INDEX);
    while (done_address_ptr[0] == 0) {
        uint64_t host_src_addr = pcie_noc_xy_encoding | pcie_read_ptr;
        noc_async_read(host_src_addr, done_address, read_sizeB);
        pcie_read_ptr += read_sizeB;
        if (pcie_read_ptr > pcie_base + pcie_sizeB) {
            pcie_read_ptr = pcie_base;
        }
    }
}
