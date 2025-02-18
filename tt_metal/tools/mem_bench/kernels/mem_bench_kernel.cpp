// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include <cstdint>
#include <cstdlib>
#include "noc_overlay_parameters.h"

//
// Test Kernel for mem_bench
//
// Performs PCIe reads and/or writes
//

// reader kernel
constexpr uint32_t my_rd_dst_addr = get_compile_time_arg_val(0);  // L1
constexpr uint32_t pcie_rd_base = get_compile_time_arg_val(1);
constexpr uint32_t pcie_rd_size = get_compile_time_arg_val(2);
constexpr uint32_t pcie_rd_end = pcie_rd_base + pcie_rd_size;
constexpr uint32_t pcie_rd_transfer_size = get_compile_time_arg_val(3);
constexpr uint32_t my_bytes_rd_addr = get_compile_time_arg_val(4);

// writer kernel
constexpr uint32_t reserved_0 = get_compile_time_arg_val(5);
constexpr uint32_t pcie_wr_base = get_compile_time_arg_val(6);
constexpr uint32_t pcie_wr_size = get_compile_time_arg_val(7);
constexpr uint32_t pcie_wr_end = pcie_wr_base + pcie_wr_size;
constexpr uint32_t pcie_wr_transfer_size = get_compile_time_arg_val(8);
constexpr uint32_t my_bytes_wr_addr = get_compile_time_arg_val(9);

// common to both
constexpr uint32_t my_total_work = get_compile_time_arg_val(10);  // Total bytes to read+write
constexpr uint32_t my_cycles_addr = get_compile_time_arg_val(11);

static_assert(my_bytes_rd_addr && my_bytes_wr_addr, "Must provide addresses for my_bytes_rd/wr_addr");
static_assert(my_cycles_addr, "Must provide L1 address for cycles elapsed");

uint64_t get_cycles() {
    uint32_t timestamp_low = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_L);
    uint32_t timestamp_high = reg_read(RISCV_DEBUG_REG_WALL_CLOCK_H);
    return (((uint64_t)timestamp_high) << 32) | timestamp_low;
}

void kernel_main() {
    auto my_cycles = reinterpret_cast<volatile uint32_t*>(my_cycles_addr);
    auto my_bytes_read = reinterpret_cast<volatile uint32_t*>(my_bytes_rd_addr);
    auto my_bytes_written = reinterpret_cast<volatile uint32_t*>(my_bytes_wr_addr);

    my_bytes_read[0] = 0;
    my_bytes_written[0] = 0;
    my_cycles[0] = 0;

    uint64_t pcie_noc_xy_encoding = (uint64_t)NOC_XY_PCIE_ENCODING(PCIE_NOC_X, PCIE_NOC_Y);
    uint32_t rd_ptr = pcie_rd_base;
    uint32_t wr_ptr = pcie_wr_base;

    const auto start = get_cycles();

    uint32_t total_bytes_read = 0;
    uint32_t total_bytes_written = 0;
    while (total_bytes_read + total_bytes_written < my_total_work) {
        if constexpr (my_rd_dst_addr) {
            uint64_t host_src_addr = pcie_noc_xy_encoding | rd_ptr;
            noc_async_read(host_src_addr, my_rd_dst_addr, pcie_rd_transfer_size);
            rd_ptr += pcie_rd_transfer_size;
            total_bytes_read += pcie_rd_transfer_size;
            if (rd_ptr >= pcie_rd_end) {
                rd_ptr = pcie_rd_base;
            }
        }
        if constexpr (pcie_wr_size) {
            uint64_t host_dst_addr = pcie_noc_xy_encoding | wr_ptr;
            noc_async_write(
                wr_ptr,  // Any data
                host_dst_addr,
                pcie_wr_transfer_size);
            wr_ptr += pcie_wr_transfer_size;
            total_bytes_written += pcie_wr_transfer_size;
            if (wr_ptr >= pcie_wr_end) {
                wr_ptr = pcie_wr_base;
            }
        }
    }

    if constexpr (my_rd_dst_addr) {
        noc_async_read_barrier();
    }
    if constexpr (pcie_wr_size) {
        noc_async_write_barrier();
    }

    auto end = get_cycles();
    my_cycles[0] = end - start;
    my_bytes_read[0] = total_bytes_read;
    my_bytes_written[0] = total_bytes_written;
}
