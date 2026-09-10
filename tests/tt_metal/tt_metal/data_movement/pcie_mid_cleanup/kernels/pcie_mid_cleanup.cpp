// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Regression test for NOC_TARG_ADDR_MID / NOC_RET_ADDR_MID cleanup after a PCIe-routed transaction.
//
// noc_async_read_pcie()/noc_async_write_pcie() set the MID register on their command buffer to route
// through the PCIe core, then clear it back to 0 afterward -- because the plain noc_async_read()/
// noc_async_write() no longer write that register at all, and rely on it already being 0. If the
// clear is missing or mistimed, the *next* ordinary on-chip transaction on that same command buffer
// silently misroutes, invisibly to Watcher (which validates the software address argument, not the
// live register state).
//
// This kernel does, per command buffer: one PCIe transaction, then one ordinary on-chip loopback
// transaction, and reports the post-PCIe MID register value for the host to check.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#ifdef ARCH_BLACKHOLE

void kernel_main() {
    constexpr uint32_t pcie_read_offset = get_compile_time_arg_val(0);
    constexpr uint32_t pcie_write_offset = get_compile_time_arg_val(1);
    constexpr uint32_t l1_pcie_read_scratch_addr = get_compile_time_arg_val(2);
    constexpr uint32_t l1_onchip_read_src_addr = get_compile_time_arg_val(3);
    constexpr uint32_t l1_onchip_read_dst_addr = get_compile_time_arg_val(4);
    constexpr uint32_t l1_write_src_addr = get_compile_time_arg_val(5);
    constexpr uint32_t l1_onchip_write_src_addr = get_compile_time_arg_val(6);
    constexpr uint32_t l1_onchip_write_dst_addr = get_compile_time_arg_val(7);
    constexpr uint32_t l1_mid_result_addr = get_compile_time_arg_val(8);
    constexpr uint32_t transfer_size = get_compile_time_arg_val(9);
    constexpr uint32_t packed_self_coords = get_compile_time_arg_val(10);

    const uint32_t self_x = packed_self_coords >> 16;
    const uint32_t self_y = packed_self_coords & 0xFFFF;

    const uint64_t pcie_noc_xy = uint64_t(NOC_XY_PCIE_ENCODING(PCIE_NOC_X, PCIE_NOC_Y));
    volatile tt_l1_ptr uint32_t* mid_result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_mid_result_addr);

    // --- read_cmd_buf: a PCIe read, then an ordinary on-chip (self-loopback) read ---
    const uint64_t pcie_read_addr = pcie_noc_xy | pcie_read_offset;
    noc_async_read_pcie(pcie_read_addr, l1_pcie_read_scratch_addr, transfer_size);
    noc_async_read_barrier();

    mid_result[0] = NOC_CMD_BUF_READ_REG(noc_index, read_cmd_buf, NOC_TARG_ADDR_MID);

    const uint64_t onchip_read_addr = get_noc_addr(self_x, self_y, l1_onchip_read_src_addr);
    noc_async_read(onchip_read_addr, l1_onchip_read_dst_addr, transfer_size);
    noc_async_read_barrier();

    // --- write_cmd_buf: a PCIe write, then an ordinary on-chip (self-loopback) write ---
    const uint64_t pcie_write_addr = pcie_noc_xy | pcie_write_offset;
    noc_async_write_pcie(l1_write_src_addr, pcie_write_addr, transfer_size);
    noc_async_write_barrier();

    mid_result[1] = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_MID);

    const uint64_t onchip_write_addr = get_noc_addr(self_x, self_y, l1_onchip_write_dst_addr);
    noc_async_write(l1_onchip_write_src_addr, onchip_write_addr, transfer_size);
    noc_async_write_barrier();
}

#else

// MID/PCIe-routing is Blackhole-specific; nothing to regress-test elsewhere.
void kernel_main() {}

#endif
