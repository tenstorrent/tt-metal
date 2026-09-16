// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exercises both PCIe paths, single transaction and batch, on each command buffer. Each path moves real
// data so the host can confirm the transfer was routed to host memory, and is followed by an ordinary
// on-chip transfer on the same command buffer so a stale MID shows up as corrupted data. The MID register
// itself is also reported for a direct check.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#ifdef ARCH_BLACKHOLE

void kernel_main() {
    constexpr uint32_t pcie_read_offset = get_compile_time_arg_val(0);
    constexpr uint32_t pcie_write_offset = get_compile_time_arg_val(1);
    constexpr uint32_t pcie_batch_write_offset = get_compile_time_arg_val(2);
    constexpr uint32_t l1_pcie_read_scratch_addr = get_compile_time_arg_val(3);
    constexpr uint32_t l1_pcie_batch_read_scratch_addr = get_compile_time_arg_val(4);
    constexpr uint32_t l1_onchip_read_src_addr = get_compile_time_arg_val(5);
    constexpr uint32_t l1_onchip_read_dst_addr = get_compile_time_arg_val(6);
    constexpr uint32_t l1_batch_onchip_read_dst_addr = get_compile_time_arg_val(7);
    constexpr uint32_t l1_write_src_addr = get_compile_time_arg_val(8);
    constexpr uint32_t l1_onchip_write_src_addr = get_compile_time_arg_val(9);
    constexpr uint32_t l1_onchip_write_dst_addr = get_compile_time_arg_val(10);
    constexpr uint32_t l1_batch_onchip_write_dst_addr = get_compile_time_arg_val(11);
    constexpr uint32_t l1_mid_result_addr = get_compile_time_arg_val(12);
    constexpr uint32_t transfer_size = get_compile_time_arg_val(13);
    constexpr uint32_t packed_self_coords = get_compile_time_arg_val(14);

    const uint32_t self_x = packed_self_coords >> 16;
    const uint32_t self_y = packed_self_coords & 0xFFFF;

    const uint64_t pcie_noc_xy = uint64_t(NOC_XY_PCIE_ENCODING(PCIE_NOC_X, PCIE_NOC_Y));
    volatile tt_l1_ptr uint32_t* mid_result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_mid_result_addr);

    // Liveness marker. The host poisons these slots, so if slot 4 does not come back as 0x5A5A5A5A then
    // this kernel body never ran and every other result below is meaningless.
    mid_result[4] = 0x5A5A5A5Au;

    const uint64_t pcie_read_addr = pcie_noc_xy | pcie_read_offset;
    const uint64_t pcie_write_addr = pcie_noc_xy | pcie_write_offset;
    const uint64_t pcie_batch_write_addr = pcie_noc_xy | pcie_batch_write_offset;

    const uint64_t onchip_read_addr = get_noc_addr(self_x, self_y, l1_onchip_read_src_addr);
    const uint64_t onchip_write_addr = get_noc_addr(self_x, self_y, l1_onchip_write_dst_addr);
    const uint64_t batch_onchip_write_addr = get_noc_addr(self_x, self_y, l1_batch_onchip_write_dst_addr);

    // read_cmd_buf, single transaction path. No explicit clear here: noc_async_read_pcie is responsible
    // for putting MID back itself, which is exactly what slot 0 checks.
    noc_async_read_pcie(pcie_read_addr, l1_pcie_read_scratch_addr, transfer_size);
    noc_async_read_barrier();
    mid_result[0] = NOC_CMD_BUF_READ_REG(noc_index, read_cmd_buf, NOC_TARG_ADDR_MID);

    noc_async_read(onchip_read_addr, l1_onchip_read_dst_addr, transfer_size);
    noc_async_read_barrier();

    // write_cmd_buf, single transaction path.
    noc_async_write_pcie(l1_write_src_addr, pcie_write_addr, transfer_size);
    noc_async_write_barrier();
    mid_result[1] = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_MID);

    noc_async_write(l1_onchip_write_src_addr, onchip_write_addr, transfer_size);
    noc_async_write_barrier();

    // read_cmd_buf, batch path. Two transfers share one setup, and the caller clears MID after the
    // barrier. Slot 2 fails if the clear is missing, and the on-chip read below fails if it came too late.
    noc_async_read_set_pcie_state(pcie_read_addr);
    noc_async_read_with_state(pcie_read_offset, l1_pcie_batch_read_scratch_addr, transfer_size);
    noc_async_read_with_state(pcie_read_offset, l1_pcie_batch_read_scratch_addr, transfer_size);
    noc_async_read_barrier();
    noc_async_read_clear_pcie_state();
    mid_result[2] = NOC_CMD_BUF_READ_REG(noc_index, read_cmd_buf, NOC_TARG_ADDR_MID);

    noc_async_read(onchip_read_addr, l1_batch_onchip_read_dst_addr, transfer_size);
    noc_async_read_barrier();

    // write_cmd_buf, batch path.
    noc_async_write_set_pcie_state(pcie_batch_write_addr);
    noc_async_write_with_state(l1_write_src_addr, pcie_batch_write_offset, transfer_size);
    noc_async_write_with_state(l1_write_src_addr, pcie_batch_write_offset, transfer_size);
    noc_async_write_barrier();
    noc_async_write_clear_pcie_state();
    mid_result[3] = NOC_CMD_BUF_READ_REG(noc_index, write_cmd_buf, NOC_RET_ADDR_MID);

    noc_async_write(l1_onchip_write_src_addr, batch_onchip_write_addr, transfer_size);
    noc_async_write_barrier();
}

#else

// MID/PCIe-routing is Blackhole-specific; nothing to regress-test elsewhere.
void kernel_main() {}

#endif
