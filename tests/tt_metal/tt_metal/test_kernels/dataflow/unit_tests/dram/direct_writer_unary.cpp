// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// NOC register offsets (relative to NOC_REGS_START_ADDR)
#define NOC_TARG_ADDR_LO_OFFSET 0x0
#define NOC_TARG_ADDR_MID_OFFSET 0x4
#define NOC_TARG_ADDR_HI_OFFSET 0x8
#define NOC_RET_ADDR_LO_OFFSET 0xC
#define NOC_RET_ADDR_MID_OFFSET 0x10
#define NOC_RET_ADDR_HI_OFFSET 0x14
#define NOC_PACKET_TAG_OFFSET 0x18
#define NOC_CTRL_OFFSET 0x1C
#define NOC_AT_LEN_BE_OFFSET 0x20
#define NOC_AT_LEN_BE_1_OFFSET 0x24
#define NOC_AT_DATA_OFFSET 0x28
#define NOC_BRCST_EXCLUDE_OFFSET 0x2C
#define NOC_L1_ACC_AT_INSTRN_OFFSET 0x30
#define NOC_SEC_CTRL_OFFSET 0x34
#define NOC_CMD_CTRL_OFFSET 0x40

// Helper function to serialize NOC write command to memory as a CONTIGUOUS array
// This fills ALL register offsets (0x0 to 0x40) so it can be copied directly
// to the NOC command buffer register space as a contiguous block.
//
// Register values match what noc_init() and ncrisc_noc_fast_write() program:
FORCE_INLINE void serialize_noc_write_to_memory(
    uint32_t* mem_buf,           // Pointer to memory buffer (must be at least 17 words)
    uint32_t src_local_l1_addr,  // Source address in local L1
    uint64_t dst_noc_addr,       // Destination NOC address
    uint32_t size,               // Transfer size in bytes
    uint32_t local_noc_x,        // Local core NOC X coordinate
    uint32_t local_noc_y,        // Local core NOC Y coordinate
    uint32_t vc = NOC_UNICAST_WRITE_VC,
    bool posted = true) {
    // Build the NOC control field (same as ncrisc_noc_fast_write)
    uint32_t noc_cmd_field =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(vc) | (posted ? 0 : NOC_CMD_RESP_MARKED);

    // Extract destination address components
    uint32_t dest_addr_lo = (uint32_t)dst_noc_addr;
    uint32_t dest_addr_mid = (uint32_t)(dst_noc_addr >> 32) & NOC_PCIE_MASK;
    uint32_t dest_addr_hi = (uint32_t)(dst_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;

    // Build local source coordinates (pre-initialized by noc_init)
    uint64_t xy_local_addr = NOC_XY_ADDR(local_noc_x, local_noc_y, 0);
    uint32_t src_coord = (uint32_t)(xy_local_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;

    // Serialize ALL registers as contiguous array from offset 0x0 to 0x40
    // This allows copying the entire buffer to NOC register space in one operation

    mem_buf[NOC_TARG_ADDR_LO_OFFSET / 4] = src_local_l1_addr;  // 0x0: Source L1 address
    mem_buf[NOC_TARG_ADDR_MID_OFFSET / 4] = 0x0;               // 0x4: Source MID (always 0 for L1)
    mem_buf[NOC_TARG_ADDR_HI_OFFSET / 4] = src_coord;          // 0x8: Source coordinates (pre-init by noc_init)

    mem_buf[NOC_RET_ADDR_LO_OFFSET / 4] = dest_addr_lo;    // 0xC: Dest address low
    mem_buf[NOC_RET_ADDR_MID_OFFSET / 4] = dest_addr_mid;  // 0x10: Dest address mid (PCIE)
    mem_buf[NOC_RET_ADDR_HI_OFFSET / 4] = dest_addr_hi;    // 0x14: Dest coordinates

    mem_buf[NOC_PACKET_TAG_OFFSET / 4] = 0;        // 0x18: Transaction ID (0 = no trid)
    mem_buf[NOC_CTRL_OFFSET / 4] = noc_cmd_field;  // 0x1C: Control field

    mem_buf[NOC_AT_LEN_BE_OFFSET / 4] = size;      // 0x20: Transfer size
    mem_buf[NOC_AT_LEN_BE_1_OFFSET / 4] = 0;       // 0x24: Atomic len (0 = no atomic)
    mem_buf[NOC_AT_DATA_OFFSET / 4] = 0;           // 0x28: Inline data (0 = no inline write)
    mem_buf[NOC_BRCST_EXCLUDE_OFFSET / 4] = 0;     // 0x2C: Multicast exclude mask
    mem_buf[NOC_L1_ACC_AT_INSTRN_OFFSET / 4] = 0;  // 0x30: L1 access control

    mem_buf[NOC_CMD_CTRL_OFFSET / 4] = NOC_CTRL_SEND_REQ;  // 0x40: Trigger command
    // 0x38 and 0x3C are unused/reserved

    mem_buf[(NOC_CMD_CTRL_OFFSET / 4) + 1] = 0x11111111;
    mem_buf[(NOC_CMD_CTRL_OFFSET / 4) + 2] = 0x12222222;
    mem_buf[(NOC_CMD_CTRL_OFFSET / 4) + 3] = 0x33333333;
    mem_buf[(NOC_CMD_CTRL_OFFSET / 4) + 4] = 0x44444444;
    mem_buf[(NOC_CMD_CTRL_OFFSET / 4) + 5] = 0x55555555;
    mem_buf[(NOC_CMD_CTRL_OFFSET / 4) + 6] = 0x66666666;
    mem_buf[(NOC_CMD_CTRL_OFFSET / 4) + 7] = 0x77777777;
    mem_buf[(NOC_CMD_CTRL_OFFSET / 4) + 8] = 0x88888888;
    mem_buf[(NOC_CMD_CTRL_OFFSET / 4) + 9] = 0x99999999;

    DPRINT << "Serialized NOC write (contiguous):" << ENDL();
    DPRINT << "  [0x00] TARG_ADDR_LO: 0x" << HEX() << src_local_l1_addr << ENDL();
    DPRINT << "  [0x04] TARG_ADDR_MID: 0x0" << ENDL();
    DPRINT << "  [0x08] TARG_COORD: 0x" << HEX() << src_coord << ENDL();
    DPRINT << "  [0x1C] CTRL: 0x" << HEX() << noc_cmd_field << ENDL();
    DPRINT << "  [0x20] AT_LEN_BE: " << DEC() << size << ENDL();
    DPRINT << "  [0x24] AT_LEN_BE_1: 0 (no atomic)" << ENDL();
    DPRINT << "  [0x28] AT_DATA: 0 (no inline)" << ENDL();
}

void kernel_main() {
    const uint32_t cb_id = get_compile_time_arg_val(0);
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t dst_bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t noc_cmd_buffer_addr = get_arg_val<uint32_t>(3);    // 1KB L1 buffer for serialization
    uint32_t noc_cmd_buffer_addr_2 = get_arg_val<uint32_t>(4);  // Second 1KB L1 buffer

    // single-tile ublocks
    uint32_t ublock_size_bytes = get_tile_size(cb_id);
    uint32_t ublock_size_tiles = 1;

    // Use the L1 buffer passed from host for NOC command serialization
    uint32_t* noc_cmd_buffer = reinterpret_cast<uint32_t*>(noc_cmd_buffer_addr);
    uint32_t* noc_cmd_buffer_2 = reinterpret_cast<uint32_t*>(noc_cmd_buffer_addr_2);

    DPRINT << "Starting NOC write experiment" << ENDL();
    DPRINT << "  num_tiles: " << num_tiles << ENDL();
    DPRINT << "  noc_cmd_buffer @ 0x" << HEX() << noc_cmd_buffer_addr << ENDL();
    DPRINT << "  noc_cmd_buffer_2 @ 0x" << HEX() << noc_cmd_buffer_addr_2 << ENDL();

    // Get local NOC coordinates for this core (using NOC 0)
    uint32_t local_noc_x = my_x[0];
    uint32_t local_noc_y = my_y[0];

    // Experiment: Serialize the NOC write command to memory instead of executing it
    serialize_noc_write_to_memory(noc_cmd_buffer, 0x88888, 0x99999, ublock_size_bytes, local_noc_x, local_noc_y);
    noc_async_write(noc_cmd_buffer_addr, get_noc_addr(noc_cmd_buffer_addr_2), NOC_CMD_CTRL_OFFSET + 4);
    noc_async_write_barrier();

    for (uint32_t i = 0; i < (NOC_CMD_CTRL_OFFSET / 4) + 10; i++) {
        DPRINT << "noc_cmd_buffer[" << i << "], dest_buf = 0x" << HEX() << noc_cmd_buffer[i] << ", "
               << noc_cmd_buffer_2[i] << ENDL();
        // DPRINT << "noc_cmd_buffer_2[" << i << "] = 0x" << HEX() <<  << ENDL();
        if (i == NOC_CMD_CTRL_OFFSET / 4) {
            DPRINT << "-------\n";
        }
    }

    // Now it's time for business
    // first setup the noc command we actually want to execute
    // -> we want to copy from one part of the buffer to another. Store the register programming
    //    sequence at noc_cmd_buffer
    serialize_noc_write_to_memory(
        noc_cmd_buffer,
        get_noc_addr(noc_cmd_buffer_addr_2),
        get_noc_addr(noc_cmd_buffer_addr_2 + 256),
        NOC_CMD_CTRL_OFFSET + 4,
        local_noc_x,
        local_noc_y);

    // now that we have the register programming setup in L1, we want to send that to the noc cmd buf, via another noc
    // cmd buf
    auto get_cmd_buf_reg_base_addr = [](uint32_t noc, uint32_t cmd_buf) {
        uint32_t offset = (cmd_buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT) +
                          0;  // want the first register address
        return offset;
    };
    uint32_t trid = 0;
    auto slave_cmd_buf = write_cmd_buf;
    auto wr_cmd_buf_regs_base_addr = get_cmd_buf_reg_base_addr(noc_index, slave_cmd_buf);
    constexpr size_t size_regs_in_bytes = (NOC_CMD_CTRL_OFFSET / 4) + 1;
    auto master_cmd_buf = write_reg_cmd_buf;
    // waiting for the slave cmd buf to be ready
    while (!noc_cmd_buf_ready(noc_index, slave_cmd_buf));
    noc_async_write_one_packet_with_trid(
        noc_cmd_buffer_addr,
        get_noc_addr(wr_cmd_buf_regs_base_addr),
        size_regs_in_bytes,
        trid,
        master_cmd_buf,
        noc_index,
        NOC_UNICAST_WRITE_VC);

    DPRINT << "ISSUED RECURSIVE NOC WRITE!!!" << ENDL();
    while (!ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, trid));

    DPRINT << "RECURSIVE COPY ISSUED!!!" << ENDL();
    // TODO, we'll need to add some code to just increment the credits so that barriers can implement properly
    for (uint32_t i = (256 / 4); i < (256 / 4) + (NOC_CMD_CTRL_OFFSET / 4) + 10; i++) {
        DPRINT << "noc_cmd_buffer[" << i << "], dest_buf = 0x" << HEX() << noc_cmd_buffer[i] << ", "
               << noc_cmd_buffer_2[i] << ENDL();
        // DPRINT << "noc_cmd_buffer_2[" << i << "] = 0x" << HEX() <<  << ENDL();
    }

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
         uint64_t dst_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, dst_addr);

        cb_wait_front(cb_id, ublock_size_tiles);
        uint32_t l1_read_addr = get_read_ptr(cb_id);

        // Experiment: Serialize the NOC write command to memory instead of executing it
        // serialize_noc_write_to_memory(
        //     noc_cmd_buffer, l1_read_addr, dst_noc_addr, ublock_size_bytes,
        //     local_noc_x, local_noc_y);

        // Also do the actual NOC write so the test still passes
        noc_async_write(l1_read_addr, dst_noc_addr, ublock_size_bytes);
        noc_async_write_barrier();

        cb_pop_front(cb_id, ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }

    DPRINT << "NOC write experiment complete!" << ENDL();
}
