// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

void kernel_main() {
    // ========================================================================
    // Part 1: Direct CPU writes to cmd_buf registers, CPU trigger
    // Baseline test: use CPU to directly program NOC cmd_buf registers
    // ========================================================================
    DPRINT << "=== Part 1: Direct CPU writes to cmd_buf ===" << ENDL();

    // Get compile-time args for CB indices
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb = get_compile_time_arg_val(1);

    // Get L1 addresses from CBs
    uint32_t input_l1_addr = get_write_ptr(input_cb);
    uint32_t output_l1_addr = get_write_ptr(output_cb);
    constexpr uint32_t transfer_size = 256;  // bytes

    DPRINT << "input_l1:  0x" << HEX() << input_l1_addr << ENDL();
    DPRINT << "output_l1: 0x" << HEX() << output_l1_addr << ENDL();

    // Fill input buffer with random values
    volatile uint32_t* input = (volatile uint32_t*)input_l1_addr;
    volatile uint32_t* output = (volatile uint32_t*)output_l1_addr;
    uint32_t seed = 0xDEADBEEF;
    for (uint32_t i = 0; i < transfer_size / 4; i++) {
        seed = seed * 1103515245 + 12345;
        input[i] = seed;
        output[i] = 0;  // Clear output
    }

    DPRINT << "input[0]=0x" << HEX() << input[0] << " input[1]=0x" << input[1] << ENDL();
    DPRINT << "output[0]=0x" << HEX() << output[0] << " output[1]=0x" << output[1] << ENDL();

    // Setup for recursive NOC write
    uint32_t noc = static_cast<uint32_t>(noc_index);
    uint32_t buf = write_cmd_buf;
    uint32_t base = (buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT);

    // Get local NOC coordinates
    uint32_t local_noc_x = my_x[noc];
    uint32_t local_noc_y = my_y[noc];
    uint64_t local_coord = NOC_XY_ADDR(local_noc_x, local_noc_y, 0);
    uint32_t coord = (uint32_t)(local_coord >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK;

    DPRINT << "noc=" << DEC() << noc << " buf=" << buf << " coord=0x" << HEX() << coord << ENDL();

    // Build NOC control field for write (non-posted to get completion notification)
    uint32_t noc_cmd_field =
        NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_VC_STATIC | NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED;

    // Wait for cmd_buf to be ready
    while (!noc_cmd_buf_ready(noc, buf));
    DPRINT << "cmd_buf ready" << ENDL();

    // Program the cmd_buf registers using direct pointer writes
    DPRINT << "Programming cmd_buf..." << ENDL();
    volatile uint32_t* reg_targ_lo = (volatile uint32_t*)(base + NOC_TARG_ADDR_LO);
    volatile uint32_t* reg_targ_mid = (volatile uint32_t*)(base + NOC_TARG_ADDR_MID);
    volatile uint32_t* reg_ret_lo = (volatile uint32_t*)(base + NOC_RET_ADDR_LO);
    volatile uint32_t* reg_ret_mid = (volatile uint32_t*)(base + NOC_RET_ADDR_MID);
    volatile uint32_t* reg_ctrl = (volatile uint32_t*)(base + NOC_CTRL);
    volatile uint32_t* reg_len = (volatile uint32_t*)(base + NOC_AT_LEN_BE);
    volatile uint32_t* reg_cmd_ctrl = (volatile uint32_t*)(base + NOC_CMD_CTRL);

    *reg_targ_lo = input_l1_addr;  // Source L1 address
    *reg_targ_mid = coord;         // Source coordinate
    *reg_ret_lo = output_l1_addr;  // Dest L1 address
    *reg_ret_mid = coord;          // Dest coordinate (same core)
    *reg_ctrl = noc_cmd_field;     // Control field
    *reg_len = transfer_size;      // Transfer size

    DPRINT << "Regs: targ_lo=0x" << HEX() << *reg_targ_lo << " targ_mid=0x" << *reg_targ_mid << ENDL();
    DPRINT << "      ret_lo=0x" << HEX() << *reg_ret_lo << " ret_mid=0x" << *reg_ret_mid << ENDL();
    DPRINT << "      ctrl=0x" << HEX() << *reg_ctrl << " len=0x" << *reg_len << ENDL();

    // Update counters for non-posted write (so barrier knows to wait)
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;  // 1 for unicast

    // Trigger the command!
    DPRINT << "Triggering..." << ENDL();
    *reg_cmd_ctrl = NOC_CTRL_SEND_REQ;

    // Wait for NOC write to complete (non-posted write barrier)
    noc_async_write_barrier();
    DPRINT << "NOC write complete" << ENDL();

    // Verify the copy
    DPRINT << "output[0]=0x" << HEX() << output[0] << " output[1]=0x" << output[1] << ENDL();

    bool pass = true;
    for (uint32_t i = 0; i < transfer_size / 4; i++) {
        if (input[i] != output[i]) {
            pass = false;
            DPRINT << "MISMATCH at " << DEC() << i << ": 0x" << HEX() << input[i] << " != 0x" << output[i] << ENDL();
            break;
        }
    }

    if (pass) {
        DPRINT << "SUCCESS: L1-to-L1 copy worked!" << ENDL();
    } else {
        DPRINT << "FAILED!" << ENDL();
    }

    // ========================================================================
    // Part 2: CPU copy cmd[] from L1 to slave regs, CPU trigger
    // Build serialized command in L1, then use CPU to copy to slave regs
    // ========================================================================
    DPRINT << "=== Part 2: CPU copy cmd[] to slave regs ===" << ENDL();

    // Get cmd_cb for serialized command buffer
    constexpr uint32_t cmd_cb = get_compile_time_arg_val(2);
    uint32_t cmd_l1_addr = get_write_ptr(cmd_cb);
    DPRINT << "cmd_l1: 0x" << HEX() << cmd_l1_addr << ENDL();

    // Re-fill input with new random values, clear output again
    seed = 0xCAFEBABE;
    for (uint32_t i = 0; i < transfer_size / 4; i++) {
        seed = seed * 1103515245 + 12345;
        input[i] = seed;
        output[i] = 0;
    }
    DPRINT << "input[0]=0x" << HEX() << input[0] << " input[1]=0x" << input[1] << ENDL();
    DPRINT << "output[0]=0x" << HEX() << output[0] << " output[1]=0x" << output[1] << ENDL();

    // Use write_cmd_buf as slave (will execute the L1-to-L1 copy)
    // Use write_reg_cmd_buf as master (will program the slave via NOC)
    uint32_t slave_buf = write_cmd_buf;
    uint32_t master_buf = write_reg_cmd_buf;
    uint32_t slave_base = (slave_buf << NOC_CMD_BUF_OFFSET_BIT) + (noc << NOC_INSTANCE_OFFSET_BIT);

    DPRINT << "slave_buf=" << DEC() << slave_buf << " master_buf=" << master_buf << ENDL();

    // Wait for slave cmd_buf to be ready
    while (!noc_cmd_buf_ready(noc, slave_buf));
    DPRINT << "Slave cmd_buf ready" << ENDL();

    // Build serialized command in L1 (cmd_l1_addr already declared above)
    volatile uint32_t* cmd = (volatile uint32_t*)cmd_l1_addr;
    cmd[0] = input_l1_addr;   // 0x00: TARG_ADDR_LO
    cmd[1] = coord;           // 0x04: TARG_ADDR_MID
    cmd[2] = 0;               // 0x08: TARG_ADDR_HI
    cmd[3] = output_l1_addr;  // 0x0C: RET_ADDR_LO
    cmd[4] = coord;           // 0x10: RET_ADDR_MID
    cmd[5] = 0;               // 0x14: RET_ADDR_HI
    cmd[6] = 0;               // 0x18: PACKET_TAG
    cmd[7] = noc_cmd_field;   // 0x1C: CTRL
    cmd[8] = transfer_size;   // 0x20: AT_LEN_BE
    cmd[9] = 0;               // 0x24: AT_DATA

    // Copy cmd[] to slave registers via CPU
    volatile uint32_t* slave_regs = (volatile uint32_t*)(slave_base + NOC_TARG_ADDR_LO);
    for (uint32_t i = 0; i < 10; i++) {
        slave_regs[i] = cmd[i];
    }

    DPRINT << "Copied cmd to slave regs:" << ENDL();
    DPRINT << "  [0]=0x" << HEX() << slave_regs[0] << " [1]=0x" << slave_regs[1] << ENDL();
    DPRINT << "  [3]=0x" << HEX() << slave_regs[3] << " [4]=0x" << slave_regs[4] << ENDL();
    DPRINT << "  [7]=0x" << HEX() << slave_regs[7] << " [8]=0x" << slave_regs[8] << ENDL();

    // Update counters and trigger
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;

    DPRINT << "Triggering slave..." << ENDL();
    slave_regs[10] = NOC_CTRL_SEND_REQ;  // 0x28: CMD_CTRL

    // Wait for slave write to complete
    noc_async_write_barrier();
    DPRINT << "Part 2 slave write complete" << ENDL();

    // Verify the copy
    DPRINT << "output[0]=0x" << HEX() << output[0] << " output[1]=0x" << output[1] << ENDL();

    pass = true;
    for (uint32_t i = 0; i < transfer_size / 4; i++) {
        if (input[i] != output[i]) {
            pass = false;
            DPRINT << "MISMATCH at " << DEC() << i << ": 0x" << HEX() << input[i] << " != 0x" << output[i] << ENDL();
            break;
        }
    }

    if (pass) {
        DPRINT << "SUCCESS: Part 2 (CPU copy cmd[] to slave regs) worked!" << ENDL();
    } else {
        DPRINT << "FAILED: Part 2!" << ENDL();
    }

    // ========================================================================
    // Part 3: NOC inline writes (4B each) to slave regs, CPU trigger
    // Use NOC inline write to program each register one at a time
    // ========================================================================
    DPRINT << "=== Part 3: NOC inline writes to slave regs ===" << ENDL();

    // Re-fill input with new random values, clear output
    seed = 0x12345678;
    for (uint32_t i = 0; i < transfer_size / 4; i++) {
        seed = seed * 1103515245 + 12345;
        input[i] = seed;
        output[i] = 0;
    }
    DPRINT << "input[0]=0x" << HEX() << input[0] << " input[1]=0x" << input[1] << ENDL();

    // Wait for slave cmd_buf to be ready, then clear it
    while (!noc_cmd_buf_ready(noc, slave_buf));
    for (uint32_t i = 0; i < 11; i++) {
        slave_regs[i] = 0;
    }
    DPRINT << "Cleared slave regs" << ENDL();

    // Update cmd[] with new addresses (input/output already updated)
    cmd[0] = input_l1_addr;
    cmd[1] = coord;
    cmd[2] = 0;
    cmd[3] = output_l1_addr;
    cmd[4] = coord;
    cmd[5] = 0;
    cmd[6] = 0;
    cmd[7] = noc_cmd_field;
    cmd[8] = transfer_size;
    cmd[9] = 0;

    // NOC inline write: 4 bytes at a time using TARG_ADDR (not RET_ADDR!)
    DPRINT << "NOC inline writes to slave regs..." << ENDL();

    // Inline write uses TARG_ADDR for destination, AT_DATA for value, AT_LEN_BE for byte enable
    uint32_t inline_cmd = NOC_CMD_CPY | NOC_CMD_WR | NOC_CMD_WR_INLINE | NOC_CMD_VC_STATIC |
                          NOC_CMD_STATIC_VC(NOC_UNICAST_WRITE_VC) | NOC_CMD_RESP_MARKED;

    for (uint32_t i = 0; i < 10; i++) {
        uint64_t reg_noc_addr = NOC_XY_ADDR(local_noc_x, local_noc_y, slave_base + NOC_TARG_ADDR_LO + i * 4);

        while (!noc_cmd_buf_ready(noc, master_buf));
        noc_nonposted_writes_num_issued[noc] += 1;
        noc_nonposted_writes_acked[noc] += 1;

        NOC_CMD_BUF_WRITE_REG(noc, master_buf, NOC_AT_DATA, cmd[i]);                       // Data to write
        NOC_CMD_BUF_WRITE_REG(noc, master_buf, NOC_TARG_ADDR_LO, (uint32_t)reg_noc_addr);  // Dest addr
        NOC_CMD_BUF_WRITE_REG(
            noc, master_buf, NOC_TARG_ADDR_MID, (uint32_t)(reg_noc_addr >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
        NOC_CMD_BUF_WRITE_REG(noc, master_buf, NOC_CTRL, inline_cmd);
        NOC_CMD_BUF_WRITE_REG(noc, master_buf, NOC_AT_LEN_BE, 0xF);  // Byte enable: all 4 bytes
        NOC_CMD_BUF_WRITE_REG(noc, master_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);
    }

    noc_async_write_barrier();
    DPRINT << "NOC write done" << ENDL();

    // Check what got written
    DPRINT << "Slave regs after NOC:" << ENDL();
    DPRINT << "  [0]=0x" << HEX() << slave_regs[0] << " (expect 0x" << input_l1_addr << ")" << ENDL();
    DPRINT << "  [1]=0x" << HEX() << slave_regs[1] << " (expect 0x" << coord << ")" << ENDL();
    DPRINT << "  [3]=0x" << HEX() << slave_regs[3] << " (expect 0x" << output_l1_addr << ")" << ENDL();
    DPRINT << "  [7]=0x" << HEX() << slave_regs[7] << " (expect 0x" << noc_cmd_field << ")" << ENDL();
    DPRINT << "  [8]=0x" << HEX() << slave_regs[8] << " (expect 0x" << DEC() << transfer_size << ")" << ENDL();

    // CPU trigger
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;
    DPRINT << "CPU trigger..." << ENDL();
    slave_regs[10] = NOC_CTRL_SEND_REQ;

    noc_async_write_barrier();
    DPRINT << "Done" << ENDL();

    // Verify
    DPRINT << "output[0]=0x" << HEX() << output[0] << " output[1]=0x" << output[1] << ENDL();
    pass = true;
    for (uint32_t i = 0; i < transfer_size / 4; i++) {
        if (input[i] != output[i]) {
            pass = false;
            DPRINT << "MISMATCH at " << DEC() << i << ": 0x" << HEX() << input[i] << " != 0x" << output[i] << ENDL();
            break;
        }
    }

    if (pass) {
        DPRINT << "SUCCESS: Part 3 (NOC inline writes to slave regs) worked!" << ENDL();
    } else {
        DPRINT << "FAILED: Part 3!" << ENDL();
    }

    // ========================================================================
    // Part 4: NOC bulk write (40B) from L1 to slave regs, CPU trigger
    // Use single NOC write to copy all 10 registers at once
    // ========================================================================
    DPRINT << "=== Part 4: NOC bulk write to slave regs ===" << ENDL();

    // Re-fill input, clear output
    seed = 0xABCDEF01;
    for (uint32_t i = 0; i < transfer_size / 4; i++) {
        seed = seed * 1103515245 + 12345;
        input[i] = seed;
        output[i] = 0;
    }
    DPRINT << "input[0]=0x" << HEX() << input[0] << " input[1]=0x" << input[1] << ENDL();

    // Wait for slave cmd_buf to be ready, then clear it
    while (!noc_cmd_buf_ready(noc, slave_buf));
    for (uint32_t i = 0; i < 11; i++) {
        slave_regs[i] = 0;
    }
    DPRINT << "Cleared slave regs" << ENDL();

    // Update cmd[] with new values
    cmd[0] = input_l1_addr;
    cmd[1] = coord;
    cmd[2] = 0;
    cmd[3] = output_l1_addr;
    cmd[4] = coord;
    cmd[5] = 0;
    cmd[6] = 0;
    cmd[7] = noc_cmd_field;
    cmd[8] = transfer_size;
    cmd[9] = 0;

    // Single NOC write: 40 bytes from cmd[] to slave regs
    uint64_t slave_noc_addr_4 = NOC_XY_ADDR(local_noc_x, local_noc_y, slave_base + NOC_TARG_ADDR_LO);
    constexpr uint32_t cmd_size_40 = 0x28;  // 40 bytes = 10 words

    while (!noc_cmd_buf_ready(noc, master_buf));
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;

    DPRINT << "NOC write 40B from L1 to slave regs..." << ENDL();
    NOC_CMD_BUF_WRITE_REG(noc, master_buf, NOC_TARG_ADDR_LO, cmd_l1_addr);
    NOC_CMD_BUF_WRITE_REG(noc, master_buf, NOC_TARG_ADDR_MID, coord);
    NOC_CMD_BUF_WRITE_REG(noc, master_buf, NOC_RET_ADDR_LO, (uint32_t)slave_noc_addr_4);
    NOC_CMD_BUF_WRITE_REG(
        noc, master_buf, NOC_RET_ADDR_MID, (uint32_t)(slave_noc_addr_4 >> NOC_ADDR_COORD_SHIFT) & NOC_COORDINATE_MASK);
    NOC_CMD_BUF_WRITE_REG(noc, master_buf, NOC_CTRL, noc_cmd_field);
    NOC_CMD_BUF_WRITE_REG(noc, master_buf, NOC_AT_LEN_BE, cmd_size_40);
    NOC_CMD_BUF_WRITE_REG(noc, master_buf, NOC_CMD_CTRL, NOC_CTRL_SEND_REQ);

    noc_async_write_barrier();
    DPRINT << "NOC write done" << ENDL();

    // Check what got written
    DPRINT << "Slave regs after NOC bulk write:" << ENDL();
    for (uint32_t i = 0; i < 10; i++) {
        DPRINT << "  [" << DEC() << i << "]=0x" << HEX() << slave_regs[i] << " (expect 0x" << cmd[i] << ")" << ENDL();
    }

    // CPU trigger
    noc_nonposted_writes_num_issued[noc] += 1;
    noc_nonposted_writes_acked[noc] += 1;
    DPRINT << "CPU trigger..." << ENDL();
    slave_regs[10] = NOC_CTRL_SEND_REQ;

    noc_async_write_barrier();
    DPRINT << "Done" << ENDL();

    // Verify
    DPRINT << "output[0]=0x" << HEX() << output[0] << " output[1]=0x" << output[1] << ENDL();
    pass = true;
    for (uint32_t i = 0; i < transfer_size / 4; i++) {
        if (input[i] != output[i]) {
            pass = false;
            DPRINT << "MISMATCH at " << DEC() << i << ": 0x" << HEX() << input[i] << " != 0x" << output[i] << ENDL();
            break;
        }
    }

    if (pass) {
        DPRINT << "SUCCESS: Part 4 (NOC bulk write to slave regs) worked!" << ENDL();
    } else {
        DPRINT << "FAILED: Part 4!" << ENDL();
    }

    DPRINT << "=== Done ===" << ENDL();
}
