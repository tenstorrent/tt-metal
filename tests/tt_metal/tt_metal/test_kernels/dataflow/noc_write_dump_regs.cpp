// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Minimal dataflow kernel: noc_async_write L1→DRAM, then dump NOC0 cmd buf regs to L1.
//
// Runtime args:
//   0: dram_addr_lo      (DRAM destination addr, low 32 bits)
//   1: dram_bank_id      (DRAM bank ID for address translation)
//   2: l1_src_addr       (L1 source address for the write)
//   3: transfer_size     (bytes to transfer)
//   4: l1_dump_addr      (L1 address to write register dump)

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

// NOC0 command buffer 0 base
constexpr uint32_t NOC0_BASE = 0xFFB20000;

// All register offsets to capture
constexpr uint32_t OFFSETS[] = {
    0x00,
    0x04,
    0x08,
    0x0C,
    0x10,
    0x14,  // TARG_ADDR, RET_ADDR
    0x18,
    0x1C,
    0x20,
    0x24,
    0x28,
    0x2C,  // TAG, CTRL, LEN, etc.
    0x30,
    0x34,
    0x40,
    0x44,
    0x48,  // ACC, SEC, CMD_CTRL, NODE_ID, ENDPOINT_ID
};
constexpr uint32_t NUM_REGS = sizeof(OFFSETS) / sizeof(OFFSETS[0]);

void kernel_main() {
    uint32_t dram_addr_lo = get_arg_val<uint32_t>(0);
    uint32_t dram_bank_id = get_arg_val<uint32_t>(1);
    uint32_t l1_src_addr = get_arg_val<uint32_t>(2);
    uint32_t transfer_size = get_arg_val<uint32_t>(3);
    uint32_t l1_dump_addr = get_arg_val<uint32_t>(4);

    // Get NOC address for DRAM bank
    uint64_t dram_noc_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, dram_addr_lo);

    // --- Dump registers BEFORE the write (idle state) ---
    volatile uint32_t* dump_before = reinterpret_cast<volatile uint32_t*>(l1_dump_addr);
    for (uint32_t i = 0; i < NUM_REGS; i++) {
        dump_before[i] = *reinterpret_cast<volatile uint32_t*>(NOC0_BASE + OFFSETS[i]);
    }
    dump_before[NUM_REGS] = 0xBEF00001u;  // sentinel: before-write dump done

    // --- Issue the NOC write ---
    noc_async_write(l1_src_addr, dram_noc_addr, transfer_size);

    // --- Dump registers IMMEDIATELY after write (before barrier) ---
    volatile uint32_t* dump_during = reinterpret_cast<volatile uint32_t*>(l1_dump_addr + (NUM_REGS + 1) * 4);
    for (uint32_t i = 0; i < NUM_REGS; i++) {
        dump_during[i] = *reinterpret_cast<volatile uint32_t*>(NOC0_BASE + OFFSETS[i]);
    }
    dump_during[NUM_REGS] = 0xDEAD0002u;  // sentinel: during-write dump done

    // --- Wait for completion ---
    noc_async_write_barrier();

    // --- Dump registers AFTER barrier ---
    volatile uint32_t* dump_after = reinterpret_cast<volatile uint32_t*>(l1_dump_addr + 2 * (NUM_REGS + 1) * 4);
    for (uint32_t i = 0; i < NUM_REGS; i++) {
        dump_after[i] = *reinterpret_cast<volatile uint32_t*>(NOC0_BASE + OFFSETS[i]);
    }
    dump_after[NUM_REGS] = 0xAF7E0003u;  // sentinel: after-barrier dump done
}
