// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "experimental/core_local_mem.h"
#include "experimental/endpoints.h"
#include "api/debug/dprint.h"

/**
 * NOC APIs are prefixed w/ "ncrisc" (legacy name) but there's nothing NCRISC specific, they can be used on BRISC or
 * other RISCs Any two RISC processors cannot use the same CMD_BUF non_blocking APIs shouldn't be mixed with slow noc.h
 * APIs explicit flushes need to be used since the calls are non-blocking
 * */

// Debug helper to print NOC read status
inline void debug_print_noc_read_status(const char* label, uint32_t noc) {
    // Read the register that tracks responses received from DRAM
    uint32_t reg_val = NOC_STATUS_READ_REG(noc, NIU_MST_RD_RESP_RECEIVED);
    // Read the counter that tracks reads issued
    uint32_t issued = noc_reads_num_issued[noc];
    DPRINT << label << " NOC[" << noc << "]: reg(resp_received)=" << reg_val << " issued=" << issued << ENDL();
}

// Debug helper to print NOC write status
inline void debug_print_noc_write_status(const char* label, uint32_t noc) {
    // Read the software counters that track writes issued and acked
    uint32_t issued = noc_nonposted_writes_num_issued[noc];
    uint32_t acked = noc_nonposted_writes_acked[noc];
    DPRINT << label << " NOC[" << noc << "]: writes_issued=" << issued << " writes_acked=" << acked << ENDL();
}

void kernel_main() {
    // __builtin_riscv_ttrocc_llk_intf_write(0, 0);
    // __builtin_riscv_ttrocc_cmdbuf_reset(0);  // DO NOT reset here - noc_init() already configured this buf
    // __builtin_riscv_ttrocc_cmdbuf_reset(1);

    std::uint32_t l1_buffer_addr = get_arg_val<uint32_t>(0);

    std::uint32_t dram_buffer_src_addr = get_arg_val<uint32_t>(1);
    std::uint32_t dram_src_bank_id = get_arg_val<uint32_t>(2);

    std::uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(3);
    std::uint32_t dram_dst_bank_id = get_arg_val<uint32_t>(4);

    std::uint32_t dram_buffer_size = get_arg_val<uint32_t>(5);

    // Debug prints to verify addresses
    DPRINT << "L1 addr: " << l1_buffer_addr << ENDL();
    DPRINT << "SRC DRAM addr: " << dram_buffer_src_addr << " bank: " << dram_src_bank_id << ENDL();
    DPRINT << "DST DRAM addr: " << dram_buffer_dst_addr << " bank: " << dram_dst_bank_id << ENDL();
    DPRINT << "Size: " << dram_buffer_size << ENDL();

    // Print DRAM bank to NOC XY mappings for all banks
    DPRINT << "=== DRAM Bank NOC Mappings (NOC 0) ===" << ENDL();
    for (uint32_t bank = 0; bank < NUM_DRAM_BANKS; bank++) {
        uint16_t noc_xy = dram_bank_to_noc_xy[0][bank];
        uint32_t x = noc_xy & 0xFF;
        uint32_t y = (noc_xy >> 8) & 0xFF;
        DPRINT << "Bank " << bank << ": x=" << x << " y=" << y << " (raw=0x" << HEX() << noc_xy << ")" << ENDL();
    }

#if defined(SIGNAL_COMPLETION_TO_DISPATCHER)
    // We will assert later. This kernel will hang.
    // Need to signal completion to dispatcher before hanging so that
    // Dispatcher Kernel is able to finish.
    // Device Close () requires fast dispatch kernels to finish.
#if defined(COMPILE_FOR_ERISC)
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(eth_l1_mem::address_map::ERISC_MEM_MAILBOX_BASE);
#else
    tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE);
#endif
    uint64_t dispatch_addr = NOC_XY_ADDR(
        NOC_X(mailboxes->go_message.master_x),
        NOC_Y(mailboxes->go_message.master_y),
        DISPATCH_MESSAGE_ADDR + NOC_STREAM_REG_SPACE_SIZE * mailboxes->go_message.dispatch_message_offset);
    noc_fast_write_dw_inline<DM_DEDICATED_NOC>(
        noc_index,
        NCRISC_AT_CMD_BUF,
        1 << REMOTE_DEST_BUF_WORDS_FREE_INC,
        dispatch_addr,
        0xF,  // byte-enable
        NOC_UNICAST_WRITE_VC,
        false,  // mcast
        true    // posted
    );
#endif

    // ============ HARDCODED DRAM COORDINATES FOR TESTING ============
    // Set to 1 to use hardcoded coordinates, 0 to use AllocatorBank API
    constexpr bool USE_HARDCODED_COORDS = true;

    // Hardcoded DRAM NOC coordinates (modify these for your architecture)
    // These are typical values - check your SOC descriptor for correct values
    constexpr uint32_t DRAM_NOC_X = 0;
    constexpr uint32_t DRAM_NOC_Y = 0;

    if constexpr (USE_HARDCODED_COORDS) {
        // Using hardcoded coordinates with low-level API
        uint64_t src_noc_addr = get_noc_addr(DRAM_NOC_X, DRAM_NOC_Y, dram_buffer_src_addr, noc_index);
        uint64_t dst_noc_addr = get_noc_addr(DRAM_NOC_X, DRAM_NOC_Y, dram_buffer_dst_addr, noc_index);

        DPRINT << "HARDCODED: src_noc_addr=0x" << HEX() << (uint32_t)(src_noc_addr >> 32) << HEX()
               << (uint32_t)src_noc_addr << ENDL();
        DPRINT << "HARDCODED: dst_noc_addr=0x" << HEX() << (uint32_t)(dst_noc_addr >> 32) << HEX()
               << (uint32_t)dst_noc_addr << ENDL();

        // DRAM NOC src address - READ from DRAM to L1
        debug_print_noc_read_status("BEFORE READ", noc_index);
        DPRINT << "ISSUE READ FROM SRC DRAM 1" << ENDL();
        noc_async_read(src_noc_addr, l1_buffer_addr, dram_buffer_size);
        DPRINT << "ISSUE READ FROM SRC DRAM 2" << ENDL();
        debug_print_noc_read_status("AFTER READ ISSUE", noc_index);
        DPRINT << "Waiting for read barrier..." << ENDL();
        noc_async_read_barrier();
        debug_print_noc_read_status("AFTER READ BARRIER", noc_index);
        DPRINT << "ISSUE READ FROM SRC DRAM 3" << ENDL();

        // DRAM NOC dst address - WRITE from L1 to DRAM
        debug_print_noc_write_status("BEFORE WRITE", noc_index);
        DPRINT << "ISSUE WRITE TO DST DRAM 1" << ENDL();
        noc_async_write(l1_buffer_addr, dst_noc_addr, dram_buffer_size);
        DPRINT << "ISSUE WRITE TO DST DRAM 2" << ENDL();
        debug_print_noc_write_status("AFTER WRITE ISSUE", noc_index);
        DPRINT << "Waiting for write barrier..." << ENDL();
        DPRINT << "  scmdbuf_tr_ack=" << __builtin_riscv_ttrocc_scmdbuf_tr_ack() << ENDL();
        noc_async_write_barrier();
        debug_print_noc_write_status("AFTER WRITE BARRIER", noc_index);
        DPRINT << "ISSUE WRITE TO DST DRAM 3" << ENDL();
    } else {
        // Using experimental AllocatorBank API
        experimental::Noc noc;
        experimental::CoreLocalMem<std::uint32_t> l1_buffer(l1_buffer_addr);
        constexpr experimental::AllocatorBankType bank_type = experimental::AllocatorBankType::DRAM;
        experimental::AllocatorBank<bank_type> src_dram;
        experimental::AllocatorBank<bank_type> dst_dram;

        // DRAM NOC src address
        DPRINT << "ISSUE READ FROM SRC DRAM 1" << ENDL();
        noc.async_read(
            src_dram, l1_buffer, dram_buffer_size, {.bank_id = dram_src_bank_id, .addr = dram_buffer_src_addr}, {});
        DPRINT << "ISSUE READ FROM SRC DRAM 2" << ENDL();
        noc.async_read_barrier();
        DPRINT << "ISSUE READ FROM SRC DRAM 3" << ENDL();

        // DRAM NOC dst address
        DPRINT << "ISSUE WRITE TO DST DRAM 1" << ENDL();
        noc.async_write(
            l1_buffer, dst_dram, dram_buffer_size, {}, {.bank_id = dram_dst_bank_id, .addr = dram_buffer_dst_addr});
        DPRINT << "ISSUE WRITE TO DST DRAM 2" << ENDL();
        noc.async_write_barrier();
        DPRINT << "ISSUE WRITE TO DST DRAM 3" << ENDL();
    }
}
