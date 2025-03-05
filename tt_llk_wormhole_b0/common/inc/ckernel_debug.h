// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"

// Debug bus and register array dump

namespace ckernel
{

struct dbg_cfgreg
{
    constexpr static uint32_t THREAD_0_CFG    = 0; // Thread 0 config
    constexpr static uint32_t THREAD_1_CFG    = 1; // Thread 1 config
    constexpr static uint32_t THREAD_2_CFG    = 2; // Thread 2 config
    constexpr static uint32_t HW_CFG_0        = 3; // Hardware config state 0
    constexpr static uint32_t HW_CFG_1        = 4; // Hardware config state 1
    constexpr static uint32_t HW_CFG_SIZE     = 187;
    constexpr static uint32_t THREAD_CFG_SIZE = THD_STATE_SIZE;
};

struct dbg_array_id
{
    constexpr static uint32_t SRCA = 0; // SrcA last used bank
    constexpr static uint32_t SRCB = 1; // SrcB last used bank
    constexpr static uint32_t DEST = 2; // Dest acc
};

struct dbg_daisy_id
{
    constexpr static uint32_t INSTR_ISSUE_0 = 4;
    constexpr static uint32_t INSTR_ISSUE_1 = 5;
    constexpr static uint32_t INSTR_ISSUE_2 = 6;
};

typedef struct
{
    uint32_t sig_sel    : 16;
    uint32_t daisy_sel  : 8;
    uint32_t rd_sel     : 4;
    uint32_t reserved_0 : 1;
    uint32_t en         : 1;
    uint32_t reserved_1 : 2;
} dbg_bus_cntl_t;

typedef union
{
    uint32_t val;
    dbg_bus_cntl_t f;
} dbg_bus_cntl_u;

;

typedef struct
{
    uint32_t en       : 1;
    uint32_t reserved : 31;
} dbg_array_rd_en_t;

typedef union
{
    uint32_t val;
    dbg_array_rd_en_t f;
} dbg_array_rd_en_u;

typedef struct
{
    uint32_t row_addr    : 12;
    uint32_t row_32b_sel : 4;
    uint32_t array_id    : 3;
    uint32_t bank_id     : 1;
    uint32_t reserved    : 12;
} dbg_array_rd_cmd_t;

typedef union
{
    uint32_t val;
    dbg_array_rd_cmd_t f;
} dbg_array_rd_cmd_u;

typedef struct
{
    uint32_t unp      : 2;
    uint32_t pack     : 4;
    uint32_t reserved : 26;
} dbg_soft_reset_t;

typedef union
{
    uint32_t val;
    dbg_soft_reset_t f;
} dbg_soft_reset_u;

;

template <ThreadId thread_id>
inline void dbg_thread_halt()
{
    static_assert(
        (thread_id == ThreadId::MathThreadId) || (thread_id == ThreadId::UnpackThreadId) || (thread_id == ThreadId::PackThreadId),
        "Invalid thread id set in dbg_wait_for_thread_idle(...)");

    if constexpr (thread_id == ThreadId::UnpackThreadId)
    {
        // Wait for all instructions on the running thread to complete
        tensix_sync();
        // Notify math thread that unpack thread is idle
        mailbox_write(ThreadId::MathThreadId, 1);
        // Wait for math thread to complete debug dump
        volatile uint32_t temp = mailbox_read(ThreadId::MathThreadId);
    }
    else if constexpr (thread_id == ThreadId::MathThreadId)
    {
        // Wait for all instructions on the running thread to complete
        tensix_sync();
        // Wait for unpack thread to complete
        volatile uint32_t temp = mailbox_read(ThreadId::UnpackThreadId);
        // Wait for previous packs to finish
        while (semaphore_read(semaphore::MATH_PACK) > 0)
        {
        };
    }
}

template <ThreadId thread_id>
inline void dbg_thread_unhalt()
{
    static_assert(
        (thread_id == ThreadId::MathThreadId) || (thread_id == ThreadId::UnpackThreadId) || (thread_id == ThreadId::PackThreadId),
        "Invalid thread id set in dbg_wait_for_thread_idle(...)");

    if constexpr (thread_id == ThreadId::MathThreadId)
    {
        // Reset pack 0 (workaround)
        dbg_soft_reset_u dbg_soft_reset;
        dbg_soft_reset.val    = 0;
        dbg_soft_reset.f.pack = 1;
        reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, dbg_soft_reset.val);
        wait(5);
        dbg_soft_reset.val = 0;
        reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, dbg_soft_reset.val);

        // Wait for all instructions to complete
        tensix_sync();

        // Unhalt unpack thread
        mailbox_write(ThreadId::UnpackThreadId, 1);
    }
}

inline void dbg_get_array_row(const uint32_t array_id, const uint32_t row_addr, uint32_t *rd_data)
{
    // Dest offset is added to row_addr to dump currently used half of the dest accumulator (SyncHalf dest mode)
    std::uint32_t dest_offset = 0;
    if (array_id == dbg_array_id::DEST)
    {
        dest_offset = (dest_offset_id == 1) ? DEST_REGISTER_HALF_SIZE : 0;
    }

    if (array_id == dbg_array_id::SRCA)
    {
        // Save dest row
        // WWhen SrcA array is selected we need to copy row from src register into dest to be able to dump data
        // Dump from SrcA array is not supported
        // Save dest row to SFPU register
        // Move SrcA into dest row
        // Dump dest row
        // Restore dest row
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 0},
            .dest = {.incr = 0, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_0);

        // Clear counters
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
        TTI_SFPLOAD(p_sfpu::LREG3, 0, 0, 0); // Save dest addr 0 (even cols) to LREG_3
        TTI_SFPLOAD(p_sfpu::LREG3, 0, 0, 2); // Save dest addr 0 (odd cols)  to LREG_3

        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SFPU1);

        // Move to the last used bank
        TTI_CLEARDVALID(0b01, 0);

        // Copy single row from SrcA[row_addr] to dest location 0
        TT_MOVDBGA2D(0, row_addr, 0, p_mova2d::MOV_1_ROW, 0);

        // Wait for TT instructions to complete
        tensix_sync();
    }
    else if (array_id == dbg_array_id::SRCB)
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 0},
            .dest = {.incr = 0, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_0);

        // Clear counters
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);

        // Move to the last used bank
        TTI_SETDVALID(0b10);
        TTI_CLEARDVALID(0b10, 0);

        // Latch debug values
        TTI_SETDVALID(0b10);
        TTI_SHIFTXB(ADDR_MOD_0, 0, row_addr >> 1);
        TTI_CLEARDVALID(0b10, 0);

        // Wait for TT instructions to complete
        tensix_sync();

        /*
        // Get last used bank id via debug bus
        dbg_bus_cntl_u dbg_bus_cntl;
        dbg_bus_cntl.val = 0;
        dbg_bus_cntl.f.rd_sel = 6; // Read bits 111:96, 110 (write_math_id_reg)
        dbg_bus_cntl.f.sig_sel = 0x4<<1;
        dbg_bus_cntl.f.daisy_sel = dbg_daisy_id::INSTR_ISSUE_2;
        dbg_bus_cntl.f.en = 1;
        reg_write(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG, dbg_bus_cntl.val);
        wait (5); // Wait for value to get stable
        srcb_bank_id = ((reg_read(RISCV_DEBUG_REG_DBG_RD_DATA)) & 0x4000) ? 0 : 1;

        // Disable debug bus
        dbg_bus_cntl.val = 0;
        reg_write(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG, dbg_bus_cntl.val);
        */
    }

    // Get actual row address and array id used in hw
    std::uint32_t hw_row_addr = (array_id == dbg_array_id::SRCA) ? 0 : ((array_id == dbg_array_id::DEST) ? dest_offset + row_addr : row_addr & 0x1);

    std::uint32_t hw_array_id = (array_id == dbg_array_id::SRCA) ? dbg_array_id::DEST : array_id;

    std::uint32_t hw_bank_id = 0;

    bool sel_datums_15_8 = hw_array_id != dbg_array_id::DEST;

    dbg_array_rd_en_u dbg_array_rd_en;
    dbg_array_rd_en.val  = 0;
    dbg_array_rd_en.f.en = 0x1;
    reg_write(RISCV_DEBUG_REG_DBG_ARRAY_RD_EN, dbg_array_rd_en.val);

    dbg_array_rd_cmd_u dbg_array_rd_cmd;
    dbg_array_rd_cmd.val        = 0;
    dbg_array_rd_cmd.f.array_id = hw_array_id;
    dbg_array_rd_cmd.f.bank_id  = hw_bank_id;

    for (uint32_t i = 0; i < 8; i++)
    {
        dbg_array_rd_cmd.f.row_addr    = sel_datums_15_8 ? (hw_row_addr | ((i >= 4) ? (1 << 6) : 0)) : hw_row_addr;
        dbg_array_rd_cmd.f.row_32b_sel = i;
        reg_write(RISCV_DEBUG_REG_DBG_ARRAY_RD_CMD, dbg_array_rd_cmd.val);
        wait(5); // Wait for value to get stable
        rd_data[i] = reg_read(RISCV_DEBUG_REG_DBG_ARRAY_RD_DATA);
    }

    // Disable debug control
    dbg_array_rd_cmd.val = 0;
    reg_write(RISCV_DEBUG_REG_DBG_ARRAY_RD_CMD, dbg_array_rd_cmd.val);
    dbg_array_rd_en.val = 0;
    reg_write(RISCV_DEBUG_REG_DBG_ARRAY_RD_EN, dbg_array_rd_en.val);

    // Restore dest row
    if (array_id == dbg_array_id::SRCA)
    {
        TTI_SFPSTORE(p_sfpu::LREG3, 0, 0, 0); // Restore dest addr 0 (even cols) from LREG_3
        TTI_SFPSTORE(p_sfpu::LREG3, 0, 0, 2); // Restore dest addr 0 (odd cols) from LREG_3
        // Move to the current bank
        TTI_CLEARDVALID(1, 0);
    }
}

inline std::uint32_t dbg_read_cfgreg(const uint32_t cfgreg_id, const uint32_t addr)
{
    uint32_t hw_base_addr = 0;

    switch (cfgreg_id)
    {
        case dbg_cfgreg::HW_CFG_1:
            hw_base_addr = dbg_cfgreg::HW_CFG_SIZE;
            break;
        case dbg_cfgreg::THREAD_0_CFG:
        case dbg_cfgreg::THREAD_1_CFG:
        case dbg_cfgreg::THREAD_2_CFG:
            hw_base_addr = 2 * dbg_cfgreg::HW_CFG_SIZE + cfgreg_id * dbg_cfgreg::THREAD_CFG_SIZE;
            break;
        default:
            break;
    }

    uint32_t hw_addr = hw_base_addr + (addr & 0x7ff); // hw address is 4-byte aligned
    reg_write(RISCV_DEBUG_REG_CFGREG_RD_CNTL, hw_addr);

    wait(1);

    return reg_read(RISCV_DEBUG_REG_CFGREG_RDDATA);
}

} // namespace ckernel
