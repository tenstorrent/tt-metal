// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "tensix.h" //RISCV_DEBUG macros

enum class rvdbg_cmd : uint32_t
{
    PAUSE      = (1 << 0),
    STEP       = (1 << 1),
    CONTINUE   = (1 << 2),
    RD_REG     = (1 << 3),
    WR_REG     = (1 << 4),
    RD_MEM     = (1 << 5),
    WR_MEM     = (1 << 6),
    FLUSH_REGS = (1 << 7),
    FLUSH      = (1 << 8),
    RD_CSR     = (1 << 9),
    WR_CSR     = (1 << 10),
    RD_FPREG   = (1 << 11),
    WR_FPREG   = (1 << 12),
    RD_VECREG  = (1 << 13),
    WR_VECREG  = (1 << 14),
    RD_UNIT    = (1 << 15),

    DBG_MODE_BIT = (1U << 31)
};

enum class rvdbg_risc_sel : uint32_t
{
    TRISC0 = 0x0000'0000,
    TRISC1 = 0x0002'0000,
    TRISC2 = 0x0004'0000,
    TRISC3 = 0x0006'0000
};

enum class rvdbg_reg : uint32_t
{
    STATUS          = 0,
    COMMAND         = 1,
    COMMAND_ARG0    = 2,
    COMMAND_ARG1    = 3,
    COMMAND_RET_VAL = 4,
    WCHPT_SETTINGS  = 5,

    HW_WCHPT0 = 10,
    HW_WCHPT1 = 11,
    HW_WCHPT2 = 12,
    HW_WCHPT3 = 13,
    HW_WCHPT4 = 14,
    HW_WCHPT5 = 15,
    HW_WCHPT6 = 16,
    HW_WCHPT7 = 17
};

union rvdbg_status
{
    struct
    {
        unsigned paused    : 1;
        unsigned brkpt_hit : 1;
        unsigned wchpt_hit : 1;
        unsigned ebrk_hit  : 1;
        unsigned unused0   : 4;
        unsigned reason    : 8;
    };

    uint32_t val;
};

/* Quick reference:
RISC_DBG_CNTL_0[31:0] = {
    pulse,         //A 0-1 transition on this bit triggers an access pulse
    11'b0,
    risc_sel[2:0], //0 for BRISC, 1-3 for TRISCs 0-2 (respectively), 4 for NCRISC
    reg_wr,        //0 for reads, 1 for writes (when access pulsed)
    5'b0,
    reg_addr[10:0]
};

RISC_DBG_CNTL_1[31:0] = reg_wr_data[31:0];

RISC_DBG_STATUS_0[31:0] = {
    pulse,         //Read-only mirror
    rd_valid,      //Gets set to 1 when a read returns, and stays at 1 until
                   //the next request. If you make a request on the same cycle
                   //as a returned read, then the returned read is dropped.
    12'b0,
    reg_wr[4:0],   //Read-only mirror of per-RISC write signal
    2'b0,
    reg_addr[10:0] //Read-only mirror
};

RISC_DBG_STATUS_1[31:0] = reg_rd_data; //Gets saved when rd_valid goes high and
                                       //remains constant until the next read
*/

#define CNTL0_PULSE      (0x8000'0000)
#define CNTL0_WR         (0x0001'0000)
#define CNTL0_RD         (0x0000'0000)
#define STATUS0_RD_VALID (0x4000'0000)

// Using the (rather clumsy) interface in the riscv_debug_regs, write a
// value to one of the RISCV debug control registers
void riscv_dbg_wr(rvdbg_risc_sel risc_sel, rvdbg_reg index, uint32_t val)
{
    uint32_t const cntl0_wr_cmd = CNTL0_PULSE | static_cast<uint32_t>(risc_sel) | CNTL0_WR | static_cast<uint32_t>(index);

    RISCV_DEBUG_REGS->RISC_DBG_CNTL_1 = val;
    RISCV_DEBUG_REGS->RISC_DBG_CNTL_0 = 0; // Clear pulse bit so we get a 0-1 transition on it
    RISCV_DEBUG_REGS->RISC_DBG_CNTL_0 = cntl0_wr_cmd;
    // The pulse signal passes through a synchronizer, but the other
    // fields (such as reg address and write_data) don't. So, we have
    // no choice but to try and insert three cycles of delay so that
    // they stay constant
    asm volatile("nop; nop; nop;");
}

// Using the (rather clumsy) interface in the riscv_debug_regs, read a
// value from one of the RISCV debug control registers
uint32_t riscv_dbg_rd(rvdbg_risc_sel risc_sel, rvdbg_reg index)
{
    uint32_t const cntl0_rd_cmd = CNTL0_PULSE | static_cast<uint32_t>(risc_sel) | CNTL0_RD | static_cast<uint32_t>(index);

    RISCV_DEBUG_REGS->RISC_DBG_CNTL_0 = 0; // Clear pulse bit so we get a 0-1 transition on it
    RISCV_DEBUG_REGS->RISC_DBG_CNTL_0 = cntl0_rd_cmd;
    // The pulse signal passes through a synchronizer, but the other
    // fields (such as reg address and write_data) don't. So, we have
    // no choice but to try and insert three cycles of delay so that
    // they stay constant
    asm volatile("nop; nop; nop;");

    // Wait for rd_valid to go high
    while ((RISCV_DEBUG_REGS->RISC_DBG_STATUS_0 & STATUS0_RD_VALID) == 0)
        ;

    return RISCV_DEBUG_REGS->RISC_DBG_STATUS_1;
}

inline void riscv_dbg_cmd(rvdbg_cmd cmd, rvdbg_risc_sel risc_sel)
{
    riscv_dbg_wr(risc_sel, rvdbg_reg::COMMAND, static_cast<uint32_t>(cmd));
}

inline uint32_t riscv_dbg_cmd(rvdbg_cmd cmd, rvdbg_risc_sel risc_sel, uint32_t arg0)
{
    riscv_dbg_wr(risc_sel, rvdbg_reg::COMMAND_ARG0, arg0);
    riscv_dbg_wr(risc_sel, rvdbg_reg::COMMAND, static_cast<uint32_t>(cmd));

    return riscv_dbg_rd(risc_sel, rvdbg_reg::COMMAND_RET_VAL);
}

inline void riscv_dbg_cmd(rvdbg_cmd cmd, rvdbg_risc_sel risc_sel, uint32_t arg0, uint32_t arg1)
{
    riscv_dbg_wr(risc_sel, rvdbg_reg::COMMAND_ARG0, arg0);
    riscv_dbg_wr(risc_sel, rvdbg_reg::COMMAND_ARG1, arg1);
    riscv_dbg_wr(risc_sel, rvdbg_reg::COMMAND, static_cast<uint32_t>(cmd));
}
