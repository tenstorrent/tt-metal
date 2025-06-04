// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSIX_H_INCLUDED
#define TENSIX_H_INCLUDED

// #include "tensix_prototypes.h"
#include <tensix_types.h>

#include <cstdint>
#include <limits>
#include <type_traits>

#include "cfg_defines.h"

// Convenience and type defines
using uint = std::uint32_t;
using byte = std::uint8_t;

#define PREPROCESSOR_EVAL(x, y, z) x##y##z
#define PREPROCESSOR_EXPAND(x, y, z) PREPROCESSOR_EVAL(x, y, z)

#define MAX_THREADS 3  // max number of threads supported by single core

#define MAX_PACKERS 4  // number of packers in the design

// TODO: use this in firmware.cc
#define MEMORY_WORD_SIZE_IN_BYTES (16)
#define MEMORY_WORD_SHIFT_BITS (4)  // log2(MEMORY_WORD_SIZE_IN_BYTES)

#define STALLWAIT_COMPUTE (0x0)
#define STALLWAIT_TDMA (0x1)
#define STALLWAIT_FOR_TC (0x1 << 0)
#define STALLWAIT_FOR_UNP0 (0x1 << 1)
#define STALLWAIT_FOR_UNP1 (0x1 << 2)
#define STALLWAIT_FOR_PACK (0x1 << 3)

/////////////
// RISC-V Address map definition (hardware)

// TODO: Consider redefining these as uint32_t rather then #defines

// Reads and writes here access the tensix core register set. Each register is four bytes, but subword reads are
// supported through byte enables. Register indices and contents are defined in local_regs.yaml.
#define REGFILE_BASE 0xFFE00000  // 0xFFE00000 - 0xFFE3FFFF

// Writes here are appended to the tensix core instruction FIFO. This has priority over incoming instruction fetch
// returns, which are simply dropped. The instruction will stay in the queue if a loop instruction is in progress. If
// the FIFO gets overfull, writes are dropped? Additionally, the instruction queue is flushed in some cases.
#define INSTRN_BUF_BASE 0xFFE40000   // 0xFFE40000 - 0xFFE7FFFF
#define INSTRN1_BUF_BASE 0xFFE50000  // 0xFFE40000 - 0xFFE7FFFF
#define INSTRN2_BUF_BASE 0xFFE60000

// PC buffer is used to pass kernel IDs and paramters from Brisc to Triscs, and also as a sync point -- a read from pc
// buffer+1 address will not return until that thread is idle.
#define PC_BUF_BASE 0xFFE80000   // 0xFFE80000 - 0xFFEBFFFF
#define PC1_BUF_BASE 0xFFE90000  // 0xFFE80000 - 0xFFEBFFFF
#define PC2_BUF_BASE 0xFFEA0000

// Reads from here retrieve a value written by the tensix code, or 0 if there the mailbox FIFO is empty.
#define TENSIX_MAILBOX0_BASE 0xFFEC0000  // Brisc
#define TENSIX_MAILBOX1_BASE 0xFFEC1000  // Trisc0
#define TENSIX_MAILBOX2_BASE 0xFFEC2000  // Trisc1
#define TENSIX_MAILBOX3_BASE 0xFFEC3000  // Trisc2

// Config registers
#define TENSIX_CFG_BASE 0xFFEF0000  // 0xFFEF0000 - 0xFFF00000

// MOP config registers
#define TENSIX_MOP_CFG_BASE 0xFFB80000  // 0xFFB8000 - 0xFFB8100

// TDMA register base
#define RISCV_TDMA_REGS_START_ADDR 0xFFB11000
#define RISCV_TDMA_REG_XMOV_SRC_ADDR 0xFFB11000
#define RISCV_TDMA_REG_XMOV_DST_ADDR 0xFFB11004
#define RISCV_TDMA_REG_XMOV_SIZE 0xFFB11008
#define RISCV_TDMA_REG_XMOV_DIRECTION 0xFFB1100C
#define RISCV_TDMA_REG_COMMAND_ADDR 0xFFB11010
#define RISCV_TDMA_REG_STATUS 0xFFB11014
#define RISCV_TDMA_REG_PACKED_SIZE 0xFFB11018
#define RISCV_TDMA_REG_ACC_PACKED_SIZE 0xFFB1101C   // read only
#define RISCV_TDMA_REG_INITIAL_PACK_ACC 0xFFB1101C  // write only
#define RISCV_TDMA_REG_CLK_GATE_EN 0xFFB11024
#define RISCV_TDMA_REG_CLK_GATE_HYST 0xFFB11028
#define RISCV_TDMA_REG_XMOV_L1_BASE_ADDR 0xFFB1102C
#define RISCV_TDMA_REG_FIFO_PACKED_TILE_SIZE(packer) (0xFFB11030 | (packer << 8))
#define RISCV_TDMA_REG_FIFO_PACKED_TILE_ZEROMASK(packer) (0xFFB11034 | (packer << 8))
#define RISCV_TDMA_REG_FIFO_PACKED_TILE_STATUS (0xFFB11038)

#define RISCV_TDMA_PACKED_TILE_FIFO_EMPTY(status, packer) ((status >> (packer * 2)) & 0x1)
#define RISCV_TDMA_PACKED_TILE_FIFO_FULL(status, packer) ((status >> (packer * 2 + 1)) & 0x1)
#define RISCV_TDMA_STATUS_FLAG_MOVER0_BUSY_MASK 0x01
#define RISCV_TDMA_STATUS_FLAG_MOVER1_BUSY_MASK 0x02
#define RISCV_TDMA_STATUS_FLAG_FIFO_FULL_MASK 0x04
#define RISCV_TDMA_STATUS_FLAG_FIFO_EMPTY_MASK 0x08
#define RISCV_TDMA_STATUS_FLAG_ERROR_MASK 0x10

#define RISCV_SOFT_RESET_0_NONE 0x00000
#define RISCV_SOFT_RESET_0_BRISC 0x00800
#define RISCV_SOFT_RESET_0_NCRISC 0x40000
#define RISCV_SOFT_RESET_0_TRISCS 0x07000

// Debug registers
/*
#!/usr/bin/env lua

-- Stupid little helper to generate the defines shown below

root = assert(os.getenv("ROOT"), "Must provide ROOT env var porinting to repo root")
f = assert(io.open(root .. "/src/hardware/tensix/rtl/tt_risc_debug_regs.sv"))

f = f:read("*a")

for nm,addr in f:gmatch("localparam%s*([a-zA-Z0-9_]*)%s*=%s*32'h(%x*)%s*;") do
    io.write"#define RISCV_DEBUG_REG_"
    io.write(nm)
    local num = 40 - #nm
    if num < 0 then num = 0 end
    local pad = string.rep(" ", num)
    io.write(pad)
    io.write"(RISCV_DEBUG_REGS_START_ADDR | 0x"
    io.write(addr)
    io.write")\n"
end
*/
#define RISCV_DEBUG_REGS_START_ADDR 0xFFB12000
#define RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0 (RISCV_DEBUG_REGS_START_ADDR | 0x0)
#define RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD1 (RISCV_DEBUG_REGS_START_ADDR | 0x4)
#define RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD2 (RISCV_DEBUG_REGS_START_ADDR | 0x8)
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0 (RISCV_DEBUG_REGS_START_ADDR | 0xC)
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK1 (RISCV_DEBUG_REGS_START_ADDR | 0x10)
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK2 (RISCV_DEBUG_REGS_START_ADDR | 0x14)
#define RISCV_DEBUG_REG_PERF_CNT_FPU0 (RISCV_DEBUG_REGS_START_ADDR | 0x18)
#define RISCV_DEBUG_REG_PERF_CNT_FPU1 (RISCV_DEBUG_REGS_START_ADDR | 0x1C)
#define RISCV_DEBUG_REG_PERF_CNT_FPU2 (RISCV_DEBUG_REGS_START_ADDR | 0x20)
#define RISCV_DEBUG_REG_PERF_CNT_L1_0 (RISCV_DEBUG_REGS_START_ADDR | 0x30)
#define RISCV_DEBUG_REG_PERF_CNT_L1_1 (RISCV_DEBUG_REGS_START_ADDR | 0x34)
#define RISCV_DEBUG_REG_PERF_CNT_L1_2 (RISCV_DEBUG_REGS_START_ADDR | 0x38)
#define RISCV_DEBUG_REG_PERF_CNT_ALL (RISCV_DEBUG_REGS_START_ADDR | 0x3C)
#define RISCV_DEBUG_REG_DBG_L1_MEM_REG0 (RISCV_DEBUG_REGS_START_ADDR | 0x48)
#define RISCV_DEBUG_REG_DBG_L1_MEM_REG1 (RISCV_DEBUG_REGS_START_ADDR | 0x4C)
#define RISCV_DEBUG_REG_DBG_L1_MEM_REG2 (RISCV_DEBUG_REGS_START_ADDR | 0x50)
#define RISCV_DEBUG_REG_DBG_BUS_CTRL (RISCV_DEBUG_REGS_START_ADDR | 0x54)
#define RISCV_DEBUG_REG_TENSIX_CREG_READ (RISCV_DEBUG_REGS_START_ADDR | 0x58)
#define RISCV_DEBUG_REG_DBG_RD_DATA (RISCV_DEBUG_REGS_START_ADDR | 0x5C)
#define RISCV_DEBUG_REG_THREAD1_CREG_READ (RISCV_DEBUG_REGS_START_ADDR | 0x5C)
#define RISCV_DEBUG_REG_DBG_ARRAY_RD_EN (RISCV_DEBUG_REGS_START_ADDR | 0x60)
#define RISCV_DEBUG_REG_DBG_ARRAY_RD_CMD (RISCV_DEBUG_REGS_START_ADDR | 0x64)
#define RISCV_DEBUG_REG_DBG_FEATURE_DISABLE (RISCV_DEBUG_REGS_START_ADDR | 0x68)
#define RISCV_DEBUG_REG_DBG_ARRAY_RD_DATA (RISCV_DEBUG_REGS_START_ADDR | 0x6C)
#define RISCV_DEBUG_REG_CG_CTRL_HYST0 (RISCV_DEBUG_REGS_START_ADDR | 0x70)
#define RISCV_DEBUG_REG_CG_CTRL_HYST1 (RISCV_DEBUG_REGS_START_ADDR | 0x74)
#define RISCV_DEBUG_REG_TENSIX_CREG_RDDATA (RISCV_DEBUG_REGS_START_ADDR | 0x78)
#define RISCV_DEBUG_REG_CG_CTRL_HYST2 (RISCV_DEBUG_REGS_START_ADDR | 0x7C)
#define RISCV_DEBUG_REG_THREAD1_CREG_RDDATA (RISCV_DEBUG_REGS_START_ADDR | 0x7C)
#define RISCV_DEBUG_REG_RISC_DBG_CNTL_0 (RISCV_DEBUG_REGS_START_ADDR | 0x80)
#define RISCV_DEBUG_REG_RISC_DBG_CNTL_1 (RISCV_DEBUG_REGS_START_ADDR | 0x84)
#define RISCV_DEBUG_REG_RISC_DBG_STATUS_0 (RISCV_DEBUG_REGS_START_ADDR | 0x88)
#define RISCV_DEBUG_REG_RISC_DBG_STATUS_1 (RISCV_DEBUG_REGS_START_ADDR | 0x8C)
#define RISCV_DEBUG_REG_TRISC_PC_BUF_OVERRIDE (RISCV_DEBUG_REGS_START_ADDR | 0x90)
#define RISCV_DEBUG_REG_DBG_INVALID_INSTRN (RISCV_DEBUG_REGS_START_ADDR | 0x94)
#define RISCV_DEBUG_REG_DBG_INSTRN_BUF_CTRL0 (RISCV_DEBUG_REGS_START_ADDR | 0xA0)
#define RISCV_DEBUG_REG_DBG_INSTRN_BUF_CTRL1 (RISCV_DEBUG_REGS_START_ADDR | 0xA4)
#define RISCV_DEBUG_REG_DBG_INSTRN_BUF_STATUS (RISCV_DEBUG_REGS_START_ADDR | 0xA8)
#define RISCV_DEBUG_REG_STOCH_RND_MASK0 (RISCV_DEBUG_REGS_START_ADDR | 0xAC)
#define RISCV_DEBUG_REG_STOCH_RND_MASK1 (RISCV_DEBUG_REGS_START_ADDR | 0xB0)
#define RISCV_DEBUG_REG_FPU_STICKY_BITS (RISCV_DEBUG_REGS_START_ADDR | 0xB4)
#define RISCV_DEBUG_REG_ETH_RISC_PREFECTH_CTRL (RISCV_DEBUG_REGS_START_ADDR | 0xB8)
#define RISCV_DEBUG_REG_ETH_RISC_PREFECTH_PC (RISCV_DEBUG_REGS_START_ADDR | 0xBC)
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK0 (RISCV_DEBUG_REGS_START_ADDR | 0xF0)
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK1 (RISCV_DEBUG_REGS_START_ADDR | 0xF4)
#define RISCV_DEBUG_REG_PERF_CNT_TDMA_PACK2 (RISCV_DEBUG_REGS_START_ADDR | 0xF8)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD (RISCV_DEBUG_REGS_START_ADDR | 0x100)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_H_INSTRN_THREAD (RISCV_DEBUG_REGS_START_ADDR | 0x104)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK (RISCV_DEBUG_REGS_START_ADDR | 0x108)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_UNPACK (RISCV_DEBUG_REGS_START_ADDR | 0x10C)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_PACK (RISCV_DEBUG_REGS_START_ADDR | 0x110)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_H_TDMA_PACK (RISCV_DEBUG_REGS_START_ADDR | 0x114)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_DBG_L1 (RISCV_DEBUG_REGS_START_ADDR | 0x118)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_H_DBG_L1 (RISCV_DEBUG_REGS_START_ADDR | 0x11C)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU (RISCV_DEBUG_REGS_START_ADDR | 0x120)
#define RISCV_DEBUG_REG_PERF_CNT_OUT_H_FPU (RISCV_DEBUG_REGS_START_ADDR | 0x124)
#define RISCV_DEBUG_REG_SOFT_RESET_0 (RISCV_DEBUG_REGS_START_ADDR | 0x1B0)
#define RISCV_DEBUG_REG_ECC_CTRL (RISCV_DEBUG_REGS_START_ADDR | 0x1D0)
#define RISCV_DEBUG_REG_ECC_STATUS (RISCV_DEBUG_REGS_START_ADDR | 0x1D4)
#define RISCV_DEBUG_REG_WATCHDOG_TIMER (RISCV_DEBUG_REGS_START_ADDR | 0x1E0)
#define RISCV_DEBUG_REG_WDT_CNTL (RISCV_DEBUG_REGS_START_ADDR | 0x1E4)
#define RISCV_DEBUG_REG_WDT_STATUS (RISCV_DEBUG_REGS_START_ADDR | 0x1E8)
#define RISCV_DEBUG_REG_WALL_CLOCK_0 (RISCV_DEBUG_REGS_START_ADDR | 0x1F0)
#define RISCV_DEBUG_REG_WALL_CLOCK_1 (RISCV_DEBUG_REGS_START_ADDR | 0x1F4)
#define RISCV_DEBUG_REG_WALL_CLOCK_1_AT (RISCV_DEBUG_REGS_START_ADDR | 0x1F8)
#define RISCV_DEBUG_REG_TIMESTAMP_DUMP_CMD (RISCV_DEBUG_REGS_START_ADDR | 0x1FC)
#define RISCV_DEBUG_REG_TIMESTAMP_DUMP_CNTL (RISCV_DEBUG_REGS_START_ADDR | 0x200)
#define RISCV_DEBUG_REG_TIMESTAMP_DUMP_STATUS (RISCV_DEBUG_REGS_START_ADDR | 0x204)
#define RISCV_DEBUG_REG_TIMESTAMP_DUMP_BUF0_START_ADDR (RISCV_DEBUG_REGS_START_ADDR | 0x208)
#define RISCV_DEBUG_REG_TIMESTAMP_DUMP_BUF0_END_ADDR (RISCV_DEBUG_REGS_START_ADDR | 0x20C)
#define RISCV_DEBUG_REG_TIMESTAMP_DUMP_BUF1_START_ADDR (RISCV_DEBUG_REGS_START_ADDR | 0x210)
#define RISCV_DEBUG_REG_TIMESTAMP_DUMP_BUF1_END_ADDR (RISCV_DEBUG_REGS_START_ADDR | 0x214)
#define RISCV_DEBUG_REG_PERF_CNT_MUX_CTRL (RISCV_DEBUG_REGS_START_ADDR | 0x218)
#define RISCV_DEBUG_REG_DBG_L1_READBACK_OFFSET (RISCV_DEBUG_REGS_START_ADDR | 0x21C)
#define RISCV_DEBUG_REG_LFSR_HIT_MASK (RISCV_DEBUG_REGS_START_ADDR | 0x220)
#define RISCV_DEBUG_REG_DISABLE_RESET (RISCV_DEBUG_REGS_START_ADDR | 0x224)
#define RISCV_DEBUG_REG_TRISC0_RESET_PC (RISCV_DEBUG_REGS_START_ADDR | 0x228)
#define RISCV_DEBUG_REG_TRISC1_RESET_PC (RISCV_DEBUG_REGS_START_ADDR | 0x22C)
#define RISCV_DEBUG_REG_TRISC2_RESET_PC (RISCV_DEBUG_REGS_START_ADDR | 0x230)
#define RISCV_DEBUG_REG_TRISC_RESET_PC_OVERRIDE (RISCV_DEBUG_REGS_START_ADDR | 0x234)
#define RISCV_DEBUG_REG_NCRISC_RESET_PC (RISCV_DEBUG_REGS_START_ADDR | 0x238)
#define RISCV_DEBUG_REG_NCRISC_RESET_PC_OVERRIDE (RISCV_DEBUG_REGS_START_ADDR | 0x23C)
#define RISCV_DEBUG_REG_DEST_CG_CTRL (RISCV_DEBUG_REGS_START_ADDR | 0x240)
#define RISCV_DEBUG_REG_CG_CTRL_EN (RISCV_DEBUG_REGS_START_ADDR | 0x244)
#define RISCV_DEBUG_REG_CG_KICK (RISCV_DEBUG_REGS_START_ADDR | 0x248)

// Here are the old manually-written defines that weren't covered by the
// generator script, or are being depended on by legacy code:
#define RISCV_DEBUG_REG_BREAKPOINT_CTRL (RISCV_DEBUG_REGS_START_ADDR | 0x1C0)
#define RISCV_DEBUG_REG_BREAKPOINT_STATUS (RISCV_DEBUG_REGS_START_ADDR | 0x1C4)
#define RISCV_DEBUG_REG_BREAKPOINT_DATA (RISCV_DEBUG_REGS_START_ADDR | 0x1C8)
#define RISCV_DEBUG_REG_INSTRN_BUF_CTRL0 (RISCV_DEBUG_REGS_START_ADDR | 0x0A0)
#define RISCV_DEBUG_REG_INSTRN_BUF_CTRL1 (RISCV_DEBUG_REGS_START_ADDR | 0x0A4)
#define RISCV_DEBUG_REG_INSTRN_BUF_STATUS (RISCV_DEBUG_REGS_START_ADDR | 0x0A8)
#define RISCV_DEBUG_REG_THREAD0_CREG_RDDATA (RISCV_DEBUG_REGS_START_ADDR | 0x078)
#define RISCV_DEBUG_REG_WALL_CLOCK_L (RISCV_DEBUG_REGS_START_ADDR | 0x1F0)
#define RISCV_DEBUG_REG_WALL_CLOCK_H (RISCV_DEBUG_REGS_START_ADDR | 0x1F8)
#define RISCV_DEBUG_REG_WDT (RISCV_DEBUG_REGS_START_ADDR | 0x1E0)
#define RISCV_DEBUG_REG_WDT_CNTL (RISCV_DEBUG_REGS_START_ADDR | 0x1E4)
#define RISCV_DEBUG_REG_WDT_STATUS (RISCV_DEBUG_REGS_START_ADDR | 0x1E8)

struct riscv_debug_reg_dbg_dbus_cntl_t {
    uint dbg_sig_sel : 16;
    uint dbg_daisy_sel : 8;
    uint dbg_rd_sel : 4;
    uint dbg_reg_ovrd_en : 1;
    uint dbg_daisy_en : 1;
    uint dbg_reserved : 2;
};

union riscv_debug_reg_dbg_dbus_cntl_u {
    uint val;
    riscv_debug_reg_dbg_dbus_cntl_t f;
};

struct riscv_debug_reg_dbg_l1_mem_reg2_t {
    uint mem_dump_mode : 4;
    uint skip_cycles : 8;
    uint mem_write : 1;
    uint mem_read : 1;
    uint reserved : 18;
};

union riscv_debug_reg_dbg_l1_mem_reg2_u {
    uint val;
    riscv_debug_reg_dbg_l1_mem_reg2_t f;
};

#define SOFT_RESET_UNPACKER(arg) ((arg & 0x3) << 0)
#define SOFT_RESET_PACKER(arg) ((arg & 0xf) << 2)
#define SOFT_RESET_MOVER ((0x1) << 6)
#define SOFT_RESET_SEARCH ((0x1) << 7)
#define SOFT_RESET_GLUE ((0x1) << 8)
#define SOFT_RESET_THCON ((0x1) << 9)
#define SOFT_RESET_FPU ((0x1) << 10)
#define SOFT_RESET_RISC_CTRL(arg) ((arg & 0xf) << 11)  // Soft reset for RISCV cores. Bit 0 - Brisc, Bit 1+ - Trisc
#define SOFT_RESET_SRCA_REG ((0x1) << 15)
#define SOFT_RESET_SRCB_REG ((0x1) << 16)
#define SOFT_RESET_DEST_REG ((0x1) << 17)

// TDMA flop register index offset
#define TDMA_FLOPREG_IDX_BASE(arg) ((arg) * 32)

/////////////
// Interrupt controller definitions
#define RISC_PIC_BASE 0xFFB1'3000
#define RISC_PIC_BASE_PTR ((uint32_t volatile*)0xFFB1'3000)
#define RISC_PIC_BRISC_SW_INT_EN (RISC_PIC_BASE_PTR + 0)
#define RISC_PIC_BRISC_HW_INT_EN (RISC_PIC_BASE_PTR + 1)
#define RISC_PIC_BRISC_INT_NO (RISC_PIC_BASE_PTR + 2)
#define RISC_PIC_NCRISC_SW_INT_EN (RISC_PIC_BASE_PTR + 3)
#define RISC_PIC_NCRISC_HW_INT_EN (RISC_PIC_BASE_PTR + 4)
#define RISC_PIC_NCRISC_INT_NO (RISC_PIC_BASE_PTR + 5)
#define RISC_PIC_SW_INT_REGS (RISC_PIC_BASE_PTR + 6)
#define RISC_PIC_HW_INTS (RISC_PIC_BASE_PTR + 38)
#define RISC_PIC_INT_PCS (RISC_PIC_BASE_PTR + 42)

/////////////
// Instruction macro definitions
// Consult instruction documentation in assembly.yaml
#define INSTRN_GETDESC(arg) (0x40000000 | (arg))  // Unimplemented.
#define INSTRN_PACRNL(arg) (0x41000000 | (arg))   // Pack row from DST to L0/L1
#define INSTRN_UNPACR(arg) (0x42000000 | (arg))   // Unpack row from tile in L0 to SRCA/SRCB
#define INSTRN_SEARCHX(arg) \
    (0x43000000 |           \
     (arg))  // Search for start of selected row within tile. To be invoked prior to each invocation of UNPACR.
#define INSTRN_RSTDMA 0x44000000  // Soft reset of TDMA engine
#define INSTRN_SET_DMA_REG(arg) \
    (0x45000000 | (arg))  // Set TDMA register file register with 16b immediate value provided with instruction
#define INSTRN_FLUSH_DMA(arg) \
    (0x46000000 | (arg))  // Flush TDMA engine or some subset of it as specified by instruction argument
#define INSTRN_MV_REG_TO_FLOPS(arg) \
    (0x48000000 | (arg))  // Move data from TDMA register file into flip flops driving actual config signals. Used for
                          // certain TDMA configuration signal setting.
#define INSTRN_LOAD_IND(arg) \
    (0x49000000 | (arg))  // Load indirect from address specified in a TDMA register, with offset specified in TDMA
                          // register to a TDMA register. Supports autoincrementing offset
#define INSTRN_AT_INCR_GET(arg) \
    (0x61000000 | (arg))  // Atomic increment and get - will read value in targetted memory location and return it to
                          // TDMA register and post-increment it atomically
#define INSTRN_AT_INCR_GET_PTR(arg) \
    (0x62000000 |                   \
     (arg))  // Atomic increment and get pointer - will access a memory location designated as a FIFO pointer location
             // (contains a 32b read pointer and a 32b write pointer), return the pointer value to TDMA register and
             // post-increment it unless the FIFO condition precludes that. For example, write pointer will not be
             // incremented if FIFO is full. Read pointer will not be incremented if FIFO is empty. FIFO full or empty
             // conditions are returned as an unsuccessfull return condition code, so that the thread controller can
             // retry until success (retry reads if FIFO empty, retry writes if FIFO full.)
#define INSTRN_AT_SWAP(arg) \
    (0x63000000 | (arg))  // Atomic unconditional SWAP. Swaps selected 16b chunks of memory location with new ones
                          // provided on write data bus.
#define INSTRN_AT_CAS(arg) \
    (0x64000000 | (arg))  // Atomic compare-and-swap. If value at selected memory location matches that provided by
                          // programmer it is swapped to a new one, also provided by programmer. This instruction is
                          // implemented for implementations of mutual exclusion between Tensix cores and threads
#define INSTRN_STORE_IND(arg) \
    (0x66000000 |             \
     (arg))  // Store indirect. Stores data from TDMA register to memory location specified by a combination of
             // base+offset provided in other TDMA registers. Supports auto-increment on offset value.

#define INSTRN_SETC16(arg) \
    (0xb2000000 | (arg))  // Sets thread specific control register <register> to the value stored in the slot argument.
                          // 32-bit instruction. Register index (bits16-23) Value: (bits 15-0).
#define INSTRN_WRCFG(arg) (0xb0000000 | (arg))
#define INSTRN_RDCFG(arg) (0xb1000000 | (arg))

#define INSTRN_SETC(arg) \
    (0x80000000 | (arg))  // Sets thread specific control register <register> to the value stored in the slot argument.
                          // 64-bit instruction. Register index in low 11 bits of first word, register value in second
                          // word. **Deprecated**
#define INSTRN_SETRWC(arg) (0x38000000 | (arg))        //
#define INSTRN_SETADC(arg) (0x50000000 | (arg))        // Set address counter for one channel and one dimension.
#define INSTRN_SETADCXY(arg) (0x51000000 | (arg))      // Set address counters for X and Y dimensions for all channels
#define INSTRN_SETADCZW(arg) (0x54000000 | (arg))      // Set address counters for Z and W dimensions for all channels
#define INSTRN_FLUSH(arg) (0x81000000 | (arg))         // Flush all buffers of oustanding instructions, reads/writes.
#define INSTRN_NOP(arg) (0x02000000 | (arg))           // Do nothing and consume an instruction slot and a cycle
#define INSTRN_MOVA2D(arg) (0x1a000000 | (arg))        // Move SRCA register to DST
#define INSTRN_ZEROSRC(arg) (0x1b000000 | (arg))       // Clear SRC registers
#define INSTRN_SETPKEDGEOF(arg) (0x1d000000 | (arg))   // Set packer edge masking offsets
#define INSTRN_STALLWAIT(arg) (0xa2000000 | (arg))     // Stall resource until condition is met
#define INSTRN_CLEAR_DVALID(arg) (0x37000000 | (arg))  // Clear dvalid bits
#define INSTRN_SEMINIT(arg) (0xa3000000 | (arg))       // Initialize a semaphore
#define INSTRN_ZEROACC(arg) (0x10000000 | (arg))       // Zero out the accumulator
#define INSTRN_SFPENCC(arg) (0x8a000000 | (arg))       // Enable the SFPU CC state
#define INSTRN_SFPLOADI(arg) (0x71000000 | (arg))      // Load an SFPU register
#define INSTRN_SFPCONFIG(arg) (0x91000000 | (arg))     // Set SFPU config register state

#define TENSIX_UNHALT_VAL \
    0x40000000  // When written into PC_BUF_BASE, tensix core will unhalt and continue execution at the previous PC.
#define TENSIX_NEWPC_VAL(arg) \
    (0x80000000 | (arg))  // Format a PC into a value that will unhalt the tensix core and jump to that PC. This value
                          // can be written into PC_BUF_BASE.
#define TENSIX_LOOP_PC_VAL(arg) (0x00000000 | (arg))  // Start a PC buffer loop
#define TENSIX_PC_SYNC(arg) (0xC0000000 | (arg))      // Sync - block until all kernels are done

#define INSTRN_HALTF(arg) \
    (0x90000000 | (arg))  // Final Halt PC, it will stop the thread in question from executing and only tensix reset can
                          // unhalt, can't be unhalted by usual register write.

// Instruction modes (i.e., selection) definitions
#define INSTRN_SEL_L0 0
#define INSTRN_SEL_L1 1

#define INSTRN_SEL_SIZE_16B 0
#define INSTRN_SEL_SIZE_4B 1
#define INSTRN_SEL_SIZE_2B 2
#define INSTRN_SEL_SIZE_1B 3

#define INSTRN_SEL_AUTO_INC_NONE 0
#define INSTRN_SEL_AUTO_INC_2B 1
#define INSTRN_SEL_AUTO_INC_4B 2
#define INSTRN_SEL_AUTO_INC_16B 3

#define INSTRN_SEL_RD_PTR 0
#define INSTRN_SEL_WR_PTR 1

#define REG2FLOP_TARGET_TDMA 0
#define REG2FLOP_TARGET_LOCAL_REGS 1
#define REG2FLOP_TARGET_ADDR_CNTRS 2

#define BYTE_OFFSET_ZERO 0
#define BYTE_OFFSET_ONE 1
#define BYTE_OFFSET_TWO 2
#define BYTE_OFFSET_THREE 3

// Address defines for "SETC registers" aka "Local registers" -- see src/meta/regspecs/local_regs.yaml
// FIXME: This needs to be generated from that yaml file... it went out of date without anyone noticing :(
/*
#define ALU_FORMAT_SPEC_REG      1
#define DEST_TARGET_REG_CFG      2
//#define MISC_CFG                 3
#define ALU_FORMAT_SPEC_REG0     4
#define ALU_FORMAT_SPEC_REG1     5
#define ALU_FORMAT_SPEC_REG2     6
#define UNP0_ADDR_CTRL_XY_REG_0  8
#define UNP0_ADDR_CTRL_ZW_REG_0  9
#define UNP0_ADDR_BASE_REG_0     10
#define UNP0_ADDR_CTRL_XY_REG_1  11
#define UNP0_ADDR_CTRL_ZW_REG_1  12
#define UNP0_ADDR_BASE_REG_1     13
#define UNP1_ADDR_CTRL_XY_REG_0  14
#define UNP1_ADDR_CTRL_ZW_REG_0  15
#define UNP1_ADDR_BASE_REG_0     16
#define UNP1_ADDR_CTRL_XY_REG_1  17
#define UNP1_ADDR_CTRL_ZW_REG_1  18
#define UNP1_ADDR_BASE_REG_1     19
#define PCK0_ADDR_CTRL_XY_REG_0  20
#define PCK0_ADDR_CTRL_ZW_REG_0  21
#define PCK0_ADDR_BASE_REG_0     22
#define PCK0_ADDR_CTRL_XY_REG_1  23
#define PCK0_ADDR_CTRL_ZW_REG_1  24
#define PCK0_ADDR_BASE_REG_1     25
#define SRCA_REGW_BASE           26
#define SRCB_REGW_BASE           27
#define DEST_REGW_BASE           28
#define MATH_FIDELITY_CTRL       29
#define LOOP_CNT_REG0            32
#define LOOP_CNT_REG1            33
#define LOOP_CNT_REG2            34
#define LOOP_CNT_REG3            35
#define LOOP_CNT_REG4            36
#define LOOP_CNT_REG5            37
#define LOOP_CNT_REG6            38
#define LOOP_CNT_REG7            39
#define LOOP_CNT_REG8            40
#define LOOP_CNT_REG9            41
#define LOOP_CNT_REG10           42
#define LOOP_CNT_REG11           43
#define MATH_FIDELITY            44
#define TXC_IC_INVALIDATE        45
#define RISCV_IC_INVALIDATE      46
#define STACC_RELU               47
#define PCK_EDGE_OFFSET          48
#define DEST_OFFSET              49
#define DEBUG_MUX_CTRL           50
#define DEBUG_MUX_RD             51
#define SET_ADDRCNT_PROG_INC     52
#define SET_REGWCNT_PROG_INC     53
#define DEBUG_ARRAY_RD_CMD       54
#define DEBUG_ARRAY_RD_EN        55
#define CG_CTRL_EN               56
#define CG_CTRL_KICK             57
#define PERF_CNT_CMD0            58
#define PERF_CNT_CMD1            59
#define PERF_CNT_CMD2            60
#define PERF_CNT_CMD3            61
#define ENABLE_ACC_STATS         62
#define DISABLE_RISC_BP          63
*/

#define ADDR_16K 0x4000
#define ADDR_32K 0x8000
#define ADDR_64K 0x10000
#define ADDR_128K 0x20000
#define ADDR_256K 0x40000
#define ADDR_512K 0x80000
#define ONE_16B 16
#define ONE_32b 4

#define GET_16B_ADDR(arg) ((arg) >> 4)

// Tensix general purpose register file, 64 32-bit registers
static constexpr unsigned int R0 = 0;
static constexpr unsigned int R1 = 1;
static constexpr unsigned int R2 = 2;
static constexpr unsigned int R3 = 3;
static constexpr unsigned int R4 = 4;
static constexpr unsigned int R5 = 5;
static constexpr unsigned int R6 = 6;
static constexpr unsigned int R7 = 7;
static constexpr unsigned int R8 = 8;
static constexpr unsigned int R9 = 9;
static constexpr unsigned int R10 = 10;
static constexpr unsigned int R11 = 11;
static constexpr unsigned int R12 = 12;
static constexpr unsigned int R13 = 13;
static constexpr unsigned int R14 = 14;
static constexpr unsigned int R15 = 15;
static constexpr unsigned int R16 = 16;
static constexpr unsigned int R17 = 17;
static constexpr unsigned int R18 = 18;
static constexpr unsigned int R19 = 19;
static constexpr unsigned int R20 = 20;
static constexpr unsigned int R21 = 21;
static constexpr unsigned int R22 = 22;
static constexpr unsigned int R23 = 23;
static constexpr unsigned int R24 = 24;
static constexpr unsigned int R25 = 25;
static constexpr unsigned int R26 = 26;
static constexpr unsigned int R27 = 27;
static constexpr unsigned int R28 = 28;
static constexpr unsigned int R29 = 29;
static constexpr unsigned int R30 = 30;
static constexpr unsigned int R31 = 31;
static constexpr unsigned int R32 = 32;
static constexpr unsigned int R33 = 33;
static constexpr unsigned int R34 = 34;
static constexpr unsigned int R35 = 35;
static constexpr unsigned int R36 = 36;
static constexpr unsigned int R37 = 37;
static constexpr unsigned int R38 = 38;
static constexpr unsigned int R39 = 39;
static constexpr unsigned int R40 = 40;
static constexpr unsigned int R41 = 41;
static constexpr unsigned int R42 = 42;
static constexpr unsigned int R43 = 43;
static constexpr unsigned int R44 = 44;
static constexpr unsigned int R45 = 45;
static constexpr unsigned int R46 = 46;
static constexpr unsigned int R47 = 47;
static constexpr unsigned int R48 = 48;
static constexpr unsigned int R49 = 49;
static constexpr unsigned int R50 = 50;
static constexpr unsigned int R51 = 51;
static constexpr unsigned int R52 = 52;
static constexpr unsigned int R53 = 53;
static constexpr unsigned int R54 = 54;
static constexpr unsigned int R55 = 55;
static constexpr unsigned int R56 = 56;
static constexpr unsigned int R57 = 57;
static constexpr unsigned int R58 = 58;
static constexpr unsigned int R59 = 59;
static constexpr unsigned int R60 = 60;
static constexpr unsigned int R61 = 61;
static constexpr unsigned int R62 = 62;
static constexpr unsigned int R63 = 63;

// this is a "short" (i.e., 16-bit) interface to the 32-bit Tensix registers
// we can access LO or HI 16-bits of each 32-bit register
#define R0_LO 0
#define R0_HI 1
#define R1_LO 2
#define R1_HI 3
#define R2_LO 4
#define R2_HI 5
#define R3_LO 6
#define R3_HI 7
#define R4_LO 8
#define R4_HI 9
#define R5_LO 10
#define R5_HI 11
#define R6_LO 12
#define R6_HI 13
#define R7_LO 14
#define R7_HI 15
#define R8_LO 16
#define R8_HI 17
#define R9_LO 18
#define R9_HI 19
#define R10_LO 20
#define R10_HI 21
#define R11_LO 22
#define R11_HI 23
#define R12_LO 24
#define R12_HI 25
#define R13_LO 26
#define R13_HI 27
#define R14_LO 28
#define R14_HI 29
#define R15_LO 30
#define R15_HI 31
#define R16_LO 32
#define R16_HI 33
#define R17_LO 34
#define R17_HI 35
#define R18_LO 36
#define R18_HI 37
#define R19_LO 38
#define R19_HI 39
#define R20_LO 40
#define R20_HI 41
#define R21_LO 42
#define R21_HI 43
#define R22_LO 44
#define R22_HI 45
#define R23_LO 46
#define R23_HI 47
#define R24_LO 48
#define R24_HI 49
#define R25_LO 50
#define R25_HI 51
#define R26_LO 52
#define R26_HI 53
#define R27_LO 54
#define R27_HI 55
#define R28_LO 56
#define R28_HI 57
#define R29_LO 58
#define R29_HI 59
#define R30_LO 60
#define R30_HI 61
#define R31_LO 62
#define R31_HI 63
#define R32_LO 64
#define R32_HI 65
#define R33_LO 66
#define R33_HI 67
#define R34_LO 68
#define R34_HI 69
#define R35_LO 70
#define R35_HI 71
#define R36_LO 72
#define R36_HI 73
#define R37_LO 74
#define R37_HI 75
#define R38_LO 76
#define R38_HI 77
#define R39_LO 78
#define R39_HI 79
#define R40_LO 80
#define R40_HI 81
#define R41_LO 82
#define R41_HI 83
#define R42_LO 84
#define R42_HI 85
#define R43_LO 86
#define R43_HI 87
#define R44_LO 88
#define R44_HI 89
#define R45_LO 90
#define R45_HI 91
#define R46_LO 92
#define R46_HI 93
#define R47_LO 94
#define R47_HI 95
#define R48_LO 96
#define R48_HI 97
#define R49_LO 98
#define R49_HI 99
#define R50_LO 100
#define R50_HI 101
#define R51_LO 102
#define R51_HI 103
#define R52_LO 104
#define R52_HI 105
#define R53_LO 106
#define R53_HI 107
#define R54_LO 108
#define R54_HI 109
#define R55_LO 110
#define R55_HI 111
#define R56_LO 112
#define R56_HI 113
#define R57_LO 114
#define R57_HI 115
#define R58_LO 116
#define R58_HI 117
#define R59_LO 118
#define R59_HI 119
#define R60_LO 120
#define R60_HI 121
#define R61_LO 122
#define R61_HI 123
#define R62_LO 124
#define R62_HI 125
#define R63_LO 126
#define R63_HI 127

enum cnt_id_t { UNP0 = 1, UNP1 = 2, PCK0 = 4 };

#ifdef CPU_JAWBRIDGE
#define TENSIX_MAX_KERNEL_LOOP_COUNT 128u
#else
#define TENSIX_MAX_KERNEL_LOOP_COUNT 65535u
#endif

/////////////

template <class T>
inline T bitmask(unsigned int bits) {
    static_assert(!std::numeric_limits<T>::is_signed, "bitmask type must be unsigned");

    // just a limitation of the implementation:
    // we use digits() to see if 1 << bits is representable
    static_assert(std::numeric_limits<T>::radix == 2, "bitmask type must be radix 2");

    return (bits == std::numeric_limits<T>::digits) ? std::numeric_limits<T>::max() : (T(1) << bits) - 1;
}

template <class T>
inline typename std::make_unsigned<T>::type pack_field(T x, unsigned int to_shift) {
    using u_T = typename std::make_unsigned<T>::type;
    u_T u_x(x);

    // verify that no bits are shifted away
    // assert((u_x & (std::numeric_limits<u_T>::max() << (std::numeric_limits<u_T>::digits - to_shift))) == 0);

    return u_x << to_shift;
}

template <class T>
inline typename std::make_unsigned<T>::type pack_field(T x, unsigned int bits, unsigned int to_shift) {
    typename std::make_unsigned<T>::type u_x(x);

    // assert((u_x & ~bitmask<T>(bits)) == 0);
    // assert(bits + to_shift <= std::numeric_limits<T>::digits);

    return u_x << to_shift;
}

template <class T>
inline typename std::make_unsigned<T>::type pack_field(
    T x, unsigned int bits, unsigned int from_shift, unsigned int to_shift) {
    typename std::make_unsigned<T>::type u_x(x);

    // assert(bits + to_shift <= std::numeric_limits<T>::digits);
    // assert(bits + from_shift <= std::numeric_limits<T>::digits);

    return ((u_x >> from_shift) & bitmask<T>(bits)) << to_shift;
}

#define IRQ_HANDLER __attribute__((interrupt("machine"), noinline, used))

#define ADC_FLOP_ADDR(addr, counter_id, channel_index, dimension_index)                  \
    do {                                                                                 \
        if ((channel_index == 0) && (counter_id == UNP0) && (dimension_index == 0))      \
            addr = 0;                                                                    \
        else if ((channel_index == 0) && (counter_id == UNP0) && (dimension_index == 1)) \
            addr = 1;                                                                    \
        else if ((channel_index == 0) && (counter_id == UNP0) && (dimension_index == 2)) \
            addr = 2;                                                                    \
        else if ((channel_index == 0) && (counter_id == UNP0) && (dimension_index == 3)) \
            addr = 3;                                                                    \
        else if ((channel_index == 0) && (counter_id == UNP1) && (dimension_index == 0)) \
            addr = 8;                                                                    \
        else if ((channel_index == 0) && (counter_id == UNP1) && (dimension_index == 1)) \
            addr = 9;                                                                    \
        else if ((channel_index == 0) && (counter_id == UNP1) && (dimension_index == 2)) \
            addr = 10;                                                                   \
        else if ((channel_index == 0) && (counter_id == UNP1) && (dimension_index == 3)) \
            addr = 11;                                                                   \
        else if ((channel_index == 0) && (counter_id == PCK0) && (dimension_index == 0)) \
            addr = 16;                                                                   \
        else if ((channel_index == 0) && (counter_id == PCK0) && (dimension_index == 1)) \
            addr = 17;                                                                   \
        else if ((channel_index == 0) && (counter_id == PCK0) && (dimension_index == 2)) \
            addr = 18;                                                                   \
        else if ((channel_index == 0) && (counter_id == PCK0) && (dimension_index == 3)) \
            addr = 19;                                                                   \
        else if ((channel_index == 1) && (counter_id == UNP0) && (dimension_index == 0)) \
            addr = 32;                                                                   \
        else if ((channel_index == 1) && (counter_id == UNP0) && (dimension_index == 1)) \
            addr = 33;                                                                   \
        else if ((channel_index == 1) && (counter_id == UNP0) && (dimension_index == 2)) \
            addr = 34;                                                                   \
        else if ((channel_index == 1) && (counter_id == UNP0) && (dimension_index == 3)) \
            addr = 35;                                                                   \
        else if ((channel_index == 1) && (counter_id == UNP1) && (dimension_index == 0)) \
            addr = 40;                                                                   \
        else if ((channel_index == 1) && (counter_id == UNP1) && (dimension_index == 1)) \
            addr = 41;                                                                   \
        else if ((channel_index == 1) && (counter_id == UNP1) && (dimension_index == 2)) \
            addr = 42;                                                                   \
        else if ((channel_index == 1) && (counter_id == UNP1) && (dimension_index == 3)) \
            addr = 43;                                                                   \
        else if ((channel_index == 1) && (counter_id == PCK0) && (dimension_index == 0)) \
            addr = 48;                                                                   \
        else if ((channel_index == 1) && (counter_id == PCK0) && (dimension_index == 1)) \
            addr = 49;                                                                   \
        else if ((channel_index == 1) && (counter_id == PCK0) && (dimension_index == 2)) \
            addr = 50;                                                                   \
        else if ((channel_index == 1) && (counter_id == PCK0) && (dimension_index == 3)) \
            addr = 51;                                                                   \
        else                                                                             \
            addr = 0;                                                                    \
    } while (0)

#endif
