// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ckernel.h"

// XMOV programming through C kernels

namespace ckernel
{

typedef struct
{
    uint32_t instrn          : 8;
    uint32_t src_offset_addr : 8;
    uint32_t dst_addr        : 8;
    uint32_t xfer_size       : 6;
    uint32_t xfer_dir        : 1; // 0 - l1->cfgreg, 1 - l1->l1
    uint32_t no_params       : 1; // 0 - use xmov inputs from param registers, 1 - compact move. all inputs are embedded into instruction
} risc_compact_mov_instrn_t;

typedef union
{
    uint32_t val;
    risc_compact_mov_instrn_t f;
} risc_compact_mov_instrn_u;

inline void xmov_set_base(const uint32_t l1_base_addr_16B)
{
    volatile uint *XMOV_L1_BASE = reinterpret_cast<volatile uint *>(RISCV_TDMA_REG_XMOV_L1_BASE_ADDR);
    // Program mover L1 base to the command base
    XMOV_L1_BASE[0] = l1_base_addr_16B;
}

inline void xmov_cfg_instr_set(
    risc_compact_mov_instrn_u &risc_mov_instrn, const uint32_t l1_offset_16B, const uint32_t reg_addr32, const uint32_t xfer_size = 1)
{
    risc_mov_instrn.val               = 0;
    risc_mov_instrn.f.instrn          = 0x40;                      // mov instruction
    risc_mov_instrn.f.src_offset_addr = l1_offset_16B;             // final l1 address is src_offset_addr + RISCV_TDMA_REG_XMOV_L1_BASE_ADDR
    risc_mov_instrn.f.dst_addr        = cfg_addr(reg_addr32) >> 2; // movcfg address, 16B aligned
    risc_mov_instrn.f.xfer_size       = xfer_size;                 // transfer size in 16B chunks
    risc_mov_instrn.f.xfer_dir        = 0;                         // l1->cfgreg
    risc_mov_instrn.f.no_params       = 0x1;                       // compact move
}

inline void xmov_cfg_instr_program(const risc_compact_mov_instrn_t xmov, const uint32_t l1_offset_16B, const uint32_t reg_addr32)
{
    // Program tile descriptor using fast XMOV path
    volatile uint *XMOV_CMD = reinterpret_cast<volatile uint *>(RISCV_TDMA_REG_COMMAND_ADDR);
    risc_compact_mov_instrn_u risc_mov_instrn;
    risc_mov_instrn.f                 = xmov;
    risc_mov_instrn.f.src_offset_addr = l1_offset_16B;             // final l1 address is src_offset_addr + RISCV_TDMA_REG_XMOV_L1_BASE_ADDR
    risc_mov_instrn.f.dst_addr        = cfg_addr(reg_addr32) >> 2; // movcfg address, 16B aligned
    XMOV_CMD[0]                       = risc_mov_instrn.val;
}

inline void xmov_cfg_program(const uint32_t l1_offset_16B, const uint32_t reg_addr32, const uint32_t xfer_size = 1)
{
    volatile uint *XMOV_CMD = reinterpret_cast<volatile uint *>(RISCV_TDMA_REG_COMMAND_ADDR);
    // Program tile descriptor using fast XMOV path
    risc_compact_mov_instrn_u risc_mov_instrn;
    xmov_cfg_instr_set(risc_mov_instrn, l1_offset_16B, reg_addr32, xfer_size);
    XMOV_CMD[0] = risc_mov_instrn.val;
}

inline void xmov_l1_to_l1_non_compact(const uint32_t src_addr_16B, const uint32_t dst_addr_16B, const uint32_t xfer_size_16B = 1)
{
    volatile uint *XMOV_CMD  = reinterpret_cast<volatile uint *>(RISCV_TDMA_REG_COMMAND_ADDR);
    volatile uint *SRC_ADDR  = reinterpret_cast<volatile uint *>(RISCV_TDMA_REG_XMOV_SRC_ADDR);
    volatile uint *DST_ADDR  = reinterpret_cast<volatile uint *>(RISCV_TDMA_REG_XMOV_DST_ADDR);
    volatile uint *SIZE_ADDR = reinterpret_cast<volatile uint *>(RISCV_TDMA_REG_XMOV_SIZE);
    volatile uint *DIR_ADDR  = reinterpret_cast<volatile uint *>(RISCV_TDMA_REG_XMOV_DIRECTION);

    SRC_ADDR[0]  = src_addr_16B;
    DST_ADDR[0]  = dst_addr_16B;
    SIZE_ADDR[0] = xfer_size_16B;
    DIR_ADDR[0]  = 3; // L1 to L1
    risc_compact_mov_instrn_u risc_mov_instrn;
    risc_mov_instrn.f.instrn    = 0x40; // mov instruction
    risc_mov_instrn.f.no_params = 0;    // Use xmov inputs from param registers
    XMOV_CMD[0]                 = risc_mov_instrn.val;
}

inline void xmov_wait_till_idle()
{
    volatile uint *XMOV_STATUS = reinterpret_cast<volatile uint *>(RISCV_TDMA_REG_STATUS);
    while (XMOV_STATUS[0] & 0x1)
    {
    }
}

} // namespace ckernel
