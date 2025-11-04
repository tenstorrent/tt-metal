// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cfg_defines.h"
#include "ckernel.h"

inline void device_setup()
{
#if defined(ARCH_WORMHOLE)
    constexpr std::uint32_t TRISC0_START_ADDRESS_PTR              = 0x16DFF0;
    constexpr std::uint32_t TRISC1_START_ADDRESS_PTR              = 0x16DFF4;
    constexpr std::uint32_t TRISC2_START_ADDRESS_PTR              = 0x16DFF8;
    volatile std::uint32_t* const trisc0_start_address            = reinterpret_cast<volatile std::uint32_t*>(TRISC0_START_ADDRESS_PTR);
    volatile std::uint32_t* const trisc1_start_address            = reinterpret_cast<volatile std::uint32_t*>(TRISC1_START_ADDRESS_PTR);
    volatile std::uint32_t* const trisc2_start_address            = reinterpret_cast<volatile std::uint32_t*>(TRISC2_START_ADDRESS_PTR);
    volatile uint tt_reg_ptr* cfg_regs                            = reinterpret_cast<volatile uint tt_reg_ptr*>(TENSIX_CFG_BASE);
    cfg_regs[TRISC_RESET_PC_SEC0_PC_ADDR32]                       = *trisc0_start_address;
    cfg_regs[TRISC_RESET_PC_SEC1_PC_ADDR32]                       = *trisc1_start_address;
    cfg_regs[TRISC_RESET_PC_SEC2_PC_ADDR32]                       = *trisc2_start_address;
    cfg_regs[TRISC_RESET_PC_OVERRIDE_Reset_PC_Override_en_ADDR32] = 0b111;
#endif
#if defined(ARCH_BLACKHOLE) && !defined(ARCH_QUASAR) // Ugly hack for now
    ckernel::reg_write(RISCV_DEBUG_REG_DEST_CG_CTRL, 0);
#endif
#if defined(ARCH_BLACKHOLE) || defined(ARCH_QUASAR)
    TTI_ZEROACC(ckernel::p_zeroacc::CLR_ALL, 0, 0, 1, 0);
#else
    TTI_ZEROACC(ckernel::p_zeroacc::CLR_ALL, 0, 0);
#endif

// Enable CC stack
#if defined(ARCH_QUASAR)
    TTI_SFPENCC(3, 10);
#else
    TTI_SFPENCC(3, 0, 0, 10);
#endif

    TTI_NOP;

    // Set default sfpu constant register state
    TTI_SFPCONFIG(0, 11, 1); // loading -1 to LREG11 where sfpi expects it

#ifndef ARCH_QUASAR
    // Initialize tensix semaphores
    ckernel::t6_semaphore_init(ckernel::semaphore::UNPACK_TO_DEST, 0, 1);
    ckernel::t6_semaphore_init(ckernel::semaphore::MATH_DONE, 0, 1);
    ckernel::t6_semaphore_init(ckernel::semaphore::PACK_DONE, 0, 1);
#endif
}

inline void clear_trisc_soft_reset()
{
#ifdef ARCH_QUASAR
    constexpr uint32_t TRISC_SOFT_RESET_MASK = 0x3000;
#else
    constexpr uint32_t TRISC_SOFT_RESET_MASK = 0x7000;
#endif

    uint32_t soft_reset = ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0);
    soft_reset &= ~TRISC_SOFT_RESET_MASK;
    ckernel::reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset);
}
