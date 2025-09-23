// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"

inline void device_setup()
{
#ifdef ARCH_BLACKHOLE
    ckernel::reg_write(RISCV_DEBUG_REG_DEST_CG_CTRL, 0);
    TTI_ZEROACC(ckernel::p_zeroacc::CLR_ALL, 0, 0, 1, 0);
#else
    TTI_ZEROACC(ckernel::p_zeroacc::CLR_ALL, 0, 0);
#endif

    // Enable CC stack
    TTI_SFPENCC(3, 0, 0, 10);
    TTI_NOP;

    // Set default sfpu constant register state
    TTI_SFPCONFIG(0, 11, 1); // loading -1 to LREG11 where sfpi expects it

    // Initialize tensix semaphores
    TTI_SEMINIT(1, 0, ckernel::semaphore::UNPACK_TO_DEST);
    TTI_SEMINIT(1, 0, ckernel::semaphore::MATH_DONE);
    TTI_SEMINIT(1, 0, ckernel::semaphore::PACK_DONE);
}

inline void clear_trisc_soft_reset()
{
    constexpr uint32_t TRISC_SOFT_RESET_MASK = 0x7000;

    uint32_t soft_reset = ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0);
    soft_reset &= ~TRISC_SOFT_RESET_MASK;
    ckernel::reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset);
}
