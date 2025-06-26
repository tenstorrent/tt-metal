// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel.h"
#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_structs.h"
#include "dev_mem_map.h"
#include "risc_attribs.h"
#include "tensix.h"

static constexpr uint32_t TRISC_SOFT_RESET_MASK = 0x7000;

void device_setup()
{
    volatile tt_reg_ptr std::uint32_t* cfg_regs = reinterpret_cast<std::uint32_t*>(TENSIX_CFG_BASE);

#ifdef ARCH_BLACKHOLE
    *reinterpret_cast<volatile std::uint32_t*>(RISCV_DEBUG_REG_DEST_CG_CTRL) = 0;
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

void clear_trisc_soft_reset()
{
    uint32_t soft_reset = ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0);
    soft_reset &= ~TRISC_SOFT_RESET_MASK;
    ckernel::reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset);
}

int main()
{
    device_setup();

    // Release reset of triscs here in order to achieve brisc <-> trisc synchronization
    clear_trisc_soft_reset();

    // Use a volatile variable to prevent the compiler from optimizing away the loop
    volatile bool run = true;

    // Infinite loop
    while (run)
    {
    }
}
