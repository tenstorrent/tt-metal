// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "ckernel_instr_params.h"
#include "ckernel_ops.h"
#include "ckernel_structs.h"
#include "dev_mem_map.h"
#include "risc_attribs.h"
#include "tensix.h"

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
    TTI_SFPLOADI(ckernel::p_sfpu::LREG0, 0xA, 0xbf80); // -1.0f -> LREG0
    TTI_SFPCONFIG(0, 11, 0);                           // LREG0 -> LREG11

    // Initialize tensix semaphores
    TTI_SEMINIT(1, 0, ckernel::semaphore::UNPACK_TO_DEST);
    TTI_SEMINIT(1, 0, ckernel::semaphore::MATH_DONE);
    TTI_SEMINIT(1, 0, ckernel::semaphore::PACK_DONE);
}

int main()
{
    device_setup();
    // Use a volatile variable to prevent the compiler from optimizing away the loop
    volatile bool run = true;

    // Infinite loop
    while (run)
    {
    }
}
