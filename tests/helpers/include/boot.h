// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "cfg_defines.h"
#include "ckernel.h"

// C-runtime related linker symbols
extern volatile char __ldm_bss_start[], __ldm_bss_end[];
extern volatile char __loader_init_start[], __loader_init_end[];
extern volatile char __ldm_data_start[], __ldm_data_end[];
extern const std::uint32_t __stack_top[];
extern void (*__init_array_start[])(void);
extern void (*__init_array_end[])(void);

// TODO @ajankovicTT find out why GCC generates unwinding tables on coverage,
// even though -fno-asynchronous-unwind-tables -fno-exceptions flags are set
void* __gxx_personality_v0;

__attribute__((no_profile_instrument_function)) TT_ALWAYS_INLINE void do_crt0()
{
    asm volatile(
        ".option push\n"
        ".option norelax\n"
        "la gp, __global_pointer$\n"
        ".option pop" ::
            : "memory");

    // Set stack pointer
    asm volatile("la sp, %0" : : "i"(__stack_top) : "memory");

    // Initialize .bss
    for (volatile std::uint32_t* p = (volatile std::uint32_t*)__ldm_bss_start; p < (volatile std::uint32_t*)__ldm_bss_end; p++)
    {
        *p = 0;
    }

    // Copy .loader_init to .ldm_data
    if ((std::uint32_t)__loader_init_start != (std::uint32_t)__loader_init_end)
    {
        volatile std::uint32_t* src = (volatile std::uint32_t*)__loader_init_start;
        volatile std::uint32_t* dst = (volatile std::uint32_t*)__ldm_data_start;
        volatile std::uint32_t* end = (volatile std::uint32_t*)__ldm_data_end;
        while (dst < end)
        {
            *dst++ = *src++;
        }
    }

    // Execute global constructors
    for (void (**temp_constructor)(void) = __init_array_start; temp_constructor < __init_array_end; temp_constructor++)
    {
        (*temp_constructor)();
    }
}

void _init(void)
{
}

void _fini(void)
{
}

inline void device_setup()
{
#if defined(ARCH_WORMHOLE)
    // Use array-based initialization for consecutive TRISC addresses
    constexpr std::uint32_t TRISC_START_BASE    = 0x16DFF0;
    constexpr std::uint32_t TRISC_CONFIG_REGS[] = {TRISC_RESET_PC_SEC0_PC_ADDR32, TRISC_RESET_PC_SEC1_PC_ADDR32, TRISC_RESET_PC_SEC2_PC_ADDR32};

    volatile std::uint32_t* const trisc_start_addresses = reinterpret_cast<volatile std::uint32_t*>(TRISC_START_BASE);
    volatile std::uint32_t tt_reg_ptr* cfg_regs         = reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(TENSIX_CFG_BASE);

    for (unsigned int i = 0; i < std::size(TRISC_CONFIG_REGS); ++i)
    {
        cfg_regs[TRISC_CONFIG_REGS[i]] = trisc_start_addresses[i];
    }
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

#if defined(ARCH_QUASAR)
    // Reset all dest dvalid bits for all clients
    TTI_CLEARDVALID(0, 0, 0xf, 0xf, 0, 0);
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
    constexpr std::uint32_t TRISC_SOFT_RESET_MASK = 0x3000;
#else
    constexpr std::uint32_t TRISC_SOFT_RESET_MASK = 0x7000;
#endif

    volatile std::uint32_t* reset_before = reinterpret_cast<std::uint32_t*>(0x64FF0);
    volatile std::uint32_t* reset_after  = reinterpret_cast<std::uint32_t*>(0x64FF4);

    std::uint32_t soft_reset = ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0);
    *reset_before            = soft_reset;

    soft_reset &= ~TRISC_SOFT_RESET_MASK;
    ckernel::reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset);

    soft_reset   = ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0);
    *reset_after = soft_reset;
}
