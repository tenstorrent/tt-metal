// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "cfg_defines.h"
#include "ckernel.h"
#include "ckernel_helper.h"
#if defined(ARCH_QUASAR)
#include "ckernel_trisc_common.h"
#endif

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

// Typed-region declaration markers (compiler prgm-const freedom proof; trusted
// like sfprawlreg_access, CRAQ is the check).  The markers emit no instruction
// word, but they live in the IR as volatile ghosts and can shift surrounding
// scalar scheduling, so they are opt-in: builds that enable the prgm-const
// optimization define LLK_ENABLE_TTREGION_MARKERS next to the -m flag; every
// other build (and any toolchain without the builtins) compiles the plain
// unmarked code, byte-identically.
#ifndef TT_LLK_TTREGION_BEGIN
#if defined(LLK_ENABLE_TTREGION_MARKERS) && defined(__has_builtin)
#if __has_builtin(__builtin_rvtt_ttregion_begin)
#define TT_LLK_TTREGION_BEGIN(config_write_mask, reserved) __builtin_rvtt_ttregion_begin((config_write_mask), (reserved))
#define TT_LLK_TTREGION_END()                              __builtin_rvtt_ttregion_end()
#endif
#endif
#ifndef TT_LLK_TTREGION_BEGIN
#define TT_LLK_TTREGION_BEGIN(config_write_mask, reserved)
#define TT_LLK_TTREGION_END()
#endif
#endif

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
        // Typed effects declaration for the init-array walk: the constructors
        // are this translation unit's own scanned bodies -- the indirect call
        // writes no SFPCONFIG destination, no PRGM register, and no LaneConfig
        // (mask 0).  Markers sit in the loop-body block with the call they cover.
        TT_LLK_TTREGION_BEGIN(0, 0);
        (*temp_constructor)();
        TT_LLK_TTREGION_END();
    }
}

void _init(void)
{
}

void _fini(void)
{
}

using mailbox_t = volatile std::uint32_t*;

#ifdef ARCH_WORMHOLE
constexpr std::uint32_t TRISC_START_BASE    = 0x16DFF0;
constexpr std::uint32_t TRISC_CONFIG_REGS[] = {TRISC_RESET_PC_SEC0_PC_ADDR32, TRISC_RESET_PC_SEC1_PC_ADDR32, TRISC_RESET_PC_SEC2_PC_ADDR32};

mailbox_t trisc_start_addresses = reinterpret_cast<mailbox_t>(TRISC_START_BASE);
#endif

TT_ALWAYS_INLINE void device_setup()
{
#if defined(ARCH_WORMHOLE)
    // Use array-based initialization for consecutive TRISC addresses
    volatile std::uint32_t tt_reg_ptr* cfg_regs = reinterpret_cast<volatile std::uint32_t tt_reg_ptr*>(TENSIX_CFG_BASE);

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
    TTI_SEMINIT(1, 0, 0, ckernel::trisc::semaphore::t6_sem(ckernel::trisc::semaphore::PACK_UNPACK));
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

constexpr std::uint32_t TRISC_SOFT_RESET_MASK = 0x7000;

#ifndef ARCH_QUASAR

// Include functions that have ckernel definitions of underlying callees

TT_ALWAYS_INLINE void enable_branch_prediction()
{
    volatile std::uint32_t* tt_reg_ptr cfg_ptr = ckernel::get_cfg_pointer();
    cfg_ptr[DISABLE_RISC_BP_Disable_main_ADDR32] &= ~DISABLE_RISC_BP_Disable_main_MASK;
}

TT_ALWAYS_INLINE void disable_branch_prediction()
{
    volatile std::uint32_t* tt_reg_ptr cfg_ptr = ckernel::get_cfg_pointer();
    cfg_ptr[DISABLE_RISC_BP_Disable_main_ADDR32] |= DISABLE_RISC_BP_Disable_main_MASK;
}

template <typename T, typename U, typename = std::enable_if_t<std::is_trivially_copyable_v<T> && std::is_trivially_assignable_v<T&, U> > >
inline void commit_store(volatile T* ptr, U&& val)
{
    ckernel::store_blocking(ptr, val);

    do
    {
        asm volatile("nop");
    } while (ckernel::load_blocking(ptr) != val);
}

#endif

TT_ALWAYS_INLINE void clear_trisc_soft_reset()
{
    std::uint32_t soft_reset = ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0);
    soft_reset &= ~TRISC_SOFT_RESET_MASK;
    ckernel::reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset);

    do
    {
        asm volatile("nop");
    } while (ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0) != soft_reset);
}

TT_ALWAYS_INLINE void set_triscs_soft_reset()
{
    std::uint32_t soft_reset = ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0);
    soft_reset |= TRISC_SOFT_RESET_MASK;
    ckernel::reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, soft_reset);
    do
    {
        asm volatile("nop");
    } while (ckernel::reg_read(RISCV_DEBUG_REG_SOFT_RESET_0) != soft_reset);
}
