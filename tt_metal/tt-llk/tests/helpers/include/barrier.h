// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel.h"
#if !defined(ARCH_QUASAR)
#include "ckernel_structs.h" // semaphore indices; Quasar names its own set in ckernel_trisc_common.h
#endif

// Deliberately outside every build guard, so both builds compile the same barrier.

namespace llk_barrier
{

// counters.h includes this before its own build guard, so it compiles for BRISC too, which has no thread identity.
#if defined(LLK_TRISC_UNPACK) || defined(LLK_TRISC_MATH) || defined(LLK_TRISC_PACK) || defined(LLK_TRISC_ISOLATE_SFPU)
#define LLK_BARRIER_ON_TRISC 1
#endif

#if defined(ARCH_QUASAR)
constexpr std::uint32_t NUM_THREADS = 4; // unpack, math, pack, sfpu
#else
constexpr std::uint32_t NUM_THREADS = 3; // unpack, math, pack
#endif

#if defined(LLK_BARRIER_ON_TRISC)

// Lives here, not in profiler.h, because the barrier needs it first; profiler.h aliases TRISC_ID onto it.
#if defined(LLK_TRISC_UNPACK)
constexpr std::uint32_t THREAD_ID = 0;
#elif defined(LLK_TRISC_MATH)
constexpr std::uint32_t THREAD_ID = 1;
#elif defined(LLK_TRISC_PACK)
constexpr std::uint32_t THREAD_ID = 2;
#else
constexpr std::uint32_t THREAD_ID = 3;
#endif

// Fixed, not per-run-type: letting it vary is how the two builds ended up releasing from different threads.
constexpr bool is_action_thread()
{
#if defined(LLK_TRISC_PACK)
    return true;
#else
    return false;
#endif
}

#if defined(ARCH_QUASAR)

// Semaphores 8 to 11 sit in bank 1, which t6_sem() cannot address, so no LLK op can reach them even by accident.
// The spare semaphores are free, so each peer waits on its own release level and the peers stay independent.
constexpr std::uint8_t ARRIVE_SEM       = 8;
constexpr std::uint8_t RELEASE_SEM_BASE = 9; // unpack 9, math 10, sfpu 11; pack, the action thread, flips them

constexpr std::uint8_t release_sem_of(std::uint32_t peer)
{
    return static_cast<std::uint8_t>(RELEASE_SEM_BASE + peer);
}

constexpr std::uint8_t my_release_sem()
{
#if defined(LLK_TRISC_UNPACK)
    return release_sem_of(0);
#elif defined(LLK_TRISC_MATH)
    return release_sem_of(1);
#elif defined(LLK_TRISC_ISOLATE_SFPU)
    return release_sem_of(2);
#else
    return release_sem_of(0); // the action thread never waits on one
#endif
}

#else

// The only two indices no LLK op uses. Reserved below for the rest of the translation unit, because
// the arrival drain would eat the token of any driver that also posted one.
constexpr std::uint8_t ARRIVE_SEM  = ckernel::semaphore::PACK_DONE;
constexpr std::uint8_t RELEASE_SEM = ckernel::semaphore::UNPACK_OPERAND_SYNC;

// No third free semaphore, so the two peers share one release level. Sharing is safe because nobody consumes
// it: the action thread flips it once per rendezvous and each peer waits for it to differ from what it saw.
constexpr std::uint8_t release_sem_of(std::uint32_t)
{
    return RELEASE_SEM;
}

constexpr std::uint8_t my_release_sem()
{
    return RELEASE_SEM;
}

#pragma GCC poison PACK_DONE UNPACK_OPERAND_SYNC

#endif

namespace detail
{
// A PC buffer load must land before the next one is issued: two outstanding loads hang the TRISC on Blackhole,
// and semaphore_post reads the semaphore for its assert right after this one.
__attribute__((always_inline)) inline std::uint32_t settled(std::uint32_t value)
{
    asm volatile("mv %0, %0" : "+r"(value));
    return value;
}

// Only the action thread writes release semaphores, so its read-then-write flip cannot race a peer.
__attribute__((always_inline)) inline void flip(std::uint8_t sem)
{
    if (settled(ckernel::semaphore_read(sem)) == 0)
    {
        ckernel::semaphore_post(sem);
    }
    else
    {
        ckernel::semaphore_get(sem);
    }
}
} // namespace detail

// Park kinds (Wormhole perf builds): 1 KiB in run_kernel, aligned TILE_LOOP restart, unfilled in main.
constexpr int PARK_SIZED = 0;
constexpr int PARK_LOOP  = 1;
constexpr int PARK_PLAIN = 2;

#if defined(LLK_DBG_BARRIER) // Wormhole perf builds: BRISC restarts every thread from a flushed pipeline
namespace detail
{
#if defined(LLK_PERF_OOL)
// LLK_PERF_OOL threads (test_config.py) run INIT out of line (perf.h); their parks restart 512 B aligned, the TILE_LOOP
// one after llk_loop_pad bytes (perf/layout.py).
template <int MODE = PARK_SIZED>
__attribute__((always_inline)) inline void park()
{
    volatile std::uint32_t* reset_pc = reinterpret_cast<volatile std::uint32_t*>(TENSIX_CFG_BASE) + TRISC_RESET_PC_SEC0_PC_ADDR32 + THREAD_ID;
    std::uint32_t scratch;
    asm volatile(
        "la    %[s], 1f\n\t"
        "sw    %[s], 0(%[rpc])\n\t"
        "lw    %[s], 0(%[rpc])\n\t"
        "andi  %[s], %[s], 0\n\t"
        "lw    %[s], 0(%[pcb])\n\t"
        "andi  %[s], %[s], 0\n\t"
        ".word 0x00100073\n\t"
        ".rept 16\n\t"
        ".word 0x00000013\n\t"
        ".endr\n"
        ".balign 512\n" // the code after 1f keeps its address mod 512 B (branch predictor hash, icache sets) as code before it moves
        ".ifndef llk_loop_pad\n\t"
        ".set llk_loop_pad, 0\n"
        ".endif\n"
        ".rept (llk_loop_pad / 4) * %[on]\n\t"
        "nop\n\t"
        ".endr\n"
        "1:\n\t"
        "lw    %[s], 0(%[pcb])\n\t"
        "andi  %[s], %[s], 0\n\t"
        : [s] "=&r"(scratch)
        : [rpc] "r"(reset_pc), [pcb] "r"(ckernel::pc_buf_base), [on] "i"(MODE == PARK_LOOP ? 1 : 0)
        : "memory");
}
#else
// The park body is an assembler macro, so the compiler sees a one line asm: its size estimate for the branches around
// the park stays as without the barrier (it decides short or long branches from those estimates).
asm(R"ASM(
.macro llk_park mode, pcb, rpc, size
.option push
.option norelax
2:
    addi  sp, sp, -16
    sw    t0, 0(sp)
    sw    t1, 4(sp)
    li    t1, \pcb
    sw    zero, 4(t1)
    lw    t0, 4(t1)
    andi  t0, t0, 0
    li    t1, \rpc
    la    t0, 1f
    sw    t0, 0(t1)
    lw    t0, 0(t1)
    andi  t0, t0, 0
    li    t1, \pcb
    lw    t0, 0(t1)
    andi  t0, t0, 0
    .word 0x00100073
    .rept 16
    .word 0x00000013
    .endr
.if \mode == 1
.option pop
    .balign 1024
    .ifndef llk_loop_pad
    .set llk_loop_pad, 0
    .endif
    .rept llk_loop_pad / 4
    nop
    .endr
1:
    lw    t0, 0(t1)
    andi  t0, t0, 0
    lw    t0, 0(sp)
    lw    t1, 4(sp)
    addi  sp, sp, 16
.else
.if \mode == 0
    .rept (\size - 20 - (. - 2b)) / 4
    nop
    .endr
.endif
1:
    lw    t0, 0(t1)
    andi  t0, t0, 0
    lw    t0, 0(sp)
    lw    t1, 4(sp)
    addi  sp, sp, 16
.option pop
.endif
.endm
)ASM");

// Park at the BRISC barrier server (brisc.cpp); with no register operands it compiles like a plain memory barrier.
// The TILE_LOOP park restarts 1 KiB aligned plus llk_loop_pad (perf/layout.py); the other parks span `size` bytes.
template <int MODE = PARK_SIZED>
__attribute__((always_inline)) inline void park()
{
    asm volatile("llk_park %[mode], %[pcb], %[rpc], %[size]"
                 :
                 : [mode] "i"(MODE),
                   [size] "i"(THREAD_ID == 1 ? 512 : 1024), // math: BP hash and its 256 B icache repeat every 512 B
                   [pcb] "i"(PC_BUF_BASE),
                   [rpc] "i"(TENSIX_CFG_BASE + 4 * (TRISC_RESET_PC_SEC0_PC_ADDR32 + THREAD_ID))
                 : "memory");
}
#endif
} // namespace detail
#endif

// The release is a level, sampled by each peer before it arrives and flipped once every peer has; a token on a
// shared count could be consumed twice and release a peer early. Waiters poll the PC buffer, not the measured L1.
template <int LOOP_PAD = PARK_SIZED, typename Action>
__attribute__((always_inline)) inline void rendezvous(bool is_action_thread, Action action)
{
    ckernel::fence_compiler();

#if defined(LLK_DBG_BARRIER) && defined(LLK_PERF_OOL)
    ckernel::tensix_sync();
    if (is_action_thread)
    {
        while (ckernel::semaphore_read(ARRIVE_SEM) < NUM_THREADS - 1)
        {
        }
        while (ckernel::semaphore_read(ARRIVE_SEM) != 0)
        {
            ckernel::semaphore_get(ARRIVE_SEM);
        }
        action();
        detail::flip(RELEASE_SEM); // peers on the inline-INIT path wait for the release level
        ckernel::tensix_sync();
    }
    else
    {
        ckernel::semaphore_post(ARRIVE_SEM);
    }
    detail::park<LOOP_PAD>();
#else

    if (is_action_thread)
    {
        while (ckernel::semaphore_read(ARRIVE_SEM) < NUM_THREADS - 1)
        {
        }
        while (ckernel::semaphore_read(ARRIVE_SEM) != 0)
        {
            ckernel::semaphore_get(ARRIVE_SEM);
        }

        action();

#if defined(ARCH_QUASAR)
        for (std::uint32_t i = 0; i < NUM_THREADS - 1; ++i)
        {
            detail::flip(release_sem_of(i));
        }
#else
        detail::flip(RELEASE_SEM);
#endif
    }
    else
    {
        const std::uint32_t seen = detail::settled(ckernel::semaphore_read(my_release_sem()));
        ckernel::semaphore_post(ARRIVE_SEM);
        while (ckernel::semaphore_read(my_release_sem()) == seen)
        {
        }
    }
#if defined(LLK_DBG_BARRIER)
    detail::park<LOOP_PAD>();
#endif
#endif

    ckernel::fence_compiler();
}

__attribute__((always_inline)) inline void rendezvous(bool is_action_thread)
{
    rendezvous(is_action_thread, [] {});
}

#endif // LLK_BARRIER_ON_TRISC

} // namespace llk_barrier
