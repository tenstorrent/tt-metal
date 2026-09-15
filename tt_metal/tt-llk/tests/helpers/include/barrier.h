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

// Quasar has 32 Tensix semaphores (tt_tensix_pkg.sv SEM_COUNT), and a TRISC reaches all of them through the
// PC buffer (words 32 to 63), which is the path semaphore_post/get/read below take. The Tensix instructions
// see them as four banks of eight, and t6_sem() can only build a bank-0 mask, so no LLK op can reach bank 1
// even by accident: the pair below is out of the way by construction rather than by convention. Both indices
// come out of reset at 0 with max 15, and the arrival drain leaves them at 0 after every rendezvous, so a
// killed run cannot strand a count that the next one would misread as an arrival.
constexpr std::uint8_t ARRIVE_SEM  = 8;
constexpr std::uint8_t RELEASE_SEM = 9;

#else

// The only two indices no LLK op uses. Reserved below for the rest of the translation unit, because
// the arrival drain would eat the token of any driver that also posted one.
constexpr std::uint8_t ARRIVE_SEM  = ckernel::semaphore::PACK_DONE;
constexpr std::uint8_t RELEASE_SEM = ckernel::semaphore::UNPACK_OPERAND_SYNC;
#pragma GCC poison PACK_DONE UNPACK_OPERAND_SYNC

#endif

// A consumed token, not a level to observe, so a peer that samples late still finds its release. Polling
// reads the PC buffer, never the L1 being measured, which is what keeps a waiting thread out of the
// numbers: an L1 rendezvous cost the Quasar unpack windows up to 5% until this replaced it.
template <typename Action>
__attribute__((always_inline)) inline void rendezvous(bool is_action_thread, Action action)
{
    ckernel::fence_compiler();

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

        for (std::uint32_t i = 0; i < NUM_THREADS - 1; ++i)
        {
            ckernel::semaphore_post(RELEASE_SEM);
        }
    }
    else
    {
        ckernel::semaphore_post(ARRIVE_SEM);
        while (ckernel::semaphore_read(RELEASE_SEM) == 0)
        {
        }
        ckernel::semaphore_get(RELEASE_SEM);
    }

    ckernel::fence_compiler();
}

__attribute__((always_inline)) inline void rendezvous(bool is_action_thread)
{
    rendezvous(is_action_thread, [] {});
}

#endif // LLK_BARRIER_ON_TRISC

} // namespace llk_barrier
