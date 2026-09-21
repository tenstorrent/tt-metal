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
// even by accident: the indices below are out of the way by construction rather than by convention. They come
// out of reset at 0 with max 15, and the arrival drain leaves the arrival count at 0 after every rendezvous, so
// a killed run cannot strand a count that the next one would misread as an arrival.
//
// Every peer has its own release level. They date from the token protocol, where one shared count let a peer
// that had already reached the next rendezvous (the idle sfpu stub) take the token meant for a slower peer; a
// level cannot be taken, but the separate semaphores are free here and keep the peers independent.
constexpr std::uint8_t ARRIVE_SEM       = 8;
constexpr std::uint8_t RELEASE_SEM_BASE = 9; // unpack 9, math 10, sfpu 11

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
// A PC buffer load has to land in a register before the next one is issued; a second load while the first is
// still outstanding hangs the TRISC on Blackhole, and semaphore_post reads the semaphore for its assert.
__attribute__((always_inline)) inline std::uint32_t settled(std::uint32_t value)
{
    asm volatile("mv %0, %0" : "+r"(value));
    return value;
}

// Flip a release level between 0 and 1. Only the action thread ever writes a release semaphore.
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

// The release is a level the action thread flips, not a token a peer consumes. With tokens, two peers sharing one
// count could both read 1 while the action thread was still posting the second token, both issue SEMGET, and the
// second get floored at 0: that thread walked on with a token left behind, and the next rendezvous released a
// peer early. The LLK assert in semaphore_get caught it on a Wormhole fast tilize kernel, once in 47k kernels. A
// peer now records the level before it announces its arrival, and the action thread cannot flip until every peer
// has arrived, so the change is never missed and nothing is consumed; a stale level left by a killed run is
// harmless. Polling reads the PC buffer, never the L1 being measured, which is what keeps a waiting thread out
// of the numbers: an L1 rendezvous cost the Quasar unpack windows up to 5% until semaphores replaced it.
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

    ckernel::fence_compiler();
}

__attribute__((always_inline)) inline void rendezvous(bool is_action_thread)
{
    rendezvous(is_action_thread, [] {});
}

#endif // LLK_BARRIER_ON_TRISC

} // namespace llk_barrier
