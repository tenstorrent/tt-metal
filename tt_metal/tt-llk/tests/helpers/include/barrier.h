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

// The release is a level, sampled by each peer before it arrives and flipped once every peer has; a token on a
// shared count could be consumed twice and release a peer early. Waiters poll the PC buffer, not the measured L1.
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
