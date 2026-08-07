// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel.h"
#if !defined(ARCH_QUASAR)
#include "ckernel_structs.h" // semaphore indices; Quasar names its own set in ckernel_trisc_common.h
#endif

// The one cross-thread rendezvous for the perf harness, outside every build guard so both builds
// compile the same barrier. On a semaphore, not L1: symmetric release, no traffic on what is measured.

namespace llk_barrier
{

// Quasar has no semaphore to spare (MATH_PACK, UNPACK_MATH and PACK_UNPACK all belong to the ops) and
// its counter path is excluded by an #error in counters.h, so it keeps the L1 rendezvous in profiler.h.
#if !defined(ARCH_QUASAR)

constexpr std::uint32_t NUM_THREADS = 3; // unpack, math, pack

// Unused by the LLK library, though some kernels under tests/sources/ do use it as a pack-to-unpack
// handshake; safe today because sync_threads runs once and leaves the count at 0.
constexpr std::uint8_t RENDEZVOUS_SEM = ckernel::semaphore::PACK_DONE;

// Everyone announces by incrementing; the action thread waits for all, runs action(), then drains back
// to zero, and that return to zero is the release. Needs exactly one action thread and all to arrive.
template <typename Action>
__attribute__((always_inline)) inline void rendezvous(bool is_action_thread, Action action)
{
    ckernel::fence_compiler();

    ckernel::semaphore_post(RENDEZVOUS_SEM);

    if (is_action_thread)
    {
        while (ckernel::semaphore_read(RENDEZVOUS_SEM) < NUM_THREADS)
        {
        }
        action();
        while (ckernel::semaphore_read(RENDEZVOUS_SEM) != 0)
        {
            ckernel::semaphore_get(RENDEZVOUS_SEM);
        }
    }
    else
    {
        while (ckernel::semaphore_read(RENDEZVOUS_SEM) != 0)
        {
        }
    }

    ckernel::fence_compiler();
}

__attribute__((always_inline)) inline void rendezvous(bool is_action_thread)
{
    rendezvous(is_action_thread, [] {});
}

// Fixed rather than per-run-type: letting it vary is how the two builds ended up releasing from
// different threads.
constexpr bool is_action_thread()
{
#if defined(LLK_TRISC_PACK)
    return true;
#else
    return false;
#endif
}

#endif // !ARCH_QUASAR

} // namespace llk_barrier
