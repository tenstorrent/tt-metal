// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel.h"
#if !defined(ARCH_QUASAR)
#include "ckernel_structs.h" // semaphore indices; Quasar names its own set in ckernel_trisc_common.h
#endif

// One cross-thread rendezvous for the whole perf harness.
//
// There used to be three: sync_threads and sync_point in profiler.h, both built on L1 words, and
// sem_rendezvous in counters.h, built on a hardware semaphore. Two problems came from having more than
// one, and both cost real measurement error before they were found.
//
// The barriers are not interchangeable, so which one a zone used changed its answer. sync_point's actor
// releases the other threads by storing an L1 epoch word that they only notice on their next poll, and
// each poll is a fence plus an uncached L1 load, so the actor leaves tens of cycles early. Where the
// measured loop is a strict 1:1 producer/consumer ping-pong that head start is a fixed point, conserved
// for every iteration, and the window comes out about one cycle per tile short. Measured on
// perf_unpack_tilize: 106575 cycles under sync_point against 109139 under the semaphore, -1.002
// cycles/tile, on two independent run types.
//
// sem_rendezvous lived inside PERF_COUNTERS_COMPILED, so the no-counter build could not reach it and
// always fell back to sync_point. The two builds therefore measured the same zone with different
// instruments, which is not counter cost but read like it: L1_TO_L1 INIT was off by 21 cycles in every
// one of 462 variants, and unpack_tilize UNPACK_ISOLATE by 2554.
//
// This header fixes both by construction rather than by vigilance. There is one implementation, it is
// outside every build guard so NC and WC compile the identical barrier, and it uses the semaphore
// because that is the medium without the pathology: the release is symmetric, detection is a 2-4 cycle
// pc_buf poll rather than tens of cycles, and it puts no traffic on L1 at all. That last point matters
// because the L1 protocol kept its epoch word and all three threads' poll targets in one 16-byte line
// on bank 31, so the barrier competed with the very thing being measured.

namespace llk_barrier
{

// Quasar: 4 TRISCs (UNPACK, MATH, PACK, SFPU); Wormhole/Blackhole: 3.
#if defined(ARCH_QUASAR)
constexpr std::uint32_t NUM_THREADS = 4;
#else
constexpr std::uint32_t NUM_THREADS = 3;
#endif

// semaphore::PACK_DONE is the right choice and not an arbitrary one. It has no use anywhere in the LLK
// library (its own header describes it as being for recording perf events and inserting delay), so the
// barrier cannot couple to the op it is measuring, which is the failure mode that produced several of
// the bugs above. boot.h seeds it with t6_semaphore_init(PACK_DONE, 0, 1), and that max of 1 does not
// clamp anything here: the max drives o_stall_cond for SEMWAIT, and this code only ever posts, reads and
// gets, never waits. So no SEMINIT of our own is needed and none is issued.
// Quasar has no semaphore to spare: its set is MATH_PACK, UNPACK_MATH and PACK_UNPACK, all three in
// use by the ops themselves, and there is no PACK_DONE. It also does not compile the counter path at all
// (counters.h static_asserts it out), so the only caller there is sync_threads, and borrowing an op's
// semaphore for a measurement barrier is exactly the coupling this header exists to avoid. Quasar keeps
// the L1 rendezvous in profiler.h until it has a free semaphore or its own mechanism.
#if !defined(ARCH_QUASAR)
constexpr std::uint8_t RENDEZVOUS_SEM = ckernel::semaphore::PACK_DONE;

// Every thread announces by incrementing. The action thread waits for all announcements, runs action(),
// then drains the count back to zero, and that return to zero IS the release, so there is no separate
// release flag and no thread is released before the action has run. Threads that are not the action
// thread only ever observe, so none of them can be handed a head start.
//
// The action thread must be exactly one thread and every thread must arrive, or the count never reaches
// NUM_THREADS and never returns to zero. Callers get that for free when the barrier sits at a zone
// boundary reached unconditionally by all threads, which is what START_PERF_MEASURE guarantees.
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

// Plain barrier with nothing to run at the rendezvous point. Same code path, so a zone that needs no
// action is not measured through a different instrument than one that does.
__attribute__((always_inline)) inline void rendezvous(bool is_action_thread)
{
    rendezvous(is_action_thread, [] {});
}

// The thread that runs the action. Fixed rather than per-run-type, because letting it vary is how the
// two builds ended up releasing from different threads.
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
