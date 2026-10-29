// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, Blackhole: handing a tensor from its owner to the op that touches it next. Written only in HAL calls.
// tt-llk-lab llk-tensor, report 2 section 7 and report 4 sections 4 and 5:
// https://github.com/tenstorrent/tt-llk-lab/tree/lpremovic/llk-tensor/experiments/llk-tensor
//
// A tensor's owner is the (thread, unit) in its spec (tensor/tensor.h). An op on (T, E) that touches a tensor with
// spec S calls acquire<T, E, S>() first, on every thread:
//   S nobody's               nothing: no one has written it
//   S's thread is T          the same thread: if S's unit is not E, a STALLWAIT on T until S's unit is idle, holding
//                            E's instructions
//   S's thread is another    S's thread waits until S's unit is idle and posts the pair's semaphore; T waits on it
//                            (every instruction class held) and takes it. Other threads do nothing
// Every TRISC runs the same kernel body, so the owner's thread meets the call too: the producer never needs to know
// its consumer. Read after write is implicit this way. Write after read is explicit: complete<T>(t) (compute_api/
// sync.h) is acquire<T, Exu::None>, so T waits until t is written, which, units working in order, also means its
// writer is done reading its inputs.
//
// The pairs and their semaphores (semaphores.h):
//   math -> pack   math_pack_sync    pack (or an SFPU call on the pack thread) takes what math wrote
//   pack -> math   pack_math_sync    complete<Math>, or math writes a tensor the pack thread's SFPU wrote last
// Others (unpack <-> math, unpack <-> pack) fail at compile time: no op reads a tensor that way yet. Each semaphore
// counts hand-offs in flight; the poster sets it up before its first post (semaphore_setup_once, semaphores.h).
//
// migrate<E, From, To>() hands unit E (the SFPU) from one thread to the other: From waits for E to be idle and posts,
// To waits. Step 3 of report 4's migrate, setting the unit up for its new thread, is not needed here: every vec_ call
// configures the SFPU completely (llk_lib/vec.h), so a kernel whose SFPU calls stay on one thread needs no migrate.

#pragma once

#include <cstdint>

#include "hal/sync.h"
#include "llk_lib/semaphores.h"
#include "llk_lib/thread.h"
#include "tensor/tensor.h"

namespace llk2::llk_lib
{

using hal::sync::Semaphore;
using hal::sync::SemaphoreCondition;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

/// Hand-offs one pair may have in flight: the semaphore's maximum (its 4-bit maximum field).
inline constexpr std::uint32_t handoff_max_pending = 15;

/// The stall condition under which unit e has finished everything its thread gave it.
constexpr StallCondition idle(Exu e)
{
    switch (e)
    {
        case Exu::Unpacker:
            return StallCondition::Unpacker0Idle | StallCondition::Unpacker1Idle;
        case Exu::Fpu:
            return StallCondition::MathIdle;
        case Exu::Sfpu:
            return StallCondition::SfpuIdle;
        case Exu::Packer:
            return StallCondition::PackerIdle;
        default:
            return StallCondition::All;
    }
}

/// The instruction classes unit e runs: what a hand-off to e holds back. Exu::None (complete): every class.
constexpr StallTarget issued_to(Exu e)
{
    switch (e)
    {
        case Exu::Unpacker:
            return StallTarget::Unpack;
        case Exu::Fpu:
            return StallTarget::Math;
        case Exu::Sfpu:
            return StallTarget::Sfpu;
        case Exu::Packer:
            return StallTarget::Pack;
        default:
            return StallTarget::All;
    }
}

/// Whether thread from can hand a tensor to thread to.
constexpr bool has_handoff(Thread from, Thread to)
{
    return (from == Thread::Math && to == Thread::Pack) || (from == Thread::Pack && to == Thread::Math);
}

/// The semaphore from hands tensors to `to` on.
constexpr Semaphore handoff_semaphore(Thread from, Thread to)
{
    return from == Thread::Math && to == Thread::Pack ? math_pack_sync : pack_math_sync;
}

/// The posting thread: wait until unit Owner is idle, then post semaphore Sem (setting it up on the first post).
template <Exu Owner, Semaphore Sem>
inline __attribute__((always_inline)) void handoff_post()
{
    semaphore_setup_once<Sem, handoff_max_pending>();
    hal::sync::wait::stall<StallTarget::Sync, idle(Owner)>();
    hal::sync::semaphore::post<Sem>();
}

/// The taking thread: hold every instruction class until Sem has a hand-off, then take it.
template <Semaphore Sem>
inline __attribute__((always_inline)) void handoff_take()
{
    hal::sync::wait::semaphore<StallTarget::All, SemaphoreCondition::WhileZero, Sem>();
    hal::sync::semaphore::get<Sem>();
}

/// Hands a tensor with spec S to unit E on thread T, on every thread: see the top of this file.
template <Thread T, Exu E, const TensorSpec<tensor_rank>& S>
inline __attribute__((always_inline)) void acquire()
{
    if constexpr (S.thread == Thread::None)
    {
        // Nobody's: nothing to wait for.
    }
    else if constexpr (S.thread == T)
    {
        if constexpr (S.exu != E && runs_on<T>)
        {
            hal::sync::wait::stall<issued_to(E), idle(S.exu)>(); // the same thread, another unit
        }
    }
    else
    {
        static_assert(has_handoff(S.thread, T), "this hand-off has no semaphore yet: only math -> pack and pack -> math");
        constexpr Semaphore sem = handoff_semaphore(S.thread, T);
        if constexpr (runs_on<S.thread>)
        {
            handoff_post<S.exu, sem>(); // the owner's thread hands the tensor over
        }
        if constexpr (runs_on<T>)
        {
            handoff_take<sem>(); // the op's thread takes it
        }
    }
}

/// Hands unit E from thread From to thread To: see the top of this file.
template <Exu E, Thread From, Thread To>
inline __attribute__((always_inline)) void migrate()
{
    static_assert(E == Exu::Sfpu, "only the SFPU can be driven from more than one thread");
    static_assert(has_handoff(From, To), "the SFPU moves between the math and the pack thread");
    constexpr Semaphore sem = handoff_semaphore(From, To);
    if constexpr (runs_on<From>)
    {
        handoff_post<E, sem>();
    }
    if constexpr (runs_on<To>)
    {
        handoff_take<sem>();
    }
}

} // namespace llk2::llk_lib
