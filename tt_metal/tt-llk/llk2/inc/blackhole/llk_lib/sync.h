// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// llk-lib, Blackhole: handing a tensor from its owner to the op that touches it next. Written only in HAL calls.
// tt-llk-lab llk-tensor, report 2 section 7 and report 4 sections 4 and 5:
// https://github.com/tenstorrent/tt-llk-lab/tree/lpremovic/llk-tensor/experiments/llk-tensor
//
// A tensor's owner is the (thread, unit) the Tensor carries at run time (tensor/tensor.h). An op on (T, E) that
// touches a tensor t calls acquire<T, E>(t) first, on every thread:
//   t nobody's               nothing: no one has written it
//   t's thread is T          the same thread: if t's unit is not E, a STALLWAIT on T until t's unit is idle, holding
//                            E's instructions
//   t's thread is another    t's thread waits until t's unit is idle and posts the pair's semaphore; T waits on it
//                            (every instruction class held) and takes it. Other threads do nothing
// Every thread computes the same owners (they all run the same kernel body), and the owners of a kernel's tensors are
// constants after inlining, so the compiler usually folds these branches away; a hand-off with no semaphore yet is an
// LLK_ASSERT.
// Every TRISC runs the same kernel body, so the owner's thread meets the call too: the producer never needs to know
// its consumer. Read after write is implicit this way. Write after read is explicit: complete<T>(t) (compute_api/
// sync.h) is acquire<T, OwnerExu::None>, so T waits until t is written, which, units working in order, also means its
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

namespace llk::llk_lib
{

using hal::sync::Semaphore;
using hal::sync::SemaphoreCondition;
using hal::sync::StallCondition;
using hal::sync::StallTarget;

/// Hand-offs one pair may have in flight: the semaphore's maximum (its 4-bit maximum field).
inline constexpr std::uint32_t handoff_max_pending = 15;

/// The stall condition under which unit e has finished everything its thread gave it.
constexpr StallCondition idle(OwnerExu e)
{
    switch (e)
    {
        case OwnerExu::Unpack:
            return StallCondition::Unpacker0Idle | StallCondition::Unpacker1Idle;
        case OwnerExu::Fpu:
            return StallCondition::MathIdle;
        case OwnerExu::Sfpu:
            return StallCondition::SfpuIdle;
        case OwnerExu::Pack:
            return StallCondition::PackerIdle;
        default:
            return StallCondition::All;
    }
}

/// The instruction classes unit e runs: what a hand-off to e holds back. OwnerExu::None (complete): every class.
constexpr StallTarget issued_to(OwnerExu e)
{
    switch (e)
    {
        case OwnerExu::Unpack:
            return StallTarget::Unpack;
        case OwnerExu::Fpu:
            return StallTarget::Math;
        case OwnerExu::Sfpu:
            return StallTarget::Sfpu;
        case OwnerExu::Pack:
            return StallTarget::Pack;
        default:
            return StallTarget::All;
    }
}

/// Whether thread from can hand a tensor to thread to.
constexpr bool has_handoff(OwnerThread from, OwnerThread to)
{
    return (from == OwnerThread::Compute1 && to == OwnerThread::Compute2) || (from == OwnerThread::Compute2 && to == OwnerThread::Compute1);
}

/// The semaphore from hands tensors to `to` on.
constexpr Semaphore handoff_semaphore(OwnerThread from, OwnerThread to)
{
    return from == OwnerThread::Compute1 && to == OwnerThread::Compute2 ? math_pack_sync : pack_math_sync;
}

/// Holds Target's instructions until unit `unit` is idle: a STALLWAIT, with the unit known only at run time.
template <StallTarget Target>
inline __attribute__((always_inline)) void wait_idle(OwnerExu unit)
{
    switch (unit)
    {
        case OwnerExu::Unpack:
            hal::sync::wait::stall<Target, idle(OwnerExu::Unpack)>();
            break;
        case OwnerExu::Fpu:
            hal::sync::wait::stall<Target, idle(OwnerExu::Fpu)>();
            break;
        case OwnerExu::Sfpu:
            hal::sync::wait::stall<Target, idle(OwnerExu::Sfpu)>();
            break;
        case OwnerExu::Pack:
            hal::sync::wait::stall<Target, idle(OwnerExu::Pack)>();
            break;
        default:
            hal::sync::wait::stall<Target, idle(OwnerExu::None)>();
            break;
    }
}

/// The posting thread: wait until unit `owner` is idle, then post semaphore Sem (setting it up on the first post).
template <Semaphore Sem>
inline __attribute__((always_inline)) void handoff_post(OwnerExu owner)
{
    semaphore_setup_once<Sem, handoff_max_pending>();
    wait_idle<StallTarget::Sync>(owner);
    hal::sync::semaphore::post<Sem>();
}

/// The taking thread: hold every instruction class until Sem has a hand-off, then take it.
template <Semaphore Sem>
inline __attribute__((always_inline)) void handoff_take()
{
    hal::sync::wait::semaphore<StallTarget::All, SemaphoreCondition::WhileZero, Sem>();
    hal::sync::semaphore::get<Sem>();
}

/// A tensor owned by unit `unit` on thread From goes to thread To: From posts, To takes. Nothing for a pair with no
/// semaphore (acquire asserts on that).
template <OwnerThread From, OwnerThread To>
inline __attribute__((always_inline)) void handoff(OwnerExu unit)
{
    if constexpr (has_handoff(From, To))
    {
        constexpr Semaphore sem = handoff_semaphore(From, To);
        if constexpr (runs_on<From>)
        {
            handoff_post<sem>(unit); // the owner's thread hands the tensor over
        }
        if constexpr (runs_on<To>)
        {
            handoff_take<sem>(); // the op's thread takes it
        }
    }
}

/// Hands a tensor owned by (thread, unit) to unit E on thread T, on every thread: see the top of this file.
template <OwnerThread T, OwnerExu E>
inline __attribute__((always_inline)) void acquire(OwnerThread thread, OwnerExu unit)
{
    if (thread == OwnerThread::None)
    {
        return; // nobody's: nothing to wait for
    }
    if (thread == T)
    {
        if constexpr (runs_on<T>)
        {
            if (unit != E)
            {
                wait_idle<issued_to(E)>(unit); // the same thread, another unit
            }
        }
        return;
    }
    LLK_ASSERT(has_handoff(thread, T), "this hand-off has no semaphore yet: only math -> pack and pack -> math");
    switch (thread)
    {
        case OwnerThread::Compute0:
            handoff<OwnerThread::Compute0, T>(unit);
            break;
        case OwnerThread::Compute1:
            handoff<OwnerThread::Compute1, T>(unit);
            break;
        case OwnerThread::Compute2:
            handoff<OwnerThread::Compute2, T>(unit);
            break;
        default:
            break;
    }
}

/// Hands tensor t to unit E on thread T: acquire with t's owner.
template <OwnerThread T, OwnerExu E, class S, class R>
inline __attribute__((always_inline)) void acquire(const Tensor<S, R>& t)
{
    acquire<T, E>(t.thread(), t.exu());
}

/// Hands unit E from thread From to thread To: see the top of this file.
template <OwnerExu E, OwnerThread From, OwnerThread To>
inline __attribute__((always_inline)) void migrate()
{
    static_assert(E == OwnerExu::Sfpu, "only the SFPU can be driven from more than one thread");
    static_assert(has_handoff(From, To), "the SFPU moves between the math and the pack thread");
    constexpr Semaphore sem = handoff_semaphore(From, To);
    if constexpr (runs_on<From>)
    {
        handoff_post<sem>(E);
    }
    if constexpr (runs_on<To>)
    {
        handoff_take<sem>();
    }
}

} // namespace llk::llk_lib
