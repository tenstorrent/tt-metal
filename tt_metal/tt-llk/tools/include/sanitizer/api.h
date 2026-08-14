// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <utility>

#include "sanitizer/settings.h"
#include "sanitizer/types.h"

#if defined(LLK_SAN_ENABLE)

#include "sanitizer/impl.h"

namespace llk::san
{

// per thread state
extern State* const state;

#if 0
// DISABLED. The pre-refactor surface: the per-Exu configure/check hooks, the operation_init/check/
// uninit trio keyed on the old Operation enum, the thread_* context helpers, and the zone RAII types.
// All of it is written against State<T>, SanitizerState and the Operation enum, none of which exist
// any more, and all of it routes into the legacy region of impl.h that is disabled for the same
// reason.
//
// Disabled rather than deleted because the behaviour still has to be carried over -- the context
// tracking and the reporting in particular, which the guarded entry points above do not yet do.
//
// sstanisic todo: port onto the new state model and re-enable, or delete once output.h is rewritten.

// sstanisic todo: SanitizerState no longer exists -- this and the legacy hooks below it go as
// the per-exu configure/check functions are replaced by the guarded entry points.
extern SanitizerState* const sanitizer;

static inline void thread_init()
{
    thread_init_impl(*sanitizer);
}

static inline auto& thread_context_get()
{
    return thread_context_get_impl(*sanitizer);
}

static inline void thread_silent_push()
{
    thread_silent_push_impl(thread_context_get());
}

static inline void thread_silent_pop()
{
    thread_silent_pop_impl(thread_context_get());
}

static inline void thread_context_push()
{
    thread_context_push_impl(thread_context_get());
}

static inline void thread_context_pop()
{
    thread_context_pop_impl(thread_context_get());
}

#endif // legacy surface

namespace detail
{

constexpr Thread get_thread()
{
    if constexpr (COMPILE_FOR_TRISC == 0)
    {
        return Thread::TRISC0;
    }
    else if constexpr (COMPILE_FOR_TRISC == 1)
    {
        return Thread::TRISC1;
    }
    else if constexpr (COMPILE_FOR_TRISC == 2)
    {
        return Thread::TRISC2;
    }
    else if constexpr (COMPILE_FOR_TRISC == 3)
    {
        return Thread::TRISC3;
    }
}

} // namespace detail

// ---------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------
//
// Thin forwarders. Every rule about what may be passed, and every diagnostic for breaking one, lives
// in impl.h -- because that layer is arch independent and this one is not. What a thread may be
// handed is a property of the sanitizer's own model, so it is stated once and is identical for every
// target; only the two genuinely target-specific things are supplied here.
//
// Those two are T, defaulting to the thread this translation unit is compiled for, and the per-thread
// state object. T is a parameter rather than being read from COMPILE_FOR_TRISC inside impl.h for
// exactly one reason: the diagnostics tests are host compiled, where there is no such macro, and they
// have to be able to name each thread to check the rules for it.
//
// init(), execute() and uninit() name their operation as the leading template argument, so kernels
// write init<OperationUnpackTilize>(...). It is not inferred from the arguments: naming it is what
// lets a nullary uninit<Op>() still be checked, and what removes the need to prove an inferred
// operation unambiguous.

template <Thread T = detail::get_thread(), typename... Vs>
static inline void configure(Vs&&... values)
{
    detail::configure<T>(*state, std::forward<Vs>(values)...);
}

template <Thread T = detail::get_thread(), typename... Vs>
static inline void reconfigure(Vs&&... values)
{
    detail::reconfigure<T>(*state, std::forward<Vs>(values)...);
}

template <typename Op, Thread T = detail::get_thread(), typename... Vs>
static inline void init(Vs&&... values)
{
    detail::init<Op, T>(*state, std::forward<Vs>(values)...);
}

template <typename Op, Thread T = detail::get_thread(), typename... Vs>
static inline void execute(Vs&&... values)
{
    detail::execute<Op, T>(*state, std::forward<Vs>(values)...);
}

template <typename Op, Thread T = detail::get_thread(), typename... Vs>
static inline void uninit(Vs&&... values)
{
    detail::uninit<Op, T>(*state, std::forward<Vs>(values)...);
}

#if 0
// Goes in LLK_LIB in Init
// Store operation type and save arguments
template <Operation op, typename... Ts>
static inline void operation_init(Ts&&... args)
{
    const bool fsm_success = fsm_init_impl(thread_context_get(), sanitizer->fsm[COMPILE_FOR_TRISC], op);

    if (!fsm_success)
    {
        thread_silent_push();
    }

    operation_init_impl(thread_context_get(), sanitizer->operation[COMPILE_FOR_TRISC], op, std::forward<Ts>(args)...);

    if (!fsm_success)
    {
        thread_silent_pop();
    }
}

// Goes in LLK_LIB in Execute
// Check operation type and arguments against stored ones
template <Operation op, typename... Ts>
static inline void operation_check(Ts&&... args)
{
    const bool fsm_success = fsm_execute_impl(thread_context_get(), sanitizer->fsm[COMPILE_FOR_TRISC], op);

    if (!fsm_success)
    {
        thread_silent_push();
    }

    operation_execute_impl(thread_context_get(), sanitizer->operation[COMPILE_FOR_TRISC], op, std::forward<Ts>(args)...);

    if (!fsm_success)
    {
        thread_silent_pop();
    }
}

// Goes in LLK_LIB in Uninit
// Check operation type and clear must uninit flag
template <Operation op>
void operation_uninit()
{
    const bool fsm_success = fsm_uninit_impl(thread_context_get(), sanitizer->fsm[COMPILE_FOR_TRISC], op);

    if (!fsm_success)
    {
        thread_silent_push();
    }

    operation_uninit_impl(thread_context_get(), sanitizer->operation[COMPILE_FOR_TRISC], op);

    if (!fsm_success)
    {
        thread_silent_pop();
    }
}

class FunctionZone
{
public:
    FunctionZone()
    {
        thread_context_push();
    }

    ~FunctionZone()
    {
        thread_context_pop();
    }
};

class SilentZone
{
public:
    SilentZone()
    {
        thread_silent_push();
    }

    ~SilentZone()
    {
        thread_silent_pop();
    }
};

#endif // legacy surface

} // namespace llk::san

#define LLK_SAN_FUNCTION() llk::san::FunctionZone _function_zone_

#define LLK_SAN_SILENT_ZONE() [[maybe_unused]] llk::san::SilentZone _silent_zone_

#else

namespace llk::san
{

// Sanitizer disabled. The five entry points still have to exist and still have to accept exactly what
// the enabled build accepts, so that turning the sanitizer off can never change whether a kernel
// compiles -- only what it does. They take their arguments by forwarding reference and discard them.
//
// No guard runs here, deliberately. The rules are compile-time-only, and duplicating them in the
// disabled path would be a second copy free to disagree with the first; a kernel is expected to be
// built with the sanitizer on at least once, which is where the rules are enforced.

template <Thread T = Thread::TRISC0, typename... Vs>
static inline void configure([[maybe_unused]] Vs&&... values)
{
}

template <Thread T = Thread::TRISC0, typename... Vs>
static inline void reconfigure([[maybe_unused]] Vs&&... values)
{
}

template <typename Op, Thread T = Thread::TRISC0, typename... Vs>
static inline void init([[maybe_unused]] Vs&&... values)
{
}

template <typename Op, Thread T = Thread::TRISC0, typename... Vs>
static inline void execute([[maybe_unused]] Vs&&... values)
{
}

template <typename Op, Thread T = Thread::TRISC0, typename... Vs>
static inline void uninit([[maybe_unused]] Vs&&... values)
{
}

#if 0
// DISABLED, mirroring the enabled build: the legacy no-op surface, written against State<T> and the
// old Operation enum. Re-enable alongside the legacy hooks in the LLK_SAN_ENABLE branch.
static inline void thread_init()
{
}


template <bool reconfig = false>
static inline void unpack_operand_configure(
    [[maybe_unused]] State<bool> dst_acc_en,
    [[maybe_unused]] State<std::uint32_t> src_fmt_A,
    [[maybe_unused]] State<std::uint32_t> src_fmt_B,
    [[maybe_unused]] State<std::uint32_t> dst_fmt_A,
    [[maybe_unused]] State<std::uint32_t> dst_fmt_B,
    [[maybe_unused]] State<std::uint32_t> face_height_A,
    [[maybe_unused]] State<std::uint32_t> face_height_B,
    [[maybe_unused]] State<std::uint32_t> num_faces_A,
    [[maybe_unused]] State<std::uint32_t> num_faces_B)
{
}

template <bool reconfig = false>
static inline void math_operand_configure([[maybe_unused]] State<std::uint32_t> math_fmt_A, [[maybe_unused]] State<std::uint32_t> math_fmt_B)
{
}

template <bool reconfig = false>
static inline void pack_operand_configure(
    [[maybe_unused]] State<bool> dest_acc_en,
    [[maybe_unused]] State<std::uint32_t> src_fmt,
    [[maybe_unused]] State<std::uint32_t> dst_fmt,
    [[maybe_unused]] State<std::uint32_t> face_height,
    [[maybe_unused]] State<std::uint32_t> tile_width,
    [[maybe_unused]] State<std::uint32_t> num_faces,
    [[maybe_unused]] State<bool> partial_face,
    [[maybe_unused]] State<bool> narrow_tile)
{
}

static inline void unpack_operand_check(
    [[maybe_unused]] State<bool> dst_acc_en,
    [[maybe_unused]] State<std::uint32_t> src_fmt_A,
    [[maybe_unused]] State<std::uint32_t> src_fmt_B,
    [[maybe_unused]] State<std::uint32_t> dst_fmt_A,
    [[maybe_unused]] State<std::uint32_t> dst_fmt_B,
    [[maybe_unused]] State<std::uint32_t> face_height_A,
    [[maybe_unused]] State<std::uint32_t> face_height_B,
    [[maybe_unused]] State<std::uint32_t> num_faces_A,
    [[maybe_unused]] State<std::uint32_t> num_faces_B)
{
}

static inline void math_operand_check([[maybe_unused]] State<std::uint32_t> math_fmt_A, [[maybe_unused]] State<std::uint32_t> math_fmt_B)
{
}

static inline void pack_operand_check(
    [[maybe_unused]] State<bool> dest_acc_en,
    [[maybe_unused]] State<std::uint32_t> src_fmt,
    [[maybe_unused]] State<std::uint32_t> dst_fmt,
    [[maybe_unused]] State<std::uint32_t> face_height,
    [[maybe_unused]] State<std::uint32_t> tile_width,
    [[maybe_unused]] State<std::uint32_t> num_faces,
    [[maybe_unused]] State<bool> partial_face,
    [[maybe_unused]] State<bool> narrow_tile)
{
}

template <Operation op, typename... Ts>
static inline void operation_init([[maybe_unused]] Ts&&... args)
{
}

template <Operation op, typename... Ts>
static inline void operation_check([[maybe_unused]] Ts&&... args)
{
}

template <Operation op>
void operation_uninit()
{
}

#endif // legacy surface

} // namespace llk::san

#define LLK_SAN_FUNCTION() \
    do                     \
    {                      \
    } while (false)

#define LLK_SAN_SILENT_ZONE() \
    do                        \
    {                         \
    } while (false)

#endif
