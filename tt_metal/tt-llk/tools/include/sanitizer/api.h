// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sanitizer/settings.h"

#if defined(LLK_SAN_ENABLE)

#include <utility>

#include "sanitizer/impl.h"
#include "sanitizer/operation.h"
#include "sanitizer/types.h"

namespace llk::san
{

// per thread state
extern State* const state;

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
    else
    {
        static_assert(COMPILE_FOR_TRISC >= 0 && COMPILE_FOR_TRISC <= 3, "Invalid COMPILE_FOR_TRISC value");
    }
}

} // namespace detail

template <Thread T = detail::get_thread()>
SAN_FUNC static inline void thread_init()
{
    detail::exu_dispatch([](auto exu) { detail::exu_init<exu.value, T>(*state); });
}

// ------------
// Entry points
// ------------

template <Thread T = detail::get_thread(), typename... Vs>
SAN_FUNC static inline void configure(Vs&&... values)
{
    detail::configure<T>(*state, std::forward<Vs>(values)...);
}

template <Thread T = detail::get_thread(), typename... Vs>
SAN_FUNC static inline void reconfigure(Vs&&... values)
{
    detail::reconfigure<T>(*state, std::forward<Vs>(values)...);
}

template <typename Op, Thread T = detail::get_thread(), typename... Vs>
SAN_FUNC static inline void init(Vs&&... values)
{
    detail::init<Op, T>(*state, std::forward<Vs>(values)...);
}

template <typename Op, Thread T = detail::get_thread(), typename... Vs>
SAN_FUNC static inline void execute(Vs&&... values)
{
    detail::execute<Op, T>(*state, std::forward<Vs>(values)...);
}

template <typename Op, Thread T = detail::get_thread(), typename... Vs>
SAN_FUNC static inline void uninit(Vs&&... values)
{
    detail::uninit<Op, T>(*state, std::forward<Vs>(values)...);
}

// -----------
// Unsupported
// -----------

namespace detail
{

[[gnu::error("llk::san | fault   | this LLK operation is not modelled by the sanitizer")]] void unsupported_operation();

} // namespace detail

SAN_FUNC static inline void unsupported()
{
    detail::unsupported_operation();
}

// ------------
// FunctionZone
// ------------

template <Thread T = detail::get_thread()>
class FunctionZone
{
public:
    FunctionZone()
    {
        const UnwindContext current = detail::unwind_context_read();

        detail::exu_dispatch([&current](auto exu) { detail::exu_context_push<exu.value, T>(*state, current); });
    }

    ~FunctionZone()
    {
        detail::exu_dispatch([](auto exu) { detail::exu_context_pop<exu.value, T>(*state); });
    }
};

// ----------
// SilentZone
// ----------

template <Thread T = detail::get_thread()>
class SilentZone
{
public:
    SilentZone()
    {
        detail::exu_dispatch([](auto exu) { detail::exu_silent_push<exu.value, T>(*state); });
    }

    ~SilentZone()
    {
        detail::exu_dispatch([](auto exu) { detail::exu_silent_pop<exu.value, T>(*state); });
    }
};

} // namespace llk::san

/**
 * Attaches an UnwindContext using RAII.
 */
#define LLK_SAN_FUNCTION() llk::san::FunctionZone<> _function_zone_

/**
 * Silences Sanitizer in the RAII scope.
 */
#define LLK_SAN_SILENT_ZONE() [[maybe_unused]] llk::san::SilentZone<> _silent_zone_

#else

#define LLK_SAN_FUNCTION()    ((void)0)
#define LLK_SAN_SILENT_ZONE() ((void)0)

#endif

#if defined(LLK_SAN_ENABLE)

#define SAN_HOOK(...)             \
    do                            \
    {                             \
        using namespace llk::san; \
        using llk::san::Operand;  \
        __VA_ARGS__;              \
    } while (false)

#else

#define SAN_HOOK(...) ((void)0)

#endif
