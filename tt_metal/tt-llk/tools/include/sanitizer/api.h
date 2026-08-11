// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>

#include "sanitizer/operation.h"
#include "sanitizer/settings.h"
#include "sanitizer/types.h"

#if defined(LLK_SAN_ENABLE)

#include "sanitizer/impl.h"

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
static inline void thread_init()
{
    detail::exu_dispatch([](auto exu) { detail::exu_init<exu.value, T>(*state); });
}

// ------------
// Entry points
// ------------

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

namespace llk::san
{

template <Thread T = Thread::TRISC0>
static inline void thread_init()
{
}

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

#define SAN_HOOK(...)             \
    do                            \
    {                             \
        using namespace llk::san; \
        using llk::san::Operand;  \
        __VA_ARGS__;              \
    } while (false)
