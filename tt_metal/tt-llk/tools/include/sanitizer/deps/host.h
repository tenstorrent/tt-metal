// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The report vocabulary for host unit tests: libfmt stands in for the device print buffer and
// SAN_ASSERT reduces to fputs + abort. The carriers come from deps/common.h. Selected by
// LLK_SAN_SETTING_HOST_DEPS.

#pragma once

#include <cstdint>

#include "sanitizer/settings.h"

#define SAN_DEPS_CARRIERS_FROM_METAL 0

#if defined(DEBUG_PRINT_ENABLED)
#define SAN_DEPS_PRINT_LIVE 1
#else
#define SAN_DEPS_PRINT_LIVE 0
#endif

#include "sanitizer/deps/common.h"

#if SAN_DEPS_PRINT_LIVE

#define FMT_HEADER_ONLY
#include <fmt/format.h>

#if defined(__GNUC__) && !defined(__GXX_RTTI)
#error "llk::san | fault   | the host mock spells type names with typeid; compile without -fno-rtti"
#endif

#include <cxxabi.h>

#include <cstdlib>
#include <memory>
#include <typeinfo>

template <>
struct fmt::formatter<llk::san::string> : fmt::formatter<const char*>
{
    auto format(const llk::san::string s, format_context& ctx) const
    {
        return fmt::formatter<const char*>::format(s.ptr, ctx);
    }
};

template <>
struct fmt::formatter<llk::san::callstack>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const llk::san::callstack&, format_context& ctx) const
    {
        return fmt::format_to(ctx.out(), "│  └ ...");
    }
};

// The device serializer parks T's name in .device_print_strings; this demangles typeid(T) instead,
// so a few spellings differ from a device report -- std::string and lambdas in particular.
template <typename T>
struct fmt::formatter<llk::san::type_name<T>>
{
    constexpr auto parse(format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const llk::san::type_name<T>&, format_context& ctx) const
    {
        // __cxa_demangle returns null when it cannot, leaving the mangled name as the only thing to print.
        const std::unique_ptr<char, void (*)(void*)> name {abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr), std::free};

        return fmt::format_to(ctx.out(), "{}", name ? name.get() : typeid(T).name());
    }
};

#define SAN_PRINT(...) fmt::print(__VA_ARGS__)

#endif // SAN_DEPS_PRINT_LIVE

// Stands in for common/llk_assert.h, whose arms are an ebreak and metal's watcher-backed ASSERT.
// Unlike the ebreak arm this one prints message, so it requires a string literal.
#if defined(ENABLE_LLK_ASSERT)

#include <cstdio>
#include <cstdlib>

#define SAN_ASSERT(condition, message)        \
    do                                        \
    {                                         \
        if (!(condition))                     \
        {                                     \
            std::fputs(message "\n", stderr); \
            std::abort();                     \
        }                                     \
    } while (false)

#else

#define SAN_ASSERT(condition, message) SAN_DISCARD((condition))

#endif // ENABLE_LLK_ASSERT
