// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// LLK_SAN_ENABLE : Master switch. Enable/Disable sanitizer completely. Default DISABLED

// The report method is implicit and depends on which of the following are enabled:
// ENABLE_LLK_ASSERT   : Once sanitizer is tripped, causes an LLK_ASSERT, stops execution
// DEBUG_PRINT_ENABLED : Once sanitizer is tripped, causes a DEVICE_PRINT, continues execution
// At least one of the two must be enabled whenever the sanitizer is enabled.

// LLK_SAN_SETTING_PEDANTIC : Breaks LLK Contract, won't cause incorrect behavior, or check is redundant. Default FALSE.
// LLK_SAN_SETTING_WARN     : Breaks LLK Contract, might cause incorrect behavior, or performance degradation. Default TRUE.
// LLK_SAN_SETTING_ERROR    : Breaks LLK Contract, will cause incorrect behavior, must be fixed. Default TRUE.

// LLK_SAN_SETTING_INFO     : Used to inform about unavoidable breakages in Contract, or Sanitizer implementation. Default TRUE.
// LLK_SAN_SETTING_FAULT    : Unexpected fault in the Sanitizer implementation. Unrecoverable, report immediately. Default FALSE.

// LLK_SAN_SETTING_INTERNAL : Only useful when debugging LLK LIB implementation, currently unimplemented. Default FALSE.

#pragma once

#ifndef LLK_SAN_ENABLE

#if defined(LLK_SAN_SETTING_PEDANTIC)
#error "llk::san | fault   | LLK_SAN_SETTING_PEDANTIC is set but LLK_SAN_ENABLE is not defined"
#endif
#define LLK_SAN_SETTING_PEDANTIC 0

#if defined(LLK_SAN_SETTING_WARN)
#error "llk::san | fault   | LLK_SAN_SETTING_WARN is set but LLK_SAN_ENABLE is not defined"
#endif
#define LLK_SAN_SETTING_WARN 0

#if defined(LLK_SAN_SETTING_ERROR)
#error "llk::san | fault   | LLK_SAN_SETTING_ERROR is set but LLK_SAN_ENABLE is not defined"
#endif
#define LLK_SAN_SETTING_ERROR 0

#if defined(LLK_SAN_SETTING_INFO)
#error "llk::san | fault   | LLK_SAN_SETTING_INFO is set but LLK_SAN_ENABLE is not defined"
#endif
#define LLK_SAN_SETTING_INFO 0

#if defined(LLK_SAN_SETTING_FAULT)
#error "llk::san | fault   | LLK_SAN_SETTING_FAULT is set but LLK_SAN_ENABLE is not defined"
#endif
#define LLK_SAN_SETTING_FAULT 0

#if defined(LLK_SAN_SETTING_INTERNAL)
#error "llk::san | fault   | LLK_SAN_SETTING_INTERNAL is set but LLK_SAN_ENABLE is not defined"
#endif
#define LLK_SAN_SETTING_INTERNAL 0

#else

#if !defined(LLK_SAN_SETTING_PEDANTIC)
#define LLK_SAN_SETTING_PEDANTIC 0
#endif

#if !defined(LLK_SAN_SETTING_WARN)
#define LLK_SAN_SETTING_WARN 1
#endif

#if !defined(LLK_SAN_SETTING_ERROR)
#define LLK_SAN_SETTING_ERROR 1
#endif

#if !defined(LLK_SAN_SETTING_INFO)
#define LLK_SAN_SETTING_INFO 1
#endif

#if !defined(LLK_SAN_SETTING_FAULT)
#define LLK_SAN_SETTING_FAULT 0
#endif

#if !defined(LLK_SAN_SETTING_INTERNAL)
#define LLK_SAN_SETTING_INTERNAL 0
#endif

#endif

namespace llk::san
{

enum class Trigger
{
    PEDANTIC,
    WARN,
    ERROR,
    FAULT,
    INFO,
    INTERNAL
};

constexpr bool enabled_trigger(Trigger level)
{
    switch (level)
    {
        case Trigger::PEDANTIC:
            return LLK_SAN_SETTING_PEDANTIC;
        case Trigger::WARN:
            return LLK_SAN_SETTING_WARN;
        case Trigger::ERROR:
            return LLK_SAN_SETTING_ERROR;
        case Trigger::FAULT:
            return LLK_SAN_SETTING_FAULT;
        case Trigger::INFO:
            return LLK_SAN_SETTING_INFO;
        case Trigger::INTERNAL:
            return LLK_SAN_SETTING_INTERNAL;
    }
    __builtin_unreachable();
}

}; // namespace llk::san
