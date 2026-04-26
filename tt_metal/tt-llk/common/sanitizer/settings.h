// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// LLK_SAN_ENABLE : Master switch. Enable/Disable sanitizer completely. Default DISABLED

// If none of the below two are set, defaults to ASSERT. Mutually exclusive.
// LLK_SAN_SETTING_ASSERT   : Once sanitizer is tripped, causes an LLK_ASSERT, stops execution
// LLK_SAN_SETTING_PRINT    : Once sanitizer is tripped, causes a DEVICE_PRINT, continues execution

// LLK_SAN_SETTING_PEDANTIC : Breaks LLK Contract, won't cause incorrect behavior, or check is redundant. Default FALSE.
// LLK_SAN_SETTING_WARN     : Breaks LLK Contract, might cause incorrect behavior, or performance degradation. Default TRUE.
// LLK_SAN_SETTING_ERROR    : Breaks LLK Contract, will cause incorrect behavior, must be fixed. Default TRUE.

// LLK_SAN_SETTING_INFO     : Used to inform about unavoidable breakages in Contract, or Sanitizer implementation. Default TRUE.
// LLK_SAN_SETTING_FAULT    : Unexpected fault in the Sanitizer implementation. Unrecoverable, report immediately. Default FALSE.

// LLK_SAN_SETTING_INTERNAL : Only useful when debugging LLK LIB implementation, currently unimplemented. Default FALSE.

#pragma once

#ifndef LLK_SAN_ENABLE

#if defined(LLK_SAN_SETTING_ASSERT)
#error "LLK_SAN_SETTING_ASSERT is set but LLK_SAN_ENABLE is not defined"
#endif
#define LLK_SAN_SETTING_ASSERT 0

#if defined(LLK_SAN_SETTING_PRINT)
#error "LLK_SAN_SETTING_PRINT is set but LLK_SAN_ENABLE is not defined"
#endif
#define LLK_SAN_SETTING_PRINT 0

#if defined(LLK_SAN_SETTING_PEDANTIC)
#error "LLK_SAN_SETTING_PEDANTIC is set but LLK_SAN_ENABLE is not defined"
#endif
#define LLK_SAN_SETTING_PEDANTIC 0

#if defined(LLK_SAN_SETTING_WARN)
#error "LLK_SAN_SETTING_WARN is set but LLK_SAN_ENABLE is not defined"
#endif
#define LLK_SAN_SETTING_WARN 0

#if defined(LLK_SAN_SETTING_ERROR)
#error "LLK_SAN_SETTING_ERROR is set but LLK_SAN_ENABLE is not defined"
#endif
#define LLK_SAN_SETTING_ERROR 0

#if defined(LLK_SAN_SETTING_INFO)
#error "LLK_SAN_SETTING_INFO is set but LLK_SAN_ENABLE is not defined"
#endif
#define LLK_SAN_SETTING_INFO 0

#if defined(LLK_SAN_SETTING_FAULT)
#error "LLK_SAN_SETTING_FAULT is set but LLK_SAN_ENABLE is not defined"
#endif
#define LLK_SAN_SETTING_FAULT 0

#else

#if defined(LLK_SAN_SETTING_ASSERT) && defined(LLK_SAN_SETTING_PRINT)
#error "LLK_SAN_SETTING_ASSERT and LLK_SAN_SETTING_PRINT cannot both be defined"
#endif

#if !defined(LLK_SAN_SETTING_ASSERT) && !defined(LLK_SAN_SETTING_PRINT)
// If override is not provided, default to assert
#define LLK_SAN_SETTING_ASSERT 1
#endif

#if !defined(LLK_SAN_SETTING_PEDANTIC)
// If override is not provided, default to pedantic
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
