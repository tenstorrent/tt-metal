// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sanitizer/settings.h"
#include "sanitizer/types.h"

#ifndef LLK_SAN_ENABLE

#define LLK_SAN_ASSERT(condition, ...)

#elif defined(LLK_SAN_SETTING_ASSERT)

#include "llk_assert.h"

#define LLK_SAN_ASSERT(condition, fmt, ...) LLK_ASSERT((condition), "llk::san | " fmt)

#elif defined(LLK_SAN_SETTING_PRINT)

#ifdef ENV_LLK_INFRA
#error "llk_san: error: LLK_SAN_SETTING_PRINT is not supported in LLK INFRA, only in metal"
#endif

#include "api/debug/device_print.h"

#define LLK_SAN_ASSERT(condition, fmt, ...)                      \
    do                                                           \
    {                                                            \
        if (!(condition))                                        \
        {                                                        \
            DEVICE_PRINT("llk::san | " fmt "\n", ##__VA_ARGS__); \
        }                                                        \
    } while (0)

#else
// todo(sstanisic): better message
#error "llk_san: fault: What a Terrible Failure"
#endif

#define LLK_SAN_PANIC(condition, fmt, ...) LLK_SAN_ASSERT(!(condition), fmt, ##__VA_ARGS__)

#if LLK_SAN_SETTING_PEDANTIC == 0
#define LLK_SAN_PEDANTIC_ASSERT(condition, ...)
#elif LLK_SAN_SETTING_PEDANTIC == 1
#define LLK_SAN_PEDANTIC_ASSERT(condition, fmt, ...) LLK_SAN_ASSERT((condition), "pedantic | " fmt, ##__VA_ARGS__)
#else
#error "llk_san: fault: invalid value for LLK_SAN_SETTING_PEDANTIC"
#endif

#define LLK_SAN_PEDANTIC_PANIC(condition, fmt, ...) LLK_SAN_PEDANTIC_ASSERT(!(condition), fmt, ##__VA_ARGS__)
#define LLK_SAN_PEDANTIC_MSG(fmt, ...)              LLK_SAN_PEDANTIC_ASSERT(false, fmt, ##__VA_ARGS__)

#if LLK_SAN_SETTING_WARN == 0
#define LLK_SAN_WARN_ASSERT(condition, ...)
#elif LLK_SAN_SETTING_WARN == 1
#define LLK_SAN_WARN_ASSERT(condition, fmt, ...) LLK_SAN_ASSERT((condition), "warn     | " fmt, ##__VA_ARGS__)
#else
#error "llk_san: fault: invalid value for LLK_SAN_SETTING_WARN"
#endif

#define LLK_SAN_WARN_PANIC(condition, fmt, ...) LLK_SAN_WARN_ASSERT(!(condition), fmt, ##__VA_ARGS__)
#define LLK_SAN_WARN_MSG(fmt, ...)              LLK_SAN_WARN_ASSERT(false, fmt, ##__VA_ARGS__)

#if LLK_SAN_SETTING_ERROR == 0
#define LLK_SAN_ERROR_ASSERT(condition, ...)
#elif LLK_SAN_SETTING_ERROR == 1
#define LLK_SAN_ERROR_ASSERT(condition, fmt, ...) LLK_SAN_ASSERT((condition), "error    | " fmt, ##__VA_ARGS__)
#else
#error "llk_san: fault: invalid value for LLK_SAN_SETTING_ERROR"
#endif

#define LLK_SAN_ERROR_PANIC(condition, fmt, ...) LLK_SAN_ERROR_ASSERT(!(condition), fmt, ##__VA_ARGS__)
#define LLK_SAN_ERROR_MSG(fmt, ...)              LLK_SAN_ERROR_ASSERT(false, fmt, ##__VA_ARGS__)

#if LLK_SAN_SETTING_FAULT == 0
#define LLK_SAN_FAULT_ASSERT(condition, ...)
#elif LLK_SAN_SETTING_FAULT == 1
#define LLK_SAN_FAULT_ASSERT(condition, fmt, ...) LLK_SAN_ASSERT((condition), "fault    | " fmt, ##__VA_ARGS__)
#else
#error "llk_san: fault: invalid value for LLK_SAN_SETTING_FAULT"
#endif

#define LLK_SAN_FAULT_PANIC(condition, fmt, ...) LLK_SAN_FAULT_ASSERT(!(condition), fmt, ##__VA_ARGS__)
#define LLK_SAN_FAULT_MSG(fmt, ...)              LLK_SAN_FAULT_ASSERT(false, fmt, ##__VA_ARGS__)

#if LLK_SAN_SETTING_INFO == 0
#define LLK_SAN_INFO_ASSERT(condition, ...)
#elif LLK_SAN_SETTING_INFO == 1
#define LLK_SAN_INFO_ASSERT(condition, fmt, ...) LLK_SAN_ASSERT((condition), "info     | " fmt, ##__VA_ARGS__)
#else
#error "llk_san: fault: invalid value for LLK_SAN_SETTING_INFO"
#endif

#define LLK_SAN_INFO_PANIC(condition, fmt, ...) LLK_SAN_INFO_ASSERT(!(condition), fmt, ##__VA_ARGS__)
#define LLK_SAN_INFO_MSG(fmt, ...)              LLK_SAN_INFO_ASSERT(false, fmt, ##__VA_ARGS__)

#if LLK_SAN_SETTING_INTERNAL == 0
#define LLK_SAN_INTERNAL_ASSERT(condition, ...)
#elif LLK_SAN_SETTING_INTERNAL == 1
#define LLK_SAN_INTERNAL_ASSERT(condition, fmt, ...) LLK_SAN_ASSERT((condition), "internal | " fmt, ##__VA_ARGS__)
#else
#error "llk_san: fault: invalid value for LLK_SAN_SETTING_INTERNAL"
#endif

#define LLK_SAN_INTERNAL_PANIC(condition, fmt, ...) LLK_SAN_INTERNAL_ASSERT(!(condition), fmt, ##__VA_ARGS__)
#define LLK_SAN_INTERNAL_MSG(fmt, ...)              LLK_SAN_INTERNAL_ASSERT(false, fmt, ##__VA_ARGS__)
