// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The report vocabulary as metal supplies it: ckernel.h and llk_assert.h always, and
// device_print.h only where its carriers can actually be printed. An assert-only build, and the LLK
// infra build, fall back to deps/common.h.

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "llk_assert.h"
#include "sanitizer/settings.h"

#if defined(DEBUG_PRINT_ENABLED) && defined(ENV_LLK_INFRA)
#error "llk::san | fault   | DEBUG_PRINT_ENABLED is not supported in LLK INFRA, only in metal"
#endif

// Only the print path can use metal's carriers: with no print backend SAN_PRINT reduces to a sizeof,
// and metal's CTSTR is a lambda, which C++17 bars from an unevaluated operand.
#if defined(DEBUG_PRINT_ENABLED)
#define SAN_DEPS_PRINT_LIVE          1
#define SAN_DEPS_CARRIERS_FROM_METAL 1
#include "api/debug/device_print.h"
#else
#define SAN_DEPS_PRINT_LIVE          0
#define SAN_DEPS_CARRIERS_FROM_METAL 0
#endif

#include "sanitizer/deps/common.h"

#if SAN_DEPS_CARRIERS_FROM_METAL

namespace llk::san
{

using string    = ::ct_string;
using callstack = ::dp_top_callstack_t;

template <typename T>
using type_name = ::dp_type_name_t<T>;

} // namespace llk::san

// CTSTR parks the literal in .device_print_strings, so it has to expand at the report site.
#define SAN_STRING(literal) CTSTR(literal)

#endif // SAN_DEPS_CARRIERS_FROM_METAL

#if SAN_DEPS_PRINT_LIVE
// metal's DEVICE_PRINT expands to a brace block, so SAN_PRINT is a statement, not an expression.
#define SAN_PRINT(...) DEVICE_PRINT(__VA_ARGS__)
#endif

#define SAN_ASSERT(condition, message) LLK_ASSERT(condition, message)
