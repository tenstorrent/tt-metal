// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The named compile-time argument API, split out of api/compile_time_args.h.
//
// Deliberately has no #pragma once and no include guard of its own. It must stay
// re-includable because it depends on KERNEL_COMPILE_TIME_ARG_MAP, which
// named_ct_arg_map_generated.h defines per kernel. Under TT_METAL_JIT_PCH the
// precompiled prelude already contains compile_time_args.h, parsed before that
// macro existed, so a once-only header would have its declarations suppressed for
// good. Instead the body below is gated on the macro and made idempotent by
// TT_METAL_NAMED_CT_ARGS_DEFINED, which is only set once the macro is present.
// That lets the generated map header include this file straight after defining
// the macro, whether or not a PCH is in play.
//
// Kept separate from compile_time_args.h so the generated header can pull in just
// this part: compile_time_args.h needs FORCE_INLINE, which is not yet defined at
// the point a force-included header is processed.

#if defined(KERNEL_COMPILE_TIME_ARG_MAP) && !defined(TT_METAL_NAMED_CT_ARGS_DEFINED)
#define TT_METAL_NAMED_CT_ARGS_DEFINED

#include <cstdint>
#include <string_view>
#include <utility>

namespace {
constexpr std::pair<std::string_view, uint32_t> named_args_map[] = {KERNEL_COMPILE_TIME_ARG_MAP};
}

// TODO #28026: Migrate to C++20 standards when available, see related issue for more details.
constexpr uint32_t get_named_ct_arg(std::string_view name) {
    for (const auto& [arg_name, arg_value] : named_args_map) {
        if (name == arg_name) {
            return arg_value;
        }
    }
    // This should never be reached if the named argument is defined in KERNEL_COMPILE_TIME_ARG_MAP.
    // Upon reaching this point, compilation should fail.
    // Note: Compilation currently fails with a segfault.
    __builtin_unreachable();  // Invalid named compile time argument
}

// clang-format off
/**
 * Returns the value of a named constexpr argument from kernel_compile_time_args array provided during kernel creation using
 * CreateKernel calls. The name-to-index mapping is defined via KERNEL_COMPILE_TIME_ARG_MAP. Migrating all existing kernels to
 * use named compile time arguments is not a trivial task, so backward-compatibility is maintained by allowing the use of the
 * get_compile_time_arg_val function with an index. get_compile_time_arg_val can be deprecated in the future upon completion
 * of the migration and by request. See vecadd_multi_core.cpp for an example of how to use named compile time arguments.
 * Note: Return value must be stored in a constexpr variable to guarantee compile time evaluation.
 *
 * Return value: constexpr uint32_t
 *
 * | Argument              | Description                        | Type                  | Valid Range | Required |
 * |-----------------------|------------------------------------|-----------------------|-------------|----------|
 * | arg_name              | The name of the argument           | string literal        | defined names | True   |
 */
// clang-format on
constexpr uint32_t get_named_compile_time_arg_val(std::string_view name) { return get_named_ct_arg(name); }

#endif  // KERNEL_COMPILE_TIME_ARG_MAP && !TT_METAL_NAMED_CT_ARGS_DEFINED
