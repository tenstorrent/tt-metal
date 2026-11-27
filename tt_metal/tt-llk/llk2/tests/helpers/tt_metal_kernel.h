// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// TEST ONLY. What a tt-metal compute kernel file expects from tt-metal's build, for the tt-llk harness, which builds
// the same file (a test source includes the kernel .cpp after this):
//
//   DST_ACCUM_MODE                      the harness's is_fp32_dest_acc_en (MATH_FIDELITY the harness defines itself)
//   get_compile_time_arg_val(i)         the test defines it, from its own parameters
//   get_named_compile_time_arg_val(n)   the test defines it with find_named_arg over a NamedArg table
//   DataflowBuffer                      api/dataflow/dataflow_buffer.h, next to this file
//   kernel_main()                       the kernel defines it; the test's run_kernel calls it

#pragma once

#include <cstddef>
#include <cstdint>

#define DST_ACCUM_MODE is_fp32_dest_acc_en

namespace llk2_test
{

/// One named compile-time arg: what the host passes as {name, value}.
struct NamedArg
{
    const char* name;    ///< the arg's name
    std::uint32_t value; ///< its value
};

/// Whether two C strings are equal, at compile time.
constexpr bool same_name(const char* a, const char* b)
{
    while (*a != '\0' && *a == *b)
    {
        ++a;
        ++b;
    }
    return *a == *b;
}

/// Not constexpr: reached only for a name the table lacks, so such a name fails the build where the kernel reads it.
inline void named_arg_missing()
{
}

/// The value of named arg `name` in args.
template <std::size_t N>
constexpr std::uint32_t find_named_arg(const NamedArg (&args)[N], const char* name)
{
    for (const NamedArg& arg : args)
    {
        if (same_name(arg.name, name))
        {
            return arg.value;
        }
    }
    named_arg_missing();
    return 0;
}

} // namespace llk2_test
