// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>

// A translation unit measuring one candidate's parse and instantiation cost must not pay for the
// others, so it defines these before including this header. bench.cpp wants all three.
#ifndef BENCH_SKIP_FU2
#include <function2/function2.hpp>
#endif
#ifndef BENCH_SKIP_ZOO
#include <zoo/FunctionPolicy.h>
#endif

// The single definition of each contender, so every translation unit measures the same types.
namespace bench {

// The inline capacity every contender is pinned to: whatever std::function actually has in the
// current configuration, so no side is handed a larger buffer than the baseline.
//
// The buffer belongs to the standard library, not the compiler, so clang with libstdc++ matches
// gcc. Both libraries size it in pointers rather than bytes, which is how it is expressed here:
//
//   libstdc++  2 pointers  - _Any_data is a union whose widest member is a pointer to member
//                            function, two pointers wide under the Itanium ABI
//   libc++     3 pointers  - __buf_ is declared as char[3 * sizeof(void*)]
//
// It cannot be recovered from sizeof(std::function) either: the overhead differs between the two
// implementations (2 pointers vs 3), so any `sizeof - N * sizeof(void*)` rule is correct for one
// and wrong for the other. These values are measured instead, and sbo_probe verifies them at
// runtime and fails if a standard library moves the boundary.
#if defined(_LIBCPP_VERSION)
inline constexpr std::size_t kInlinePointers = 3;  // libc++
inline constexpr const char* kStdlibName = "libc++";
#else
inline constexpr std::size_t kInlinePointers = 2;  // libstdc++
inline constexpr const char* kStdlibName = "libstdc++";
#endif

inline constexpr std::size_t kInlineBytes = kInlinePointers * sizeof(void*);

// --- The contenders ---------------------------------------------------------------------------

template <typename Signature>
using StdFunction = std::function<Signature>;

#ifndef BENCH_SKIP_FU2
// HasStrongExceptGuarantee=true makes the move constructor noexcept, matching std::function, zoo
// and the std::move_only_function this will become. fu2's own unique_function default is false,
// which would leave this the only contender with a throwing move. The cost is that fu2 then
// refuses callables whose move can throw; see README.
template <typename Signature>
using Fu2Function = fu2::function_base<
    /*IsOwning=*/true,
    /*IsCopyable=*/false,
    fu2::capacity_fixed<kInlineBytes>,
    /*IsThrowing=*/true,
    /*HasStrongExceptGuarantee=*/true,
    Signature>;
#endif

#ifndef BENCH_SKIP_ZOO
// zoo::VTableFunction is deliberately not used: AnyContainer provides no operator bool and no
// has_value(), so it cannot stand in for std::move_only_function. Both live on zoo::Function, and
// the RTTI affordance is what makes operator bool reliable -- without it the fallback compares
// destructor pointers against Destroy::noOp, which identical-code folding can merge for a
// trivially destructible target, making an engaged function report itself empty.
template <typename Signature>
using ZooFunction = zoo::
    Function<zoo::AnyContainer<zoo::Policy<void* [kInlinePointers], zoo::Destroy, zoo::Move, zoo::RTTI>>, Signature>;
#endif

}  // namespace bench
