// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>

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
namespace bench_config {

#if defined(_LIBCPP_VERSION)
inline constexpr std::size_t kInlinePointers = 3;  // libc++
inline constexpr const char* kStdlibName = "libc++";
#else
inline constexpr std::size_t kInlinePointers = 2;  // libstdc++
inline constexpr const char* kStdlibName = "libstdc++";
#endif

inline constexpr std::size_t kInlineBytes = kInlinePointers * sizeof(void*);

}  // namespace bench_config
