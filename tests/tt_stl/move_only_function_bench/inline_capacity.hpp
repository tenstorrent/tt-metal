// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>

// The inline capacity every contender is pinned to: whatever std::function actually gets in this
// configuration, so no side is handed a larger buffer than the baseline.
//
// The buffer belongs to the standard library, not the compiler, so clang with libstdc++ matches
// gcc. It cannot be derived from sizeof: libstdc++ is 32 B with a 16 B buffer (2 pointers of
// overhead) while libc++ is 48 B with a 24 B buffer (3 pointers), so any `sizeof - N * sizeof(void*)`
// formula is right for one and wrong for the other.
//
// These constants are therefore measured, not computed. sbo_probe verifies them at runtime and
// fails if a standard library moves the boundary.
namespace bench_config {

#if defined(_LIBCPP_VERSION)
inline constexpr std::size_t kInlineBytes = 24;  // libc++
inline constexpr const char* kStdlibName = "libc++";
#else
inline constexpr std::size_t kInlineBytes = 16;  // libstdc++
inline constexpr const char* kStdlibName = "libstdc++";
#endif

inline constexpr std::size_t kInlinePointers = kInlineBytes / sizeof(void*);

}  // namespace bench_config
