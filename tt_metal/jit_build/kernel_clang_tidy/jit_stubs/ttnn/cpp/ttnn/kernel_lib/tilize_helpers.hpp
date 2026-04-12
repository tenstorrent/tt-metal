// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KCT stub for ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp
//
// The real header includes tilize_helpers.inl which contains:
//
//   static_assert(input_cb != output_cb, "Tilize cannot be done in-place...");
//
// With KERNEL_COMPILE_TIME_ARGS=1,1,...,1 all CT args evaluate to 1, so
// any kernel that passes the same CT arg index for both input_cb and output_cb
// triggers this assertion (e.g. paged_cache, groupnorm, conv3d compute kernels).
//
// For KCT analysis these assertions fire spuriously — we just need clang to
// parse the code, not validate the runtime configuration.  We temporarily
// suppress static_assert (clang-20 allows #define static_assert as an
// extension) before including the real header.
//
// Note: #pragma push/pop_macro("static_assert") is supported by clang.

#pragma once

#pragma push_macro("static_assert")
#define static_assert(...)  // KCT: suppressed to allow dummy CT args
#include_next "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#pragma pop_macro("static_assert")
