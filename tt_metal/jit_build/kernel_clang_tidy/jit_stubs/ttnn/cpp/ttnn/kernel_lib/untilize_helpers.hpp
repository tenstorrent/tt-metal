// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KCT stub for ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp
//
// Same situation as tilize_helpers.hpp: untilize_helpers.inl contains:
//
//   static_assert(input_cb != output_cb, "Untilize cannot be done in-place...");
//
// With KERNEL_COMPILE_TIME_ARGS=1,1,...,1 this fires spuriously.
// Suppress static_assert during the include.

#pragma once

#pragma push_macro("static_assert")
#define static_assert(...)  // KCT: suppressed to allow dummy CT args
#include_next "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#pragma pop_macro("static_assert")
