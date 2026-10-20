// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_assert.h"

namespace ckernel {
namespace sfpu {

// cumprod_tile is not ported to Quasar. The Compute API keeps it on every arch, so its kernel
// entry points reject the call here.
template <bool APPROXIMATION_MODE /*unused*/, int ITERATIONS = 8 /*unused*/>
inline void calculate_cumprod([[maybe_unused]] const bool first) {
    LLK_ASSERT(false, "cumprod_tile is not supported on Quasar");
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void cumprod_init() {
    LLK_ASSERT(false, "cumprod_tile is not supported on Quasar");
}

}  // namespace sfpu
}  // namespace ckernel
