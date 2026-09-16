// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"

#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_sfpu/llk_math_eltwise_unary_sfpu_csa_index_remap.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)
// Call before csa_index_remap, including when switching from another operation.
// Re-establish the common SFPU configuration, address modifiers and counters.
ALWI void csa_index_remap_init() { MATH((llk_math_eltwise_unary_sfpu_csa_index_remap_init<DST_ACCUM_MODE>())); }

template <std::uint32_t ROW_OFFSET>
ALWI void csa_index_remap(std::uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_csa_index_remap<ROW_OFFSET>(idst)));
}
#endif

}  // namespace ckernel
