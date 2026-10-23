// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/compute_kernel_api.h"

#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "sfpu/experimental/ckernel_sfpu_csa_index_remap.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)
// Call before csa_index_remap, including when switching from another operation.
// Re-establish the common SFPU configuration, address modifiers and counters.
ALWI void csa_index_remap_init() { MATH(SFPU_UNARY_INIT(unused)); }

template <std::uint32_t ROW_OFFSET>
ALWI void csa_index_remap(std::uint32_t idst) {
    // RC_custom advances over the 32 destination rows that make up one tile.
    MATH(SFPU_UNARY_CALL(
        DST_SYNC_MODE, DST_ACCUM_MODE, _csa_index_remap_, (32, ROW_OFFSET), idst, VectorMode::RC_custom));
}
#endif

}  // namespace ckernel
