// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/compute_kernel_api.h"

#if defined(TRISC_MATH) && defined(ARCH_BLACKHOLE)
#include "experimental/llk_sfpu/ckernel_sfpu_csa_index_remap.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE)
// Call before csa_index_remap, including when switching from another operation.
// Re-establish the common SFPU configuration, address modifiers and counters.
ALWI void csa_index_remap_init() { MATH((sfpu::CsaIndexRemap<>::init())); }

template <std::uint32_t ROW_OFFSET>
ALWI void csa_index_remap(std::uint32_t idst) {
    // One call advances over the 32 destination rows that make up one tile.
    MATH((sfpu::CsaIndexRemap<ROW_OFFSET>::run(idst)));
}
#endif

}  // namespace ckernel
