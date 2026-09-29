// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common.h"

// Gated to the arches that have both LLK API halves under tt_metal/hw/ckernels/<arch>/metal/llk_api/
// experimental/; a new arch must add those before it can pull this in.
#if defined(TRISC_MATH) && (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE) || defined(ARCH_QUASAR))
#include "experimental/llk_math_eltwise_binary_custom_api.h"
#endif

#if defined(TRISC_UNPACK) && (defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE) || defined(ARCH_QUASAR))
#include "experimental/llk_unpack_AB_sub_bcast_col_custom_api.h"
#endif

namespace ckernel {

#if defined(ARCH_BLACKHOLE) || defined(ARCH_WORMHOLE) || defined(ARCH_QUASAR)

ALWI void sub_bcast_cols_init_short_custom(
    std::uint32_t icb0, std::uint32_t icb1, std::uint32_t ct_dim, std::uint32_t call_line = __builtin_LINE()) {
    state_configure(icb0, icb1, call_line);
    MATH((llk_math_eltwise_binary_sub_bcast_cols_init_custom<MATH_FIDELITY>(icb0, icb1)));
    UNPACK((llk_unpack_AB_sub_bcast_col_init_custom(icb0, icb1)));
}

ALWI void sub_tiles_bcast_cols_custom(
    std::uint32_t icb0,
    std::uint32_t icb1,
    std::uint32_t itile0,
    std::uint32_t itile1,
    std::uint32_t idst,
    std::uint32_t ct_dim) {
    MATH((llk_math_eltwise_binary_sub_bcast_cols_custom(icb0, idst, ct_dim)));
    UNPACK((llk_unpack_AB_sub_bcast_col_custom(icb0, icb1, itile0, itile1, ct_dim)));
}

#endif

}  // namespace ckernel
