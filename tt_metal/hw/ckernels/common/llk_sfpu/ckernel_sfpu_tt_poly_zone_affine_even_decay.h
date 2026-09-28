// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
namespace sfpi {
#include "ckernel_sfpu_tt_poly_mirrored_terminals.h"
}
#include "ckernel_sfpu_tt_poly_zone_affine_even_decay_core.h"
namespace ckernel::sfpu::ttpoly {
template <class Config, int Iterations = 32>
inline void calculate_zone_affine_even_decay() {
    static_assert(Iterations == 32, "selected indexed zone body requires the complete tile");
    sfpi::zone_affine_even_decay_init<Config>();
    sfpi::zone_affine_even_decay_tile<Config>(
        [](sfpi::vFloat raw) { return raw; },
        []([[maybe_unused]] sfpi::vFloat raw, [[maybe_unused]] sfpi::vFloat& result) {
#if defined(ARCH_BLACKHOLE)
            sfpi::negative_infinity_terminal<3>(raw, result);
#elif !defined(ARCH_WORMHOLE)
#error "selected affine-even decay requires BH or WH"
#endif
        });
}
}  // namespace ckernel::sfpu::ttpoly
#define TT_POLY_LLK_ZONE_AFFINE_EVEN_DECAY_SELECTED_V1 1
