// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_ops.h"

namespace ckernel {
namespace sfpu {

// Calculates ADD for one pair of rows (Quasar SFPU ops cover 2 rows)
inline void _calculate_add_rows_(
    const int in0_addr,
    const int in1_addr,
    const int store_addr,
    const std::uint32_t load_sfpmem,
    const std::uint32_t store_sfpmem) {
    TT_SFPLOAD(p_sfpu::LREG0, load_sfpmem, ADDR_MOD_7, 0, in0_addr);
    TT_SFPLOAD(p_sfpu::LREG1, load_sfpmem, ADDR_MOD_7, 0, in1_addr);
    TTI_SFPADD(p_sfpu::LCONST_1, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LREG2, 0x0);
    TT_SFPSTORE(p_sfpu::LREG2, store_sfpmem, ADDR_MOD_7, 0, store_addr);
}

// Addresses select Dest (bit 10 = 0) or SrcS (bit 10 = 1). Float16 needs an explicit FP16A.
inline void _calculate_add_(
    const int in0_base_addr,
    const int in1_base_addr,
    const int store_base_addr,
    const int num_sfpu_iterations,
    const std::uint32_t load_sfpmem,
    const std::uint32_t store_sfpmem) {
#pragma GCC unroll 8
    for (int d = 0; d < num_sfpu_iterations; d++) {
        _calculate_add_rows_(
            in0_base_addr + (d << 1), in1_base_addr + (d << 1), store_base_addr + (d << 1), load_sfpmem, store_sfpmem);
    }
}

}  // namespace sfpu
}  // namespace ckernel
