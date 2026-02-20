// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "tensix_types.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

// Generalized copy_dest_value that works with any DataFormat
template <DataFormat DATA_FORMAT, bool APPROXIMATION_MODE, int ITERATIONS = 8>
void copy_dest_value(const uint dst_index_in, const uint dst_index_out, const uint /* unused */) {
    constexpr uint8_t instr_mod_index = GetSfpLoadStoreInstrMod<DATA_FORMAT>();
    // size of each tile in Dest is 64 rows
    constexpr uint dst_tile_size = 64;
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG0, instr_mod_index, ADDR_MOD_7, dst_index_in * dst_tile_size);
        TT_SFPSTORE(p_sfpu::LREG0, instr_mod_index, ADDR_MOD_7, dst_index_out * dst_tile_size);
        dst_reg++;
    }
}

// Deprecated: Use the DataFormat template parameter version instead
template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
[[deprecated("Use copy_dest_value<DataFormat, APPROXIMATION_MODE, ITERATIONS> instead")]]
void copy_dest_value(const uint dst_index_in, const uint dst_index_out, const uint /* unused */) {
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr uint dst_tile_size_sfpi = 32;
        dst_reg[dst_index_out * dst_tile_size_sfpi] = dst_reg[dst_index_in * dst_tile_size_sfpi];
        dst_reg++;
    }
}

void copy_dest_value_init() {
    // No initialization required
}

}  // namespace sfpu
}  // namespace ckernel
