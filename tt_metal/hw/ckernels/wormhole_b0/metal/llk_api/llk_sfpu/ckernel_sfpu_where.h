// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

/*
Expects following input in Dst register:
Index 0 ( Tile 0 ) -> condition tensor
Index 32 ( Tile 1 ) -> true tensor
Index 64 ( Tile 2 ) -> false tensor

*/

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_where_fp16_b() {
    constexpr uint dst_tile_size = 32;

    sfpi::vFloat cond = sfpi::dst_reg[0];
    sfpi::vFloat output_tensor = 0;
    sfpi::vFloat true_tensor = 0;
    sfpi::vFloat false_tensor = 0;

    for (int i = 0; i < ITERATIONS; i++) {
        cond = sfpi::dst_reg[0];
        true_tensor = sfpi::dst_reg[dst_tile_size];
        false_tensor = sfpi::dst_reg[dst_tile_size * 2];

        v_if(cond == 0.0f) {
            output_tensor = false_tensor;
            // TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16, ADDR_MOD_3, 2 * dst_tile_size_rows);
        }
        v_else {
            output_tensor = true_tensor;
            // TTI_SFPLOAD(p_sfpu::LREG3, InstrModLoadStore::LO16, ADDR_MOD_3, dst_tile_size_rows);
        }
        v_endif;
        sfpi::dst_reg[0] = output_tensor;
        // TTI_SFPSTORE(p_sfpu::LREG3, InstrModLoadStore::LO16, ADDR_MOD_3, 0);

        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_where_fp32() {
    constexpr uint dst_tile_size = 32;

    sfpi::vFloat output_tensor = 0;
    sfpi::vFloat true_tensor = 0;
    sfpi::vFloat false_tensor = 0;
    sfpi::vFloat cond = sfpi::dst_reg[0];

    for (int i = 0; i < ITERATIONS; i++) {
        cond = sfpi::dst_reg[0];
        true_tensor = sfpi::dst_reg[dst_tile_size];
        false_tensor = sfpi::dst_reg[dst_tile_size * 2];

        v_if(cond != 0.0f) { output_tensor = true_tensor; }
        v_else { output_tensor = false_tensor; }
        v_endif;

        sfpi::dst_reg[0] = output_tensor;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_where_int32() {
    constexpr uint dst_tile_size = 32;

    sfpi::vInt output_tensor = 0;
    sfpi::vInt true_tensor = 0;
    sfpi::vInt false_tensor = 0;
    sfpi::vInt cond = sfpi::dst_reg[0];

    for (int i = 0; i < ITERATIONS; i++) {
        cond = sfpi::dst_reg[0];
        true_tensor = sfpi::dst_reg[dst_tile_size];
        false_tensor = sfpi::dst_reg[dst_tile_size * 2];

        v_if(cond != 0.0f) { output_tensor = true_tensor; }
        v_else { output_tensor = false_tensor; }
        v_endif;

        sfpi::dst_reg[0] = output_tensor;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, uint8_t data_format>
inline void calculate_where() {
    if (data_format == 0) {
        calculate_where_fp32<APPROXIMATION_MODE>();
        // _calculate_where_<APPROXIMATION_MODE, DataFormat::Float32>();
    } else if (data_format == 1) {
        calculate_where_fp16_b<APPROXIMATION_MODE>();
        // _calculate_where_<APPROXIMATION_MODE, DataFormat::Float16_b>();
    } else if (data_format == 2) {
        calculate_where_int32<APPROXIMATION_MODE>();
    }
}

}  // namespace sfpu
}  // namespace ckernel
