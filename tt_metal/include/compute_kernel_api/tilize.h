// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#include "llk_math_reduce_api.h"
#include "llk_math_matmul_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_tilize_api.h"
#include "llk_unpack_common_api.h"
#endif

namespace ckernel {

// clang-format off
/**
 * Initializes the tilize operation. Should be called once at the beginning of a kernel.
 *
 * Return value: None
 *
 * | Param Type | Name   | Description                                   | Type     | Valid Range | Required |
 * |----------- |--------|-----------------------------------------------|----------|-------------|----------|
 * | Function   | icb    | Input circular buffer identifier              | uint32_t | 0 to 31     | True     |
 * | Function   | block  | Size of tile block to work on                 | uint32_t | > 0         | True     |
 * | Function   | ocb    | Output circular buffer identifier             | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void tilize_init(uint32_t icb, uint32_t block, uint32_t ocb) {
    UNPACK((llk_unpack_tilize_init(icb, block)));
    MATH((llk_math_eltwise_unary_datacopy_init<
          A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false /*is_int_en*/,
          true /*tilize en*/>(icb)));
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_init<false /*untilize*/, false /*zero output*/, true /*tilize en*/>(ocb)));
#endif
}

#if (defined(REDUCE_OP) and defined(REDUCE_DIM)) or defined(__DOXYGEN__)

// clang-format off
/**
 * Initializes the tilize operation with reduction. Should be called once at the beginning of a kernel.
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                              | Type     | Valid Range | Required |
 * |------------|----------------|------------------------------------------|----------|-------------|----------|
 * | Template   | neginf_srcA    | NegInf source A flag                     | bool     | true/false  | False    |
 * | Template   | zero_srcA_reduce| Zero source A for reduce flag           | bool     | true/false  | False    |
 * | Function   | icb0           | Input circular buffer A identifier       | uint32_t | 0 to 31     | True     |
 * | Function   | icb1_scaler    | Input circular buffer for scaler         | uint32_t | 0 to 31     | True     |
 * | Function   | block          | Size of tile block to work on            | uint32_t | > 0         | True     |
 * | Function   | ocb            | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 * | Function   | num_faces      | Number of faces per tile                 | uint32_t | 1 to 4      | False    |
 * | Function   | face_r_dim     | Number of rows in each face              | uint32_t | 1 to 16     | False    |
 */
// clang-format on
template <bool neginf_srcA = true, bool zero_srcA_reduce = false>
ALWI void tilizeA_B_reduce_init(
    uint32_t icb0,
    uint32_t icb1_scaler,
    uint32_t block,
    uint32_t ocb,
    uint32_t num_faces = 4,
    uint32_t face_r_dim = 16) {
    UNPACK((llk_unpack_hw_configure<DST_ACCUM_MODE>(icb0, icb1_scaler)));
    UNPACK((llk_unpack_tilizeA_B_init<neginf_srcA, true, false, zero_srcA_reduce>(
        icb0, icb1_scaler, block, num_faces, face_r_dim, 1)));

    MATH((llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, DST_ACCUM_MODE, MATH_FIDELITY>()));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure<DST_ACCUM_MODE>(icb0, icb1_scaler)));

    PACK((llk_pack_hw_configure_disaggregated<DST_ACCUM_MODE, false>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<DST_ACCUM_MODE, false>(ocb)));
}
#endif

// clang-format off
/**
 * Re-initializes the tilize operation and reconfigures the unpacker with CB data type.
 *
 * Return value: None
 *
 * | Param Type | Name     | Description                              | Type     | Valid Range | Required |
 * |----------- |----------|------------------------------------------|----------|-------------|----------|
 * | Function   | old_icb  | Previous input circular buffer identifier| uint32_t | 0 to 31     | True     |
 * | Function   | new_icb  | New input circular buffer identifier     | uint32_t | 0 to 31     | True     |
 * | Function   | block    | Size of tile block to work on            | uint32_t | > 0         | True     |
 * | Function   | ocb      | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void tilize_init_short_with_dt(uint32_t old_icb, uint32_t new_icb, uint32_t block, uint32_t ocb) {
    MATH((llk_math_eltwise_unary_datacopy_init<
          A2D,
          DST_ACCUM_MODE,
          BroadcastType::NONE,
          false /*is_int_en*/,
          true /*tilize en*/>(new_icb)));
    // This reconfig call checks if old operand has different data format to
    // new operand idx, otherwise no reconfig call occurs
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(old_icb, new_icb)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_icb, new_icb)));
    UNPACK((llk_unpack_tilize_init(new_icb, block)));

#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_init<false, false, true /*tilize en*/>(ocb)));
#endif
}

// clang-format off
/**
 * Performs the tilize operation on a block.
 *
 * Return value: None
 *
 * | Param Type | Name             | Description                              | Type     | Valid Range | Required |
 * |----------- |------------------|------------------------------------------|----------|-------------|----------|
 * | Function   | icb              | Input circular buffer identifier         | uint32_t | 0 to 31     | True     |
 * | Function   | block            | Size of tile block to work on            | uint32_t | > 0         | True     |
 * | Function   | ocb              | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 * | Function   | input_tile_index | Index of the input tile in the icb       | uint32_t | >= 0        | False    |
 * | Function   | output_tile_index| Index of the output tile in the ocb      | uint32_t | >= 0        | False    |
 */
// clang-format on
ALWI void tilize_block(
    uint32_t icb, uint32_t block, uint32_t ocb, uint32_t input_tile_index = 0, uint32_t output_tile_index = 0) {
    UNPACK((llk_unpack_tilize_block(icb, block, input_tile_index)));

    for (uint32_t t = 0; t < block; t++) {
        // Acquire dst
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));

        // Datacopy
        MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
            0 /*dst index*/)));
        PACK((llk_pack<DST_ACCUM_MODE, true, false>(0 /*tile index*/, ocb, t + output_tile_index)));

        // Release dest
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// clang-format off
/**
 * Unpacks and tilizes a block from two input CBs.
 *
 * Return value: None
 *
 * | Param Type | Name             | Description                              | Type         | Valid Range | Required |
 * |------------|------------------|------------------------------------------|--------------|-------------|----------|
 * | Template   | neginf_srcA      | NegInf source A flag                     | bool         | true/false  | False    |
 * | Template   | reload_srcB      | Reload source B flag                     | std::uint32_t| true/false  | False    |
 * | Template   | zero_srcA        | Zero source A flag                       | bool         | true/false  | False    |
 * | Template   | zero_srcA_reduce | Zero source A for reduce flag            | bool         | true/false  | False    |
 * | Function   | icb0             | Input circular buffer A identifier       | uint32_t     | 0 to 31     | True     |
 * | Function   | icb1             | Input circular buffer B identifier       | uint32_t     | 0 to 31     | True     |
 * | Function   | block            | Size of tile block to work on            | uint32_t     | > 0         | True     |
 * | Function   | tile_idx_b       | Tile index for source B                  | uint32_t     | >= 0        | True     |
 * | Function   | num_faces        | Number of faces per tile                 | uint32_t     | 1 to 4      | False    |
 * | Function   | srca_face_r_dim  | Number of rows in each face (A)          | uint32_t     | 1 to 16     | False    |
 */
// clang-format on
template <
    bool neginf_srcA = true,
    std::uint32_t reload_srcB = true,
    bool zero_srcA = false,
    bool zero_srcA_reduce = false>
ALWI void unpack_tilizeA_B_block(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t block,

    uint32_t tile_idx_b,
    uint32_t num_faces = 4,
    uint32_t srca_face_r_dim = 16) {
    UNPACK((llk_unpack_tilizeA_B_block<neginf_srcA, reload_srcB, zero_srcA, zero_srcA_reduce>(
        icb0, icb1, block, tile_idx_b, num_faces, srca_face_r_dim)));
}

// clang-format off
/**
 * Uninitializes the tilize operation before re-initializing for another operation.
 *
 * NOTE: This function is not in line with our programming model, and will be removed by the end of 2025
 * as a part of tt-metal#22904.
 *
 * Return value: None
 *
 * | Param Type | Name   | Description                              | Type     | Valid Range | Required |
 * |----------- |--------|------------------------------------------|----------|-------------|----------|
 * | Function   | icb    | Input circular buffer identifier         | uint32_t | 0 to 31     | True     |
 * | Function   | ocb    | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void tilize_uninit(uint32_t icb, uint32_t ocb) {
    UNPACK((llk_unpack_tilize_uninit(icb)));
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_init<false /*untilize*/, false /*zero output*/, false /*tilize en*/>(ocb)));
#endif
}

// clang-format off
/**
 * Uninitializes the tilize operation and reconfigures the unpacker with CB data types.
 *
 * NOTE: This function is not in line with our programming model, and will be removed by the end of 2025
 * as a part of tt-metal#22904.
 *
 * Return value: None
 *
 * | Param Type | Name     | Description                              | Type     | Valid Range | Required |
 * |----------- |----------|------------------------------------------|----------|-------------|----------|
 * | Function   | old_icb  | Previous input circular buffer identifier| uint32_t | 0 to 31     | True     |
 * | Function   | new_icb  | New input circular buffer identifier     | uint32_t | 0 to 31     | True     |
 * | Function   | ocb      | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void tilize_uninit_with_dt(uint32_t old_icb, uint32_t new_icb, uint32_t ocb) {
    UNPACK((llk_unpack_tilize_uninit(old_icb)));
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(old_icb, new_icb)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(old_icb, new_icb)));
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_init(ocb)));
#endif
}

ALWI void fast_tilize_init(uint32_t icb, uint32_t full_dim, uint32_t ocb) {
#ifdef ARCH_BLACKHOLE
    // Blackhole fallback
    tilize_init(icb, full_dim, ocb);
#else
    UNPACK((llk_unpack_fast_tilize_init(icb, full_dim)));
    MATH((llk_math_fast_tilize_init(icb, full_dim == 1 ? 1 : 2)));
    PACK((llk_pack_fast_tilize_init(icb, ocb, full_dim == 1 ? 1 : 2)));
#endif
}

ALWI void fast_tilize_init_with_dt(uint32_t icb, uint32_t full_dim, uint32_t ocb) {
    UNPACK((llk_unpack_reconfig_data_format<DST_ACCUM_MODE>(icb, icb)));
    MATH((llk_math_reconfig_data_format<true, true>(icb, icb)));

    fast_tilize_init(icb, full_dim, ocb);
}

ALWI void fast_tilize_uninit(uint32_t icb, uint32_t ocb) {
#ifdef ARCH_BLACKHOLE
    // Blackhole fallback
    tilize_uninit(icb, ocb);
#else
    UNPACK((llk_unpack_fast_tilize_uninit<DST_ACCUM_MODE>()));
    MATH((llk_math_fast_tilize_uninit<DST_ACCUM_MODE>(icb)));
    PACK((llk_pack_fast_tilize_uninit<DST_ACCUM_MODE>(ocb)));
#endif
}

ALWI void fast_tilize_block(
    uint32_t icb, uint32_t block, uint32_t ocb, uint32_t input_tile_index = 0, uint32_t output_tile_index = 0) {
#ifdef ARCH_BLACKHOLE
    // Blackhole fallback
    tilize_block(icb, block, ocb, input_tile_index, output_tile_index);
#else
    uint32_t full_dim = block;

    // Not sure if input_tile_index can be arbitrary but it works for moving across rows of files,
    // i.e. input_tile_index % full_dim == 0
    input_tile_index = input_tile_index % full_dim + (input_tile_index / full_dim) * full_dim * TILE_R_DIM;

    uint32_t packed_tiles = 0;
    uint32_t remaining_tiles = block;
    uint32_t dest_size = DST_ACCUM_MODE ? 4 : 8;
    uint32_t unit_dim = full_dim == 1 ? 1 : 2;
    uint32_t num_units = dest_size / unit_dim;

    while (packed_tiles < block) {
        uint32_t read_tile_index = input_tile_index + packed_tiles;
        uint32_t write_tile_index = output_tile_index + packed_tiles;

        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));

        if (remaining_tiles > 2 * dest_size) {
            // Three or more dests
            UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
            MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
            PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
            packed_tiles += dest_size;
            remaining_tiles -= dest_size;
        } else if (remaining_tiles > dest_size) {
            // Two dests
            uint32_t even_remainder = remaining_tiles / 2 + ((remaining_tiles / 2) % 2);
            num_units = even_remainder / unit_dim;
            UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
            MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
            PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
            packed_tiles += even_remainder;
            remaining_tiles -= even_remainder;
        } else {
            // Last dest
            if (remaining_tiles % 2 == 0 || unit_dim == 1) {
                // Single sequence
                num_units = remaining_tiles / unit_dim;
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
                MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
                PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));
            } else if (remaining_tiles == 3) {
                // only odd pack
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, 3, 1, full_dim)));
                MATH((llk_math_fast_tilize_block_(0, icb, 3, 1)));
                PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, 3, 1)));
            } else {
                // even packs plus odd pack
                num_units = (remaining_tiles - 3) / unit_dim;
                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index, unit_dim, num_units, full_dim)));
                MATH((llk_math_fast_tilize_block_(0, icb, unit_dim, num_units)));
                PACK((llk_pack_fast_tilize_block(0, ocb, write_tile_index, unit_dim, num_units)));

                UNPACK((llk_unpack_fast_tilize_block(icb, read_tile_index + remaining_tiles - 3, 3, 1, full_dim)));
                MATH((llk_math_fast_tilize_block_(remaining_tiles - 3, icb, 3, 1)));
                PACK((llk_pack_fast_tilize_block(
                    remaining_tiles - 3, ocb, write_tile_index + remaining_tiles - 3, 3, 1)));
            }
            packed_tiles += remaining_tiles;
            remaining_tiles = 0;
        }

        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
#endif
}

// clang-format off
/**
 * Uninitializes the unpack tilizeA_B configuration and restores unpacker state
 * modified by _llk_unpack_tilizeA_B_init_.
 *
 * Return value: None
 *
 * Parameters:
 *
 * | Param Type | Name | Description           | Type     | Valid Range | Required |
 * |------------|------|-----------------------|----------|-------------|----------|
 * | Function   | icb  | Input circular buffer | uint32_t | 0 - 31.     | True     |
 *
 * Restored hardware state:
 *
 * | Field / Setting           | Scope      | Description                                           | Restored value / behavior                                                                  |
 * |---------------------------|------------|-------------------------------------------------------|--------------------------------------------------------------------------------------------|
 * | X-dim & base (ADCXX)      | UNP_A/B    | Face X-extent for address counters                    | face_r_dim * FACE_C_DIM elements, start at 0                                               |
 * | XY address counters       | UNP_A/B    | X/Y counters used by tilizeA_B y-stride pattern       | Counters reset to 0 (mask selects CH0/CH1 X/Y)                                             |
 * | ZW address counters       | UNP_A/B    | Z/W counters used for face/row stepping               | Counters reset to 0 for both unpackers                                                     |
 * | Out_data_format/config[0] | THCON_SEC0 | Unpack config[0]: out format, throttle, tilize, shift | out_data_format = unpack_dst_format; throttle_mode = 2; tileize_mode = 0; shift_amount = 0 |
 * | Tile_x_dim (cntx0)        | THCON_SEC0 | Tile X dimension per context for unpacker             | Restored to FACE_DIM_16x16 (16 | (16 << 16))                                               |
 */
// clang-format on
ALWI void unpack_tilizeA_B_uninit(uint32_t icb) { UNPACK((llk_unpack_tilizeA_B_uninit(icb))); }

}  // namespace ckernel
