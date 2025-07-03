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
          true /*tilize en*/>(false /*transpose of faces*/, false /*transpose within 16x16 face*/, icb)));
#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_init<false /*untilize*/, false /*skip_inputs*/, true /*tilize en*/>(ocb)));
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
    UNPACK((llk_unpack_tilizeA_B_hw_configure_disaggregated<DST_ACCUM_MODE>(icb0, icb1_scaler)));
    UNPACK((llk_unpack_tilizeA_B_init<neginf_srcA, true, false, zero_srcA_reduce>(
        icb0, icb1_scaler, block, num_faces, face_r_dim, 1)));

    MATH((llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>()));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure_disaggregated(icb0, icb1_scaler)));

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
          true /*tilize en*/>(false /*transpose of faces*/, false /*transpose within 16x16 face*/, new_icb)));
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
 * | Param Type | Name   | Description                              | Type     | Valid Range | Required |
 * |----------- |--------|------------------------------------------|----------|-------------|----------|
 * | Function   | icb    | Input circular buffer identifier         | uint32_t | 0 to 31     | True     |
 * | Function   | block  | Size of tile block to work on            | uint32_t | > 0         | True     |
 * | Function   | ocb    | Output circular buffer identifier        | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void tilize_block(uint32_t icb, uint32_t block, uint32_t ocb) {
    UNPACK((llk_unpack_tilize_block(icb, block)));

    for (uint32_t t = 0; t < block; t++) {
        // Acquire dst
        MATH((llk_math_wait_for_dest_available()));
        PACK((llk_packer_wait_for_math_done()));

        // Datacopy
        MATH((llk_math_eltwise_unary_datacopy<A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
            0 /*dst index*/)));
        PACK((llk_pack<DST_ACCUM_MODE, false, false>(0 /*tile index*/, ocb)));

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
    PACK((llk_pack_init<false /*untilize*/, false /*skip_inputs*/, false /*tilize en*/>(ocb)));
#endif
}

// clang-format off
/**
 * Uninitializes the tilize operation and reconfigures the unpacker with CB data types.
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

}  // namespace ckernel
