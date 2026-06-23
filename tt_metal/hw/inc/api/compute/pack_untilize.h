// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "llk_assert.h"
#ifdef TRISC_MATH
#include "llk_math_unary_datacopy_api.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_unpack_A_api.h"
#endif
#ifdef TRISC_PACK
#include "llk_pack_untilize_api.h"
#include "llk_pack_tile_api.h"
#endif

namespace ckernel {

namespace pack_untilize_detail {

template <
    uint32_t block_ct_dim,
    uint32_t full_ct_dim,
    bool narrow_row,
    std::uint32_t row_num_datums,
    bool dense,
    bool configure_remap>
ALWI void pack_untilize_dest_init_impl(uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    state_configure<Operand::PACK>(ocb, call_line);
#ifdef ARCH_BLACKHOLE
    // Needed for setting swizzle_32b on Blackhole; llk_math_reconfig_remap is a no-op on Wormhole.
    // TODO NC: A workaround for tt-metal#17132. Should be addressed more systematically in tt-llk#989
    if constexpr (configure_remap) {
        MATH((llk_math_reconfig_remap(true /*remap_enable*/)));
    }
#endif
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(ocb)));
    PACK((
        llk_pack_untilize_init<block_ct_dim, full_ct_dim, false /*diagonal*/, narrow_row, row_num_datums, dense>(ocb)));
    PACK((llk_init_packer_dest_offset_registers<PackMode::Untilize, false /*diagonal*/>()));
#else
    LLK_ASSERT(narrow_row == false, "narrow_row not supported on Quasar");
    PACK((llk_pack_untilize_init<block_ct_dim, full_ct_dim>(ocb)));
#endif
}

// @deprecated Face geometry is now derived from the output CB metadata. Use the
// pack_untilize_dest_init_impl(uint32_t ocb, uint32_t call_line) overload instead. This overload is retained
// only for backwards compatibility and will be removed.
template <
    uint32_t block_ct_dim,
    uint32_t full_ct_dim,
    bool narrow_row,
    std::uint32_t row_num_datums,
    bool dense,
    bool configure_remap>
[[deprecated(
    "Face geometry is now derived from the output CB metadata; use the "
    "pack_untilize_dest_init_impl(uint32_t, uint32_t) overload instead.")]] ALWI void
pack_untilize_dest_init_impl(
    uint32_t ocb, uint32_t face_r_dim, uint32_t num_faces, uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    state_configure<Operand::PACK>(ocb, call_line);
#ifdef ARCH_BLACKHOLE
    // Needed for setting swizzle_32b on Blackhole; llk_math_reconfig_remap is a no-op on Wormhole.
    // TODO NC: A workaround for tt-metal#17132. Should be addressed more systematically in tt-llk#989
    if constexpr (configure_remap) {
        MATH((llk_math_reconfig_remap(true /*remap_enable*/)));
    }
#endif
// These calls intentionally use the deprecated explicit-face-geometry LLK APIs to preserve the
// original behavior of this compatibility overload.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    PACK((llk_pack_reconfig_data_format_disaggregated<DST_ACCUM_MODE>(ocb, face_r_dim, num_faces)));
    PACK((llk_pack_untilize_init<block_ct_dim, full_ct_dim, false /*diagonal*/, narrow_row, row_num_datums, dense>(
        ocb, face_r_dim, num_faces)));
#pragma GCC diagnostic pop
    PACK((llk_init_packer_dest_offset_registers<PackMode::Untilize, false /*diagonal*/>()));
#else
    LLK_ASSERT(narrow_row == false, "narrow_row not supported on Quasar");
    PACK((llk_pack_untilize_init<block_ct_dim, full_ct_dim>(ocb)));
#endif
}

template <uint32_t block_ct_dim, uint32_t full_ct_dim, bool configure_remap>
ALWI void pack_untilize_init_impl(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
#ifndef ARCH_QUASAR
    state_configure<Operand::SRCA, Operand::PACK>(icb, ocb, call_line);
    UNPACK((
        llk_unpack_A_init<BroadcastType::NONE, false /*acc_to_dest*/, EltwiseBinaryReuseDestType::NONE, UnpackToDestEn>(
            false /*transpose_of_faces*/,
            false /*within_face_16x16_transpose*/,
            icb)));  // init must be after configure
    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE>(icb)));
#else
    UNPACK((llk_unpack_A_init<
            BroadcastType::NONE,
            false /*acc_to_dest*/,
            EltwiseBinaryReuseDestType::NONE,
            false /*unpack_to_dest*/>(false /*transpose_of_faces*/, false /*within_face_16x16_transpose*/, icb)));
    MATH((llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE>(icb)));
#endif
    pack_untilize_dest_init_impl<
        block_ct_dim,
        full_ct_dim,
        false /*narrow_row*/,
        TILE_C_DIM,
        false /*dense*/,
        configure_remap>(ocb, call_line);
}

}  // namespace pack_untilize_detail

// clang-format off
/**
 * Performs the necessary hardware and software initialization for the pack untilize operation. This initialization
 * function should be used when the desired PACK input is already in DEST register - therefore, it doesn't
 * configure UNPACK and MATH threads for transferring data from circular buffers to DEST register (this is done with
 * `pack_untilize_init` function). Matching pack untilize operation for this initialization function is
 * `pack_untilize_dest`. In order for this untilization to be performed correctly, some other function must
 * place the tiles in the DEST register, e.g. `reduce_tile`, `copy_tile`, etc. This initialization function
 * should, therefore, be called right after the op-specific initialization function (`reduce_init`, `copy_init`, etc.).
 *
 * Since pack untilize works on a block of tiles, the user should specify the width of a single block (block_ct_dim),
 * and the width of the full block (`full_ct_dim`). It is not needed to provide the height of the block during the
 * initialization, since `pack_untilize_block` will loop over the height of the block. Note that the maximum size
 * of the block is limited by the size of the DEST and synchronization mode used. These are maximum sizes:
 * - half-sync mode (16-bit mode): 8 tiles
 * - half-sync mode (32-bit mode): 4 tiles
 * - full-sync mode (16-bit mode): 16 tiles
 * - full-sync mode (32-bit mode): 8 tiles
 *
 * NOTE: Face geometry (face_r_dim, num_faces) is derived from the output circular buffer metadata configured on
 * the host via CircularBufferConfig::set_unpack_face_geometry / CBFormatDescriptor::face_geometry. Callers that need
 * non-default face geometry must configure it on the output CB at program creation time.
 *
 * By default this init configures BH DEST remap. Pass `configure_remap = false` only when the caller has already
 * configured BH DEST remap and no intervening operation requires a different DEST remap state.
 *
 * Return value: None
 *
 * | Param Type | Name            | Description                                         | Type      | Valid Range               | Required              |
 * |------------|-----------------|-----------------------------------------------------|-----------|---------------------------|-----------------------|
 * | Template   | block_ct_dim    | Width of a single block in tiles                    | uint32_t  | 1 to max (see note)       | False (default = 8)   |
 * | Template   | full_ct_dim     | Width of a full input in tiles                      | uint32_t  | Divisible by block_ct_dim | False                 |
 * | Template   | narrow_row      | Whether the provided input is narrow                | bool      | true/false                | False                 |
 * | Template   | row_num_datums  | Number of datums per row                            | uint32_t  | >= 1                      | False                 |
 * | Template   | dense           | Packs two 2 face tiles in a single 4 face region    | bool      | true/false                | False (default false) |
 * | Template   | configure_remap | Whether to (re)configure BH DEST remap (BH only)    | bool      | true/false                | False (default true)  |
 * | Function   | ocb             | Output circular buffer identifier                   | uint32_t  | 0 to 31                   | True                  |
 */
// clang-format on
template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false,
    bool configure_remap = true>
ALWI void pack_untilize_dest_init(uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
    pack_untilize_detail::
        pack_untilize_dest_init_impl<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense, configure_remap>(
            ocb, call_line);
}

// clang-format off
/**
 * @deprecated Face geometry is now derived from the output circular buffer metadata configured on the host via
 * CircularBufferConfig::set_unpack_face_geometry / CBFormatDescriptor::face_geometry. Use the
 * `pack_untilize_dest_init(ocb)` overload (which takes face geometry from the output CB) instead. This
 * explicit-face-geometry overload is retained only for backwards compatibility and will be removed.
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                                       | Type     | Valid Range               | Required              |
 * |------------|----------------|---------------------------------------------------|----------|---------------------------|-----------------------|
 * | Template   | block_ct_dim   | Width of a single block in tiles                  | uint32_t | 1 to max (see note)       | False (default = 8)   |
 * | Template   | full_ct_dim    | Width of a full input in tiles                    | uint32_t | Divisible by block_ct_dim | False                 |
 * | Template   | narrow_row     | Whether the provided input is narrow              | bool     | true/false                | False                 |
 * | Template   | row_num_datums | Number of datums per row                          | uint32_t | >= 1                      | False                 |
 * | Template   | dense          | Packs two 2 face tiles in a single 4 face region  | bool     | true/false                | False (default false) |
 * | Function   | ocb            | Output circular buffer identifier                 | uint32_t | 0 to 31                   | True                  |
 * | Function   | face_r_dim     | Face height in rows                               | uint32_t | 1, 8 or 16                | True                  |
 * | Function   | num_faces      | Number of faces                                   | uint32_t | 1, 2 or 4                 | True                  |
 */
// clang-format on
template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false>
[[deprecated(
    "Face geometry is now derived from the output CB metadata; use the pack_untilize_dest_init(ocb) "
    "overload instead.")]] ALWI void
pack_untilize_dest_init(uint32_t ocb, uint32_t face_r_dim, uint32_t num_faces, uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    pack_untilize_detail::
        pack_untilize_dest_init_impl<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense, true>(
            ocb, face_r_dim, num_faces, call_line);
#pragma GCC diagnostic pop
}

// clang-format off
/**
 * @deprecated Use `pack_untilize_dest_init<..., configure_remap = false>(ocb)` instead. BH DEST remap is no
 * longer (re)configured here, and face geometry is now derived from the output circular buffer metadata. This
 * explicit-face-geometry skip-remap helper is retained only for backwards compatibility and will be removed.
 *
 * Return value: None
 *
 * | Param Type | Name           | Description                                       | Type     | Valid Range               | Required              |
 * |------------|----------------|---------------------------------------------------|----------|---------------------------|-----------------------|
 * | Template   | block_ct_dim   | Width of a single block in tiles                  | uint32_t | 1 to max (see note)       | False (default = 8)   |
 * | Template   | full_ct_dim    | Width of a full input in tiles                    | uint32_t | Divisible by block_ct_dim | False                 |
 * | Template   | narrow_row     | Whether the provided input is narrow              | bool     | true/false                | False                 |
 * | Template   | row_num_datums | Number of datums per row                          | uint32_t | >= 1                      | False                 |
 * | Template   | dense          | Packs two 2 face tiles in a single 4 face region  | bool     | true/false                | False (default false) |
 * | Function   | ocb            | Output circular buffer identifier                 | uint32_t | 0 to 31                   | True                  |
 * | Function   | face_r_dim     | Face height in rows                               | uint32_t | 1, 8 or 16                | False (default = 16)  |
 * | Function   | num_faces      | Number of faces                                   | uint32_t | 1, 2 or 4                 | False (default = 4)   |
 */
// clang-format on
template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    bool dense = false>
[[deprecated(
    "Use pack_untilize_dest_init<..., configure_remap = false>(ocb) instead; face geometry is now "
    "derived from the output CB metadata.")]] ALWI void
pack_untilize_dest_init_skip_remap(
    uint32_t ocb, uint32_t face_r_dim = 16, uint32_t num_faces = 4, uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    pack_untilize_detail::
        pack_untilize_dest_init_impl<block_ct_dim, full_ct_dim, narrow_row, row_num_datums, dense, false>(
            ocb, face_r_dim, num_faces, call_line);
#pragma GCC diagnostic pop
}

// clang-format off
/**
 * Performs the necessary hardware and software initialization for the pack untilize operation. Initializes all three
 * threads: UNPACK, MATH, and PACK. This function should be used when the desired PACK input is not yet in DEST
 * register. Internally, this function calls `pack_untilize_dest_init` to initialize the PACK thread.
 *
 * Since pack untilize works on a block of tiles, the user should specify the width of a single block (block_ct_dim),
 * and the width of the full block (`full_ct_dim`). It is not needed to provide the height of the block during the
 * initialization, since `pack_untilize_block` will loop over the height of the block. Note that the maximum size
 * of the block is limited by the size of the DEST and synchronization mode used. These are maximum sizes:
 * - half-sync mode (16-bit mode): 8 tiles
 * - half-sync mode (32-bit mode): 4 tiles
 * - full-sync mode (16-bit mode): 16 tiles
 * - full-sync mode (32-bit mode): 8 tiles
 *
 * NOTE: Face geometry (face_r_dim, num_faces) is derived from the output circular buffer metadata configured on
 * the host via CircularBufferConfig::set_unpack_face_geometry / CBFormatDescriptor::face_geometry.
 *
 * This default init configures BH DEST remap. Use `pack_untilize_init_skip_remap` only when the caller has already
 * configured BH DEST remap and no intervening operation requires a different DEST remap state.
 *
 * Return value: None
 *
 * | Param Type | Name         | Description                                | Type      | Valid Range               | Required                |
 * |------------|--------------|--------------------------------------------|-----------|---------------------------|-------------------------|
 * | Template   | block_ct_dim | Width of a single block in tiles           | uint32_t  | 1 to max (see note)       | False (default = 8)     |
 * | Template   | full_ct_dim  | Width of a full input in tiles             | uint32_t  | Divisible by block_ct_dim | False                   |
 * | Function   | icb          | Input circular buffer identifier           | uint32_t  | 0 to 31                   | True                    |
 * | Function   | ocb          | Output circular buffer identifier          | uint32_t  | 0 to 31                   | True                    |
 */
// clang-format on
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
ALWI void pack_untilize_init(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
    pack_untilize_detail::pack_untilize_init_impl<block_ct_dim, full_ct_dim, true>(icb, ocb, call_line);
}

// clang-format off
/**
 * Same as `pack_untilize_init`, but does not (re)configure BH DEST remap. Use this when the caller has
 * already configured BH DEST remap (and no intervening operation requires a different DEST remap state).
 * On non-Blackhole architectures this behaves the same as `pack_untilize_init`. See `pack_untilize_init`
 * for the full description and parameter list.
 *
 * Return value: None
 *
 * | Param Type | Name         | Description                       | Type     | Valid Range               | Required            |
 * |------------|--------------|-----------------------------------|----------|---------------------------|---------------------|
 * | Template   | block_ct_dim | Width of a single block in tiles  | uint32_t | 1 to max (see note)       | False (default = 8) |
 * | Template   | full_ct_dim  | Width of a full input in tiles    | uint32_t | Divisible by block_ct_dim | False               |
 * | Function   | icb          | Input circular buffer identifier  | uint32_t | 0 to 31                   | True                |
 * | Function   | ocb          | Output circular buffer identifier | uint32_t | 0 to 31                   | True                |
 */
// clang-format on
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
ALWI void pack_untilize_init_skip_remap(uint32_t icb, uint32_t ocb, uint32_t call_line = __builtin_LINE()) {
    LLK_SAN_FUNCTION();
    pack_untilize_detail::pack_untilize_init_impl<block_ct_dim, full_ct_dim, false>(icb, ocb, call_line);
}

// clang-format off
/**
 * Performs the untilize operation on a block of tiles. Loops over the provided block size. The block is characterized
 * by its width in tiles (block_ct_dim) and height in tiles (block_rt_dim). The width of the block has to be the same
 * as the one provided during the initialization of the pack untilize operation (`pack_untilize_init`). It is not
 * needed to provide the height of the block during the initialization, since `pack_untilize_block` will loop over the
 * height of the block. Note that the maximum size of the block is limited by the size of the DEST and synchronization
 * mode used. These are maximum sizes:
 * - half-sync mode (16-bit mode): 8 tiles
 * - half-sync mode (32-bit mode): 4 tiles
 * - full-sync mode (16-bit mode): 16 tiles
 * - full-sync mode (32-bit mode): 8 tiles
 *
 * Return value: None
 *
 * | Param Type | Name         | Description                                 | Type      | Valid Range               | Required            |
 * |------------|--------------|---------------------------------------------|-----------|---------------------------|---------------------|
 * | Template   | block_ct_dim | Width of a single block in tiles            | uint32_t  | 1 to max (see note)       | False (default = 8) |
 * | Template   | full_ct_dim  | Width of a full input in tiles              | uint32_t  | Divisible by block_ct_dim | False               |
 * | Function   | icb          | Input circular buffer identifier            | uint32_t  | 0 to 31                   | True                |
 * | Function   | block_rt_dim | Height of a single block in tiles           | uint32_t  | >= 1                      | True                |
 * | Function   | ocb          | Output circular buffer identifier           | uint32_t  | 0 to 31                   | True                |
 * | Function   | block_c_index | Index of the currently processed block     | uint32_t  | >= 0                      | False               |
 */
// clang-format on
template <uint32_t block_ct_dim = 8, uint32_t full_ct_dim = block_ct_dim>
ALWI void pack_untilize_block(uint32_t icb, uint32_t block_rt_dim, uint32_t ocb, uint32_t block_c_index = 0) {
    LLK_SAN_FUNCTION();
    for (uint32_t r = 0; r < block_rt_dim; ++r) {
        MATH((llk_math_wait_for_dest_available()));
        for (uint32_t c = 0; c < block_ct_dim; ++c) {
#ifndef ARCH_QUASAR
            UNPACK((llk_unpack_A<
                    BroadcastType::NONE,
                    false /*acc_to_dest*/,
                    EltwiseBinaryReuseDestType::NONE,
                    UnpackToDestEn>(icb, c)));
            MATH((
                llk_math_eltwise_unary_datacopy<DataCopyType::A2D, DST_ACCUM_MODE, BroadcastType::NONE, UnpackToDestEn>(
                    c, icb)));
#else
            UNPACK((llk_unpack_A<
                    BroadcastType::NONE,
                    false /*acc_to_dest*/,
                    EltwiseBinaryReuseDestType::NONE,
                    false /*unpack_to_dest*/>(icb, c)));
            MATH((llk_math_eltwise_unary_datacopy(c, icb)));
#endif
        }
        MATH((llk_math_dest_section_done<DST_ACCUM_MODE>()));
        PACK((llk_packer_wait_for_math_done()));
        PACK((llk_pack_untilize<block_ct_dim, full_ct_dim>(1 /*num_blocks*/, ocb, block_c_index)));
        PACK((llk_pack_dest_section_done<DST_ACCUM_MODE>()));
    }
}

// clang-format off
/**
 * Performs the pack untilize operation when PACK input is already in DEST register. In order to properly initialize the operation,
 * a call to `pack_untilize_dest_init` must be made before this function. The width of the block has to be the same
 * as the one provided during the initialization of the pack untilize operation (`pack_untilize_dest_init`). In order for this
 * untilization to be performed correctly, some other function must place the tiles in the DEST register, e.g. `reduce_tile`,
 * `copy_tile`, etc. Similarly as `pack_untilize_block`, this function operates on a whole block that needs to be untilized.
 * Note that the maximum size of the block is limited by the size of the DEST
 * and synchronization mode used. These are maximum sizes:
 * - half-sync mode (16-bit mode): 8 tiles
 * - half-sync mode (32-bit mode): 4 tiles
 * - full-sync mode (16-bit mode): 16 tiles
 * - full-sync mode (32-bit mode): 8 tiles
 *
 * Return value: None
 *
 * | Param Type | Name               | Description                                                                  | Type      | Valid Range                             | Required              |
 * |------------|--------------------|------------------------------------------------------------------------------|-----------|-----------------------------------------|-----------------------|
 * | Template   | block_ct_dim       | Width of a single block in tiles                                             | uint32_t  | 1 to max (see note)                     | False (default = 8)   |
 * | Template   | full_ct_dim        | Width of a full input in tiles                                               | uint32_t  | Divisible by block_ct_dim               | False                 |
 * | Template   | diagonal           | Whether to use diagonal packing                                              | bool      | true/false                              | False                 |
 * | Template   | narrow_row         | Whether the provided input is narrow                                         | bool      | true/false                              | False                 |
 * | Template   | row_num_datums     | Number of datums per row                                                     | uint32_t  | >= 1                                    | False                 |
 * | Template   | tile_dst_ct_offset | Compile time offset for the index of the tile in the dest from which to pack | uint32_t  | 0 to 7 (0 to 3 if fp32 dest is enabled) | False (default=0)     |
 * | Template   | dense              | Packs two 2 face tiles in a single 4 face region                             | bool      | true/false                              | False (default false) |
 * | Function   | ocb                | Output circular buffer identifier                                            | uint32_t  | 0 to 31                                 | True                  |
 * | Function   | block_rt_dim       | Height of a single block in tiles                                            | uint32_t  | >= 1                                    | False (default=1)     |
 * | Function   | block_c_index      | Block column index (used when full_ct_dim > block_ct_dim)                    | uint32_t  | >= 0                                    | False (default=0)     |
 * | Function   | tile_dst_offset    | Runtime offset for the index of the tile in the dest from which to pack      | uint32_t  | 0 to 7 (0 to 3 if fp32 dest is enabled) | False (default=0)     |
 *
 * NOTE: Face geometry (face_r_dim, num_faces) is derived from the output circular buffer metadata configured on
 * the host via CircularBufferConfig::set_unpack_face_geometry / CBFormatDescriptor::face_geometry.
 */
// clang-format on
template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    uint32_t tile_dst_ct_offset = 0,
    bool dense = false>
ALWI void pack_untilize_dest(
    uint32_t ocb,
    uint32_t block_rt_dim = 1,
    uint32_t block_c_index = 0 /* used when full_ct_dim > block_ct_dim*/,
    uint32_t tile_dst_rt_offset = 0) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    PACK((llk_pack_untilize<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums, tile_dst_ct_offset, dense>(
        block_rt_dim, ocb, block_c_index, tile_dst_rt_offset)));
#else
    PACK((llk_pack_untilize<block_ct_dim, full_ct_dim>(block_rt_dim, ocb, block_c_index, tile_dst_rt_offset)));
#endif
}

// clang-format off
/**
 * @deprecated Face geometry (face_r_dim, num_faces) is now derived from the output circular buffer metadata
 * configured on the host via CircularBufferConfig::set_unpack_face_geometry / CBFormatDescriptor::face_geometry.
 * Use the `pack_untilize_dest(ocb, block_rt_dim, block_c_index, tile_dst_rt_offset)` overload instead. This
 * explicit-face-geometry overload is retained only for backwards compatibility and will be removed.
 *
 * Return value: None
 *
 * | Param Type | Name               | Description                                                                  | Type      | Valid Range                             | Required              |
 * |------------|--------------------|------------------------------------------------------------------------------|-----------|-----------------------------------------|-----------------------|
 * | Template   | block_ct_dim       | Width of a single block in tiles                                             | uint32_t  | 1 to max (see note)                     | False (default = 8)   |
 * | Template   | full_ct_dim        | Width of a full input in tiles                                               | uint32_t  | Divisible by block_ct_dim               | False                 |
 * | Template   | diagonal           | Whether to use diagonal packing                                              | bool      | true/false                              | False                 |
 * | Template   | narrow_row         | Whether the provided input is narrow                                         | bool      | true/false                              | False                 |
 * | Template   | row_num_datums     | Number of datums per row                                                     | uint32_t  | >= 1                                    | False                 |
 * | Template   | tile_dst_ct_offset | Compile time offset for the index of the tile in the dest from which to pack | uint32_t  | 0 to 7 (0 to 3 if fp32 dest is enabled) | False (default=0)     |
 * | Template   | dense              | Packs two 2 face tiles in a single 4 face region                             | bool      | true/false                              | False (default false) |
 * | Function   | ocb                | Output circular buffer identifier                                            | uint32_t  | 0 to 31                                 | True                  |
 * | Function   | block_rt_dim       | Height of a single block in tiles                                            | uint32_t  | >= 1                                    | True                  |
 * | Function   | block_c_index      | Block column index (used when full_ct_dim > block_ct_dim)                    | uint32_t  | >= 0                                    | True                  |
 * | Function   | face_r_dim         | Face height in rows                                                          | uint32_t  | 1, 8 or 16                              | True                  |
 * | Function   | num_faces          | Number of faces                                                              | uint32_t  | 1, 2 or 4                               | True                  |
 * | Function   | tile_dst_rt_offset | Runtime offset for the index of the tile in the dest from which to pack      | uint32_t  | 0 to 7 (0 to 3 if fp32 dest is enabled) | True                  |
 */
// clang-format on
template <
    uint32_t block_ct_dim = 8,
    uint32_t full_ct_dim = block_ct_dim,
    bool diagonal = false,
    bool narrow_row = false,
    std::uint32_t row_num_datums = TILE_C_DIM,
    uint32_t tile_dst_ct_offset = 0,
    bool dense = false>
[[deprecated(
    "Face geometry is now derived from the output CB metadata; use the "
    "pack_untilize_dest(ocb, block_rt_dim, block_c_index, tile_dst_rt_offset) overload instead.")]] ALWI void
pack_untilize_dest(
    uint32_t ocb,
    uint32_t block_rt_dim,
    uint32_t block_c_index,
    [[maybe_unused]] uint32_t face_r_dim,
    [[maybe_unused]] uint32_t num_faces,
    uint32_t tile_dst_rt_offset) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    PACK((llk_pack_untilize<block_ct_dim, full_ct_dim, diagonal, narrow_row, row_num_datums, tile_dst_ct_offset, dense>(
        block_rt_dim, ocb, face_r_dim, num_faces, block_c_index, tile_dst_rt_offset)));
#pragma GCC diagnostic pop
#else
    PACK((llk_pack_untilize<block_ct_dim, full_ct_dim>(block_rt_dim, ocb, block_c_index, tile_dst_rt_offset)));
#endif
}

// clang-format off
/**
 * Uninitializes the pack untilize operation, allowing another operations to be initialized. Needs to be
 * called after the last call to `pack_untilize_dest` or `pack_untilize_block`, before initializing
 * another operation.
 *
 * NOTE: This function is not in line with our programming model, and will be removed by the end of 2025
 * as a part of tt-metal#22904.
 *
 * Return value: None
 *
 * | Param Type | Name | Description                        | Type     | Valid Range | Required |
 * |------------|------|------------------------------------|----------|-------------|----------|
 * | Function   | ocb  | Output circular buffer identifier  | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void pack_untilize_uninit(uint32_t ocb) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR

#ifdef ARCH_BLACKHOLE
    PACK((llk_pack_untilize_uninit(ocb)));
#endif

    // Reconfigure data format to match the initial configuration, before calling init.
    // Init is called to ensure special untilize init overrides are cleaned up.
    {
        LLK_SAN_SILENT_ZONE();
        PACK((llk_init_packer_dest_offset_registers<PackMode::Default>()));
        PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(ocb)));
        PACK((llk_pack_init(ocb)));
    }

#else
    // No-op: Quasar uses dedicated instructions (PACR_UNTILIZE, PACR_STRIDE) that
    // don't conflict with standard PACR paths, so no reconfiguration is needed.
#endif
}

}  // namespace ckernel
