// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "common_globals.h"
#include "sanitizer/api.h"

#ifdef TRISC_PACK
#include "llk_pack_common_api.h"
#include "llk_pack_tile_api.h"
#endif

namespace ckernel {

/**
 * Helper function to reconfigure srca and srcb data formats.
 *
 * NOTE(ARCH_QUASAR): On Quasar, buffer descriptors are programmed into the unpack MOP at op init.
 * reconfig_data_format only reprograms THCON data formats (gasket), not the MOP. When operands or
 * DFB/buffer descriptors change, call the op init again for the new
 * operand pair before the next unpack operation.
 */
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format(const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    // If is_tile_dim_reconfig_en is enabled, modify the dimension and stride according to enum; else, ignore them
    UNPACK((llk_unpack_reconfig_data_format<
            DST_ACCUM_MODE,
            is_tile_dim_reconfig_en ? p_dim_stride_target::FACE_ROW_MAJOR : p_dim_stride_target::IGNORE,
            to_from_int8>(srca_new_operand, srcb_new_operand)));
    MATH((llk_math_reconfig_data_format<DST_ACCUM_MODE, to_from_int8>(srca_new_operand, srcb_new_operand)));
}

/**
 * Helper function to reconfigure srca/srcb data formats, only if they differ from existing formats.
 *
 * NOTE(ARCH_QUASAR): See reconfig_data_format(srca_new_operand, srcb_new_operand).
 */
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format(
    const std::uint32_t srca_old_operand,
    const std::uint32_t srca_new_operand,
    const std::uint32_t srcb_old_operand,
    const std::uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    // If is_tile_dim_reconfig_en is enabled, modify the dimension and stride according to enum; else, ignore them
    UNPACK((llk_unpack_reconfig_data_format<
            DST_ACCUM_MODE,
            is_tile_dim_reconfig_en ? p_dim_stride_target::FACE_ROW_MAJOR : p_dim_stride_target::IGNORE,
            to_from_int8>(srca_old_operand, srca_new_operand, srcb_old_operand, srcb_new_operand)));
    MATH((llk_math_reconfig_data_format<DST_ACCUM_MODE, to_from_int8>(
        srca_old_operand, srca_new_operand, srcb_old_operand, srcb_new_operand)));
}

/**
 * Helper function to reconfigure srca data format.
 */
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    // If is_tile_dim_reconfig_en is enabled, modify the dimension and stride according to enum; else, ignore them
    UNPACK((llk_unpack_reconfig_data_format_srca<
            DST_ACCUM_MODE,
            is_tile_dim_reconfig_en ? p_dim_stride_target::FACE_ROW_MAJOR : p_dim_stride_target::IGNORE,
            to_from_int8>(srca_new_operand)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE, to_from_int8>(srca_new_operand)));
}

/**
 * Helper function to reconfigure srca input data format, only if it differs from existing format.
 */
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srca(const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    LLK_SAN_FUNCTION();
    // If is_tile_dim_reconfig_en is enabled, modify the dimension and stride according to enum; else, ignore them
    UNPACK((llk_unpack_reconfig_data_format_srca<
            DST_ACCUM_MODE,
            is_tile_dim_reconfig_en ? p_dim_stride_target::FACE_ROW_MAJOR : p_dim_stride_target::IGNORE,
            to_from_int8>(srca_old_operand, srca_new_operand)));
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE, to_from_int8>(srca_old_operand, srca_new_operand)));
}

/**
 * Helper function to reconfigure srcb input data format.
 */
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    // If is_tile_dim_reconfig_en is enabled, modify the dimension and stride according to enum; else, ignore them
    UNPACK((llk_unpack_reconfig_data_format_srcb<
            DST_ACCUM_MODE,
            is_tile_dim_reconfig_en ? p_dim_stride_target::FACE_ROW_MAJOR : p_dim_stride_target::IGNORE,
            to_from_int8>(srcb_new_operand)));
    MATH((llk_math_reconfig_data_format_srcb<DST_ACCUM_MODE, to_from_int8>(srcb_new_operand)));
}

/**
 * Helper function to reconfigure srcb input data format, only if it differs from existing format.
 */
template <bool to_from_int8 = false, bool is_tile_dim_reconfig_en = false>
ALWI void reconfig_data_format_srcb(const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    LLK_SAN_FUNCTION();
    // If is_tile_dim_reconfig_en is enabled, modify the dimension and stride according to enum; else, ignore them
    UNPACK((llk_unpack_reconfig_data_format_srcb<
            DST_ACCUM_MODE,
            is_tile_dim_reconfig_en ? p_dim_stride_target::FACE_ROW_MAJOR : p_dim_stride_target::IGNORE,
            to_from_int8>(srcb_old_operand, srcb_new_operand)));
    MATH((llk_math_reconfig_data_format_srcb<DST_ACCUM_MODE, to_from_int8>(srcb_old_operand, srcb_new_operand)));
}

// clang-format off
/**
 * Re-programs the per-source unpacker tile descriptor and TILE_SIZE_A/TILE_SIZE_B GPRs from the given
 * input CBs in matmul's reversed operand order (in0 -> SrcB, in1 -> SrcA), WITHOUT changing data formats.
 * Call it immediately before matmul_init / matmul_block_init when a preceding op left SrcA/SrcB with
 * asymmetric per-source tile sizes (e.g. a binary op whose two operands had different data formats) but
 * the following matmul reuses those same formats: matmul_init only builds the MOP and does not re-program
 * the tile descriptor, and reconfig_data_format is semantically wrong here because the data formats did
 * not change. Callers pass the operands in natural order (in0_cb_id, in1_cb_id); the reversed SrcA/SrcB
 * assignment is handled inside this helper (SrcA tile size <- in1, SrcB tile size <- in0), matching the
 * GPR-fed base-address auto-increment the matmul MOP performs (TILE_SIZE_A drives the SrcA stream,
 * TILE_SIZE_B drives the SrcB stream). The tile-descriptor format *value* is left unchanged, but this
 * is not a zero-footprint call for a reconfig/hazard audit: it runs the full
 * llk_unpack/llk_math_reconfig_data_format_src* paths, so the unpack THCON src/dst format fields are
 * rewritten (to their existing values), each unpack call issues TTI_STALLWAIT(STALL_CFG, UNPACK*), and
 * on WH the MATH pipe additionally RMWs ALU_FORMAT_SPEC_REG0/1_Src* and drains via
 * TTI_STALLWAIT(STALL_CFG, MATH | WAIT_SFPU).
 *
 * NOTE(ARCH_QUASAR): On Quasar, buffer descriptors are programmed into the unpack MOP at op init, so this
 * helper is a no-op; re-call matmul_init(in0_cb_id, in1_cb_id) for the new operand pair instead.
 *
 * Return value: None
 *
 * | Param Type | Name      | Description                                   | Type     | Valid Range | Required |
 * |------------|-----------|-----------------------------------------------|----------|-------------|----------|
 * | Function   | in0_cb_id | First matmul input CB (maps to SrcB)          | uint32_t | 0 to 31     | True     |
 * | Function   | in1_cb_id | Second matmul input CB (maps to SrcA)         | uint32_t | 0 to 31     | True     |
 */
// clang-format on
ALWI void reconfig_matmul_tile_dims(const std::uint32_t in0_cb_id, const std::uint32_t in1_cb_id) {
    LLK_SAN_FUNCTION();
#ifndef ARCH_QUASAR
    // Matmul reverses the operand -> source mapping (in0 -> SrcB, in1 -> SrcA), so the SrcA tile
    // descriptor / TILE_SIZE_A must come from in1 and the SrcB tile descriptor / TILE_SIZE_B from in0.
    // On the two UNPACK calls, FACE_ROW_MAJOR selects the tile-dim reconfig path, which re-asserts
    // TILE_SIZE plus the tile x_dim, Z-stride and Z-dim/num_faces descriptor fields; to_from_int8 is
    // false because the data formats are unchanged.
    UNPACK(
        (llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, p_dim_stride_target::FACE_ROW_MAJOR, false>(in1_cb_id)));
    UNPACK(
        (llk_unpack_reconfig_data_format_srcb<DST_ACCUM_MODE, p_dim_stride_target::FACE_ROW_MAJOR, false>(in0_cb_id)));
    // The MATH calls take <is_fp32_dest_acc_en, to_from_int8> only -- there is no tile-dim path here;
    // with to_from_int8 false they re-assert the (unchanged) SrcA/SrcB math format state.
    MATH((llk_math_reconfig_data_format_srca<DST_ACCUM_MODE, false>(in1_cb_id)));
    MATH((llk_math_reconfig_data_format_srcb<DST_ACCUM_MODE, false>(in0_cb_id)));
#endif
}

// clang-format off
/**
 * Reconfigures the packer output data format by specifying the CB ID of the new operand. This function
 * call will always perform the reconfiguration, regardless of the data format of the old operand.
 * If the new CB ID is the same as the current one, reconfiguration will still occur.
 *
 * NOTE(ARCH_QUASAR): On Quasar, buffer descriptors are programmed at op init. pack_reconfig_data_format
 * only reprograms THCON IN_DATA_FORMAT (gasket), not the MOP or buffer descriptors. When the pack output
 * operand changes, call pack_init(new_cb_id) before pack_tile.
 *
 * NOTE: Packer reconfiguration functions are used similarly to the initialization function, in a sense
 * that they are called before the call to the packer function that uses the new configuration. It is
 * recommended to call this function right after other op-specific initialization functions.
 *
 * Return value: None
 *
 * | Param Type | Name                    | Description                   | Type     | Valid Range | Required |
 * |------------|-------------------------|-------------------------------|----------|-------------|----------|
 * | Template   | is_tile_dim_reconfig_en | Toggle tile reconfiguration   | bool     | true/false  | False    |
 * | Function   | new_cb_id               | New data format operand value | uint32_t | Any         | True     |
 */
// clang-format on
template <bool is_tile_dim_reconfig_en = false>
ALWI void pack_reconfig_data_format(const std::uint32_t new_cb_id) {
    LLK_SAN_FUNCTION();
#ifdef ARCH_QUASAR
    static_assert(
        !is_tile_dim_reconfig_en,
        "Quasar pack reconfig does not support tile-dimension changes; call pack_init instead");
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(new_cb_id)));
#else
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(new_cb_id)));
    if constexpr (is_tile_dim_reconfig_en) {
        PACK((llk_pack_init<PackMode::Default, false /* zero_output */, true /* skip_addrmod_config */>(new_cb_id)));
    }
#endif
}

// clang-format off
/**
 * Reconfigures the packer output data format by specifying the CB IDs of the old and new operands.
 * This function internally calls the reconfiguration function with the new CB ID, but before it does so,
 * it checks if the old and new data formats are different. If they are the same, it does not perform
 * the reconfiguration. This function is useful when you want to ensure that the packer only reconfigures
 * when different data format is wanted, avoiding unnecessary reconfiguration overhead.
 *
 * NOTE(ARCH_QUASAR): See pack_reconfig_data_format(new_cb_id). Call pack_init(new_cb_id) before pack_tile
 * when switching pack output operand.
 *
 * NOTE: Packer reconfiguration functions are used similarly to the initialization function, in a sense
 * that they are called before the call to the packer function that uses the new configuration. It is
 * recommended to call this function right after other op-specific initialization functions.
 *
 * Return value: None
 *
 * | Param Type | Name                    | Description                        | Type     | Valid Range | Required |
 * |------------|-------------------------|------------------------------------|----------|-------------|----------|
 * | Template   | is_tile_dim_reconfig_en | Toggle tile reconfiguration        | bool     | true/false  | False    |
 * | Function   | old_cb_id               | Previous data format operand value | uint32_t | Any         | True     |
 * | Function   | new_cb_id               | New data format operand value      | uint32_t | Any         | True     |
 */
// clang-format on
template <bool is_tile_dim_reconfig_en = false>
ALWI void pack_reconfig_data_format(const std::uint32_t old_cb_id, const std::uint32_t new_cb_id) {
    LLK_SAN_FUNCTION();
#ifdef ARCH_QUASAR
    static_assert(
        !is_tile_dim_reconfig_en,
        "Quasar pack reconfig does not support tile-dimension changes; call pack_init instead");
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(old_cb_id, new_cb_id)));
#else
    PACK((llk_pack_reconfig_data_format<DST_ACCUM_MODE>(old_cb_id, new_cb_id)));
    if constexpr (is_tile_dim_reconfig_en) {
        PACK((llk_pack_init<PackMode::Default, false /* zero_output */, true /* skip_addrmod_config */>(new_cb_id)));
    }
#endif
}

}  // namespace ckernel
