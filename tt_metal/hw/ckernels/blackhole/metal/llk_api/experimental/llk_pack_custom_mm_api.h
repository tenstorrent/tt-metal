// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "experimental/llk_pack_custom_mm.h"
#include "llk_pack_common_api.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK PACK CUSTOM_MM
 *
 * The packer tile stride of the custom_mm family's dense layout (output tiles
 * 32 DEST rows apart), sized from the output operand's configured pack-source
 * format. The init and the uninit take the same output operand, and on the
 * dense path both check (LLK_ASSERT) that the packer is configured for it: the
 * stride is only right for the formats the packer actually holds.
 *
 * Uses experimental/llk_pack_custom_mm.h as the low-level implementation.
 *************************************************************************/

/**
 * @brief Program the packer tile stride for the custom_mm family's dense layout (output tiles 32 DEST rows
 *        apart), sized from the output operand's configured pack-source format.
 *
 * @tparam dense_packing: Lay output tiles 32 DEST rows apart instead of 64; no-op when false, values = <true/false>
 * @param output: Output CB. Its pack-source format sizes the stride; with dense_packing the packer must already
 *                be configured for this CB.
 * @note Call after the pack init or reconfig that configured the packer for @p output, with the same
 *       dense_packing as @ref llk_math_custom_mm_init or @ref llk_math_compressed_custom_mm_init on the math
 *       thread. Pair with @ref llk_pack_custom_mm_uninit on the same @p output after the last dense pack.
 */
template <bool dense_packing = false>
inline void llk_pack_custom_mm_init(const std::uint32_t output) {
    SAN_HOOK(unsupported());
    const std::uint32_t output_id = get_output_id(output);
    if constexpr (dense_packing) {
        LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));
    }
    _llk_pack_custom_mm_init_<dense_packing>(pack_src_format[output_id]);
}

/**
 * @brief Restore the packer's default 64-row tile stride after the last dense custom_mm pack, sized from the
 *        output operand's configured pack-source format.
 *
 * @tparam dense_packing: Must match the preceding @ref llk_pack_custom_mm_init; no-op when false, values = <true/false>
 * @param output: The same output CB that was passed to @ref llk_pack_custom_mm_init. With dense_packing the
 *                packer must still be configured for it; a reconfig to another CB in between would make the
 *                restored stride the wrong size.
 * @note Call after the last pack of a dense custom_mm sequence and before the next op's pack init.
 */
template <bool dense_packing = false>
inline void llk_pack_custom_mm_uninit(const std::uint32_t output) {
    SAN_HOOK(unsupported());
    const std::uint32_t output_id = get_output_id(output);
    if constexpr (dense_packing) {
        LLK_ASSERT_BLOCK(are_packers_configured_correctly(pack_src_format[output_id], pack_dst_format[output_id]));
    }
    _llk_pack_custom_mm_uninit_<dense_packing>(pack_src_format[output_id]);
}
