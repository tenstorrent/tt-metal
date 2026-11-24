// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include "llk_math_common_api.h"
#include "experimental/llk_math_sdpa_bcast_col_srcb_reuse.h"
#include "sanitizer/api.h"

/*************************************************************************
 * LLK ELTWISE BINARY
 *************************************************************************/

/**
 * @brief Configure the math thread for the SrcB-reuse column-broadcast op from the operand CBs.
 *
 * @tparam eltwise_binary_type: Type of eltwise binary op, values = <ELWADD/ELWSUB/ELWMUL>
 * @tparam num_tiles: Number of dest tiles processed per execute call.
 * @tparam math_fidelity: Math fidelity of the ELWMUL path.
 * @tparam dense: Place two-face tiles back to back in dest, values = <true/false>
 * @param operand_A: CB whose face count configures the MOP; both operands must have the same number of faces.
 * @param operand_B: CB of the second operand, unused on this thread.
 * @note Call this before @ref llk_math_sdpa_bcast_col_srcb_reuse_preamble and @ref llk_math_sdpa_bcast_col_srcb_reuse,
 *       and run @ref llk_math_sdpa_bcast_col_srcb_reuse_postamble after.
 * @note On the unpack thread, pair with @ref llk_unpack_A_sdpa_init and issue
 *       @ref llk_unpack_A_sdpa_set_srcb_dummy_valid before the operand unpacks.
 * @note The MOP accumulates into DEST for every op, so the output tiles must be zero on entry; see
 *       @ref _llk_math_sdpa_bcast_col_srcb_reuse_init_ for what clear_dest on the preamble zeroes.
 */
template <
    EltwiseBinaryType eltwise_binary_type,
    std::uint32_t num_tiles,
    MathFidelity math_fidelity,
    bool dense = false>
inline void llk_math_sdpa_bcast_col_srcb_reuse_init_with_operands(
    const std::uint32_t operand_A, [[maybe_unused]] const std::uint32_t operand_B) {
    SAN_HOOK(unsupported());
    const std::uint32_t operand_id = get_operand_id(operand_A);  // both operands must have same number of faces
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_sdpa_bcast_col_srcb_reuse_init_<eltwise_binary_type, num_tiles, math_fidelity, dense>(num_faces);
}

template <DstSync DST, bool IS_FP32_MATH_DEST_EN, bool clear_dest = false>
inline void llk_math_sdpa_bcast_col_srcb_reuse_preamble() {
    SAN_HOOK(unsupported());
    _llk_math_sdpa_bcast_col_srcb_reuse_preamble_<DST, IS_FP32_MATH_DEST_EN, clear_dest>();
}

inline void llk_math_sdpa_bcast_col_srcb_reuse_postamble() {
    SAN_HOOK(unsupported());
    _llk_math_sdpa_bcast_col_srcb_reuse_postamble_();
}

template <
    EltwiseBinaryType eltwise_binary_type,
    std::uint32_t num_tiles,
    bool is_fp32_dest_acc_en,
    MathFidelity math_fidelity>
inline void llk_math_sdpa_bcast_col_srcb_reuse(const std::uint32_t dst_index) {
    SAN_HOOK(unsupported());
    _llk_math_sdpa_bcast_col_srcb_reuse_<
        eltwise_binary_type,
        num_tiles,
        DST_SYNC_MODE,
        is_fp32_dest_acc_en,
        math_fidelity>(dst_index);
}
