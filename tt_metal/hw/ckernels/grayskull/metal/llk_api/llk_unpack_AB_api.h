// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB
 *************************************************************************/

template <bool is_fp32_dest_acc_en = false /*not used*/, StochRndType stoch_rnd_mode = StochRndType::None /*not used*/>
inline void llk_unpack_AB_hw_configure(
    const llk_unpack_AB_params_t *unpack_AB_params, const int within_face_16x16_transpose = 0 /*not used*/) {
    // In0 -> unpA
    // In1 -> unpB
    const uint32_t unpA_operand_id = get_operand_id(unpack_AB_params->unpA_operand);
    const uint32_t unpB_operand_id = get_operand_id(unpack_AB_params->unpB_operand);

    _llk_unpack_AB_hw_configure_(
        unpack_src_format[unpA_operand_id],
        unpack_src_format[unpB_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpack_dst_format[unpB_operand_id]);
}

template <bool is_fp32_dest_acc_en = false /*not used*/, StochRndType stoch_rnd_mode = StochRndType::None /*not used*/>
inline void llk_unpack_AB_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpB_operand, const int within_face_16x16_transpose = 0) {
    const llk_unpack_AB_params_t unpack_AB_params = {.unpA_operand = unpA_operand, .unpB_operand = unpB_operand};

    llk_unpack_AB_hw_configure<is_fp32_dest_acc_en, stoch_rnd_mode>(&unpack_AB_params, within_face_16x16_transpose);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_mop_config(const bool transpose_of_faces = false, const std::uint32_t operand_id = 0) {
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);
    const bool narrow_tile = get_operand_narrow_tile(operand_id);  // if narrow tile read face 0 twice for row broadcast
                                                                   // or read face 0 and 1 for col broadcast
    _llk_unpack_AB_mop_config_<BType>(transpose_of_faces, num_faces, narrow_tile);
}

//Params only used for WHB0
template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB_init(
    const std::uint32_t operandA /*not used*/,
    const std::uint32_t operandB /*not used*/,
    const std::uint32_t transpose = 0 /*not used*/,
    const std::uint32_t acc_to_dest = 0 /*not used*/) {
    _llk_unpack_AB_init_<BType>(transpose, acc_to_dest);
}

template <BroadcastType BType = BroadcastType::NONE>
inline void llk_unpack_AB(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a,
    const std::uint32_t tile_index_b,
    const bool transpose_of_faces = 0 /*not used*/) {
    std::uint32_t operandA_id = get_operand_id(operandA);
    std::uint32_t operandB_id = get_operand_id(operandB);
    std::uint32_t base_address_a = cb_interface[operandA_id].fifo_rd_ptr - 1;
    std::uint32_t offset_address_a = cb_interface[operandA_id].fifo_page_size * tile_index_a;
    std::uint32_t address_a = base_address_a + offset_address_a;
    std::uint32_t base_address_b = cb_interface[operandB_id].fifo_rd_ptr - 1;
    std::uint32_t offset_address_b = cb_interface[operandB_id].fifo_page_size * tile_index_b;
    std::uint32_t address_b = base_address_b + offset_address_b;

    _llk_unpack_AB_<BType>(address_a, address_b, transpose_of_faces > 0);
}
