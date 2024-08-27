// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common_api.h"

/*************************************************************************
 * LLK UNPACK AB MATMUL
 *************************************************************************/

template <bool is_fp32_dest_acc_en = false /*not used*/, StochRndType stoch_rnd_mode = StochRndType::None /*not used*/>
inline void llk_unpack_AB_matmul_hw_configure(const llk_unpack_AB_matmul_params_t *unpack_AB_params) {
    // In0 -> unpB
    // In1 -> unpA
    const uint32_t unpA_operand_id = get_operand_id(unpack_AB_params->unpB_operand);
    const uint32_t unpB_operand_id = get_operand_id(unpack_AB_params->unpA_operand);

    _llk_unpack_AB_matmul_hw_configure_(
        unpack_src_format[unpA_operand_id],
        unpack_src_format[unpB_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpack_dst_format[unpB_operand_id]);
}

template <bool is_fp32_dest_acc_en = false /*not used*/, StochRndType stoch_rnd_mode = StochRndType::None /*not used*/>
inline void llk_unpack_AB_matmul_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpB_operand, const std::uint32_t transpose_xy_srca = 0) {
    const llk_unpack_AB_matmul_params_t unpack_AB_matmul_params = {
        .unpA_operand = unpA_operand, .unpB_operand = unpB_operand, .transpose_xy_srca = transpose_xy_srca};
    llk_unpack_AB_matmul_hw_configure<is_fp32_dest_acc_en, stoch_rnd_mode>(&unpack_AB_matmul_params);
}

inline void llk_unpack_AB_matmul_mop_config(
    const bool transpose,
    const std::uint32_t ct_dim /*not used*/,
    const std::uint32_t rt_dim /*not used*/,
    const std::uint32_t kt_dim /*not used*/,
    const bool partial_face /*not used*/) {
    // in0 - loaded to SrcB
    // in1 - loaded to SrcA
    _llk_unpack_AB_matmul_mop_config_(transpose);
}

inline void llk_unpack_AB_matmul_init(
    const std::uint32_t operandA /*not used*/,
    const std::uint32_t operandB /*not used*/,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {

    _llk_unpack_AB_matmul_init_(
        transpose,
        ct_dim,
        rt_dim,
        kt_dim);
}

inline void llk_unpack_AB_matmul(
    const std::uint32_t operandA,
    const std::uint32_t operandB,
    const std::uint32_t tile_index_a /*not used*/,
    const std::uint32_t tile_index_b /*not used*/,
    const std::uint32_t ct_dim = 1 /*not used*/,
    const std::uint32_t rt_dim = 1 /*not used*/,
    const std::uint32_t kt_dim = 1 /*not used*/) {
    // In0/InA -> srcB
    // In1/InB -> srcA

    const std::uint32_t operandA_id = get_operand_id(operandA);
    const std::uint32_t operandB_id = get_operand_id(operandB);

    std::uint32_t base_address_a = cb_interface[operandA_id].fifo_rd_ptr - 1;
    std::uint32_t base_address_b = cb_interface[operandB_id].fifo_rd_ptr - 1;
    std::uint32_t unpA_src_format = (std::uint32_t)unpack_src_format[operandA_id];
    std::uint32_t unpB_src_format = (std::uint32_t)unpack_src_format[operandB_id];

    for (std::uint32_t rt=0; rt<rt_dim; rt++) {
        std::uint32_t offset_address_a = MUL_TILE_SIZE_AND_INDEX<true>(unpA_src_format, (tile_index_a + rt*kt_dim));
        std::uint32_t address_a = base_address_a + offset_address_a;

        for (std::uint32_t ct=0; ct<ct_dim; ct++) {

            std::uint32_t offset_address_b = MUL_TILE_SIZE_AND_INDEX<true>(unpB_src_format, (tile_index_b+ct));
            std::uint32_t address_b = base_address_b + offset_address_b;

            WAYPOINT("UPMW");
            _llk_unpack_AB_matmul_(
                address_a,
                address_b
            );
            WAYPOINT("UPMD");
        }
    }
}
