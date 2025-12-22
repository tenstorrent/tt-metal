// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_untilize.h"

/*************************************************************************
 * LLK UNPACK UNTILIZE
 *************************************************************************/
template <bool is_fp32_dest_acc_en>
inline void llk_unpack_untilize_hw_configure(const llk_unpack_A_params_t* unpack_untilize_params) {
    constexpr bool is_row_pool = false;
    constexpr bool within_face_16x16_transpose = false;
    constexpr StochRndType stoch_rnd_mode = StochRndType::None;

    const uint32_t unpA_operand_id = get_operand_id(unpack_untilize_params->unpA_operand);
    const uint32_t unpA_num_faces = 4;
    const uint32_t unpA_face_r_dim = FACE_R_DIM;

    _llk_unpack_untilize_hw_configure_<is_fp32_dest_acc_en, stoch_rnd_mode>(
        unpack_src_format[unpA_operand_id],
        unpack_dst_format[unpA_operand_id],
        unpA_face_r_dim,
        within_face_16x16_transpose,
        unpA_num_faces);
}

template <bool is_fp32_dest_acc_en>
inline void llk_unpack_untilize_hw_configure_disaggregated(const std::uint32_t unpA_operand) {
    const llk_unpack_A_params_t unpack_untilize_params = {
        .unpA_operand = unpA_operand,
    };
    llk_unpack_untilize_hw_configure<is_fp32_dest_acc_en>(&unpack_untilize_params);
}

inline void llk_unpack_untilize_mop_config() { _llk_unpack_untilize_mop_config_(); }

inline void llk_unpack_untilize_init(std::uint32_t operand = 0) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t face_r_dim = 1;

    _llk_unpack_untilize_init_(
        unpack_dst_format[operand_id], get_local_cb_interface(operand_id).fifo_page_size, face_r_dim);
}

// TODO NC: Delete unused parameters when #33825 is finished
inline void llk_unpack_untilize_uninit(const std::uint32_t operand, const std::uint32_t face_r_dim = FACE_R_DIM) {
    const std::uint32_t operand_id = get_operand_id(operand);
    _llk_unpack_untilize_uninit_(unpack_dst_format[operand_id], face_r_dim);
}

template <bool first_pass = true>
inline void llk_unpack_untilize_pass(std::uint32_t operand, std::uint32_t block_tile_cols) {
    const std::uint32_t operand_id = get_operand_id(operand);
    const std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;

    _llk_unpack_untilize_pass_<first_pass>(base_address, block_tile_cols);
}

inline void llk_unpack_untilize(std::uint32_t operand, std::uint32_t block_c_tiles) {
    WAYPOINT("UPUW");
    llk_unpack_untilize_pass<true>(operand, block_c_tiles);
    llk_unpack_untilize_pass<false>(operand, block_c_tiles);
    WAYPOINT("UPUD");
}
