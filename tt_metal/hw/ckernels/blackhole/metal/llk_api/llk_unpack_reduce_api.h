// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_unpack_common_api.h"
#include "llk_unpack_reduce.h"

/*************************************************************************
 * LLK UNPACK REDUCE
 *************************************************************************/

template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, StochRndType stoch_rnd_mode = StochRndType::None>
inline void llk_unpack_reduce_hw_configure(
    const llk_unpack_reduce_params_t* unpack_reduce_params, const float const_mult) {
    constexpr bool within_face_16x16_transpose = (ReduceDim::REDUCE_ROW == dim);

    const std::uint32_t unpA_operand_id = get_operand_id(unpack_reduce_params->unpA_operand);
    const std::uint32_t unpA_num_faces = get_operand_num_faces(unpA_operand_id);
    const std::uint32_t unpA_face_r_dim = get_operand_face_r_dim(unpA_operand_id);

    constexpr std::uint32_t unpB_src_format = (std::uint32_t)DataFormat::Float32;
    const std::uint32_t unpB_dst_format =
        ((std::uint32_t)unpack_dst_format[unpA_operand_id] == (std::uint32_t)DataFormat::Int8)
            ? (std::uint32_t)DataFormat::Float16
            :  // Int8 is treated as fp16_a
            ((((std::uint32_t)unpack_dst_format[unpA_operand_id] >> 2) & 0x1) ? (std::uint32_t)DataFormat::Float16_b
                                                                              : (std::uint32_t)DataFormat::Float16);

    _llk_unpack_reduce_hw_configure_<is_fp32_dest_acc_en, stoch_rnd_mode>(
        unpack_src_format[unpA_operand_id],
        unpB_src_format,
        unpack_dst_format[unpA_operand_id],
        unpB_dst_format,
        unpA_face_r_dim,
        unpA_face_r_dim,
        within_face_16x16_transpose,
        unpA_num_faces,
        unpA_num_faces);
}

template <PoolType type, ReduceDim dim, bool is_fp32_dest_acc_en, StochRndType stoch_rnd_mode = StochRndType::None>
inline void llk_unpack_reduce_hw_configure_disaggregated(const std::uint32_t unpA_operand, const float mult) {
    const llk_unpack_reduce_params_t unpack_reduce_params = {.unpA_operand = unpA_operand};
    llk_unpack_reduce_hw_configure<type, dim, is_fp32_dest_acc_en, stoch_rnd_mode>(&unpack_reduce_params, mult);
}

template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce_mop_config() {
    _llk_unpack_reduce_mop_config_<type, dim>();
}

template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce_init(const std::uint32_t within_face_16x16_transpose = 0) {
    constexpr std::uint32_t unpA_operand_id = 0;

    const std::uint32_t unpB_src_format = (std::uint32_t)DataFormat::Float32;
    const std::uint32_t unpB_dst_format =
        ((std::uint32_t)unpack_dst_format[unpA_operand_id] == (std::uint32_t)DataFormat::Int8)
            ? (std::uint32_t)DataFormat::Float16
            :  // Int8 is treated as fp16_a
            ((((std::uint32_t)unpack_dst_format[unpA_operand_id] >> 2) & 0x1) ? (std::uint32_t)DataFormat::Float16_b
                                                                              : (std::uint32_t)DataFormat::Float16);

    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_RMW>(unpB_dst_format);

    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0xf>(unpB_src_format);
    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(unpB_dst_format);

    TTI_WRCFG(p_gpr_unpack::L1_BUFFER_ADDR, p_cfg::WRCFG_32b, THCON_SEC1_REG3_Base_address_ADDR32);
    TTI_WRCFG(p_gpr_unpack::L1_BUFFER_ADDR, p_cfg::WRCFG_32b, THCON_SEC1_REG3_Base_cntx1_address_ADDR32);
    TTI_NOP;
    TTI_NOP;

    _llk_unpack_reduce_init_<type, dim>(within_face_16x16_transpose);
}

template <PoolType type, ReduceDim dim>
inline void llk_unpack_reduce(const std::uint32_t operand, const std::uint32_t tile_index) {
    std::uint32_t operand_id = get_operand_id(operand);
    std::uint32_t base_address = get_local_cb_interface(operand_id).fifo_rd_ptr - 1;
    std::uint32_t offset_address = get_local_cb_interface(operand_id).fifo_page_size * tile_index;
    std::uint32_t address = base_address + offset_address;

    WAYPOINT("UPRW");
    _llk_unpack_reduce_<type, dim>(address);
    WAYPOINT("UPRD");
}
