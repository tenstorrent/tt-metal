/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "fw_debug.h"
#include "cunpack_common.h"
#include "llk_param_structs.h"
#include "llk_io_unpack.h"

#ifdef PERF_DUMP
#include "ckernel_perf_api.h"
#endif

#include "hostdevcommon/common_runtime_address_map.h"


using namespace ckernel;
using namespace ckernel::unpacker;

inline void llk_zero_operand(std::uint32_t operand) {
    std::uint32_t input = get_operand_id(operand);

    TT_SETDMAREG(0, 0, 0, LO_16(p_gpr_unpack::OPERAND_OFFSET_ADDR));
    TT_SETDMAREG(0, 0, 0, HI_16(p_gpr_unpack::OPERAND_OFFSET_ADDR));

    std::uint32_t fifo_base_addr = (cb_interface[input].fifo_limit + 1) - cb_interface[input].fifo_size;
    TT_SETDMAREG(0, fifo_base_addr, 0, LO_16(p_gpr_unpack::p_gpr_unpack::OPERAND_BASE_ADDR));

    for (std::uint32_t i = 0; i < cb_interface[input].fifo_size; i++) {
        TTI_STOREIND(
            1,
            0,
            p_ind::LD_16B,
            LO_16(p_gpr_unpack::OPERAND_OFFSET_ADDR),
            p_ind::INC_16B,
            p_gpr_unpack::ZERO_0,
            p_gpr_unpack::OPERAND_BASE_ADDR);
    }
}

inline void llk_unpack_debug_dump(std::uint8_t *data, std::uint32_t byte_size) {
    debug_dump(data, byte_size);
}

inline void llk_unpack_reconfig_data_format_srca_impl(const std::uint32_t srca_operand_id) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);

    uint32_t alu_config_data = gl_alu_format_spec_reg;

    gl_alu_format_spec_reg = cfg_rmw_mmio_rd_tensix_wr(
        ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32,
        ALU_FORMAT_SPEC_REG0_SrcA_SHAMT,
        ALU_FORMAT_SPEC_REG0_SrcA_MASK,
        unpack_dst_format[srca_operand_id],
        alu_config_data);

    reconfig_unpacker_data_format(
        srca_operand_id,
        THCON_SEC0_REG0_TileDescriptor_ADDR32,
        THCON_SEC0_REG2_Out_data_format_ADDR32,
        UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
}

inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);

    if((unpack_src_format[old_srca_operand_id] != unpack_src_format[new_srca_operand_id])) {
        llk_unpack_reconfig_data_format_srca_impl(new_srca_operand_id);
    }
}

inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    llk_unpack_reconfig_data_format_srca_impl(get_operand_id(srca_new_operand));
}

inline void llk_unpack_reconfig_data_format_srcb_impl(std::uint32_t srcb_operand_id) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK1);

    uint32_t alu_config_data = gl_alu_format_spec_reg;

    gl_alu_format_spec_reg = cfg_rmw_mmio_rd_tensix_wr(
        ALU_FORMAT_SPEC_REG_SrcB_val_ADDR32,
        ALU_FORMAT_SPEC_REG1_SrcB_SHAMT,
        ALU_FORMAT_SPEC_REG1_SrcB_MASK,
        unpack_dst_format[srcb_operand_id],
        alu_config_data);

    reconfig_unpacker_data_format(
        srcb_operand_id,
        THCON_SEC1_REG0_TileDescriptor_ADDR32,
        THCON_SEC1_REG2_Out_data_format_ADDR32,
        UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
}

inline void llk_unpack_reconfig_data_format_srcb(const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if((unpack_src_format[old_srcb_operand_id] != unpack_src_format[new_srcb_operand_id])) {
        llk_unpack_reconfig_data_format_srcb_impl(new_srcb_operand_id);
    }
}

inline void llk_unpack_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srcb_impl(get_operand_id(srcb_new_operand));
}

ALWI void llk_unpack_reconfig_data_format_impl(std::uint32_t srca_operand_id, std::uint32_t srcb_operand_id) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);

    uint alu_src_format = (unpack_dst_format[srcb_operand_id] << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT) |
                          (unpack_dst_format[srca_operand_id] << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT);
    uint alu_src_mask = ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_FORMAT_SPEC_REG1_SrcB_MASK;
    uint32_t alu_config_data = gl_alu_format_spec_reg;

    gl_alu_format_spec_reg = cfg_rmw_mmio_rd_tensix_wr(
        ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, 0, alu_src_mask, alu_src_format, alu_config_data);

    reconfig_unpacker_data_format(
        srca_operand_id,
        THCON_SEC0_REG0_TileDescriptor_ADDR32,
        THCON_SEC0_REG2_Out_data_format_ADDR32,
        UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    reconfig_unpacker_data_format(
        srcb_operand_id,
        THCON_SEC1_REG0_TileDescriptor_ADDR32,
        THCON_SEC1_REG2_Out_data_format_ADDR32,
        UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
}

inline void llk_unpack_reconfig_data_format(
    const std::uint32_t srca_old_operand,
    const std::uint32_t srca_new_operand,
    const std::uint32_t srcb_old_operand,
    const std::uint32_t srcb_new_operand) {
    std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);
    std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if ((unpack_src_format[old_srca_operand_id] != unpack_src_format[new_srca_operand_id]) &&
        (unpack_src_format[old_srcb_operand_id] != unpack_src_format[new_srcb_operand_id])) {
        llk_unpack_reconfig_data_format_impl(new_srca_operand_id, new_srcb_operand_id);
    } else if ((unpack_src_format[old_srca_operand_id] != unpack_src_format[new_srca_operand_id])) {
        llk_unpack_reconfig_data_format_srca_impl(new_srca_operand_id);
    } else if ((unpack_src_format[old_srcb_operand_id] != unpack_src_format[new_srcb_operand_id])) {
        llk_unpack_reconfig_data_format_srcb_impl(new_srcb_operand_id);
    }
}

ALWI void llk_unpack_reconfig_data_format(const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_impl(get_operand_id(srca_new_operand), get_operand_id(srcb_new_operand));
}

inline void llk_unpack_dbg_feature_disable(){
     //TBD
}
