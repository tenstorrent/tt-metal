// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "fw_debug.h"
#include "cunpack_common.h"

#ifdef PERF_DUMP
#include "ckernel_perf_api.h"
#endif

using namespace ckernel;
using namespace ckernel::unpacker;

inline void _llk_zero_operand_(const std::uint32_t base_address, const std::uint32_t size) {

    TT_SETDMAREG(0, 0, 0, LO_16(p_gpr_unpack::OPERAND_OFFSET_ADDR));
    TT_SETDMAREG(0, 0, 0, HI_16(p_gpr_unpack::OPERAND_OFFSET_ADDR));

    TT_SETDMAREG(0, base_address, 0, LO_16(p_gpr_unpack::p_gpr_unpack::OPERAND_BASE_ADDR));

    for (std::uint32_t i = 0; i < size; i++) {
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

template <bool mail2math=true, bool mail2pack=true>
inline void _llk_unpack_get_tile_(const std::uint32_t address, std::uint32_t *p_tile) {
    std::uint32_t byte_address = (address)<<4;

    if constexpr (mail2math) {
       mailbox_write(ThreadId::MathThreadId, byte_address);
       semaphore_post(semaphore::UNPACK_OPERAND_SYNC);
    }

    if constexpr (mail2pack) {
       mailbox_write(ThreadId::PackThreadId, byte_address);
       semaphore_post(semaphore::UNPACK_OPERAND_SYNC);
    }

    *p_tile = byte_address;
}

template <bool mail2math=true, bool mail2pack=true>
inline void _llk_unpack_release_tile_() {
    while (semaphore_read(semaphore::UNPACK_OPERAND_SYNC) > 0);
}

inline void _llk_unpack_debug_dump_(std::uint8_t *data, std::uint32_t byte_size) {
    debug_dump(data, byte_size);
}

inline void _llk_unpack_debug_dump_seek_(std::uint8_t offset) {
    debug_dump_seek(offset);
}

inline void _llk_unpack_reconfig_data_format_srca_impl_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);

    uint32_t alu_config_data = gl_alu_format_spec_reg;

    gl_alu_format_spec_reg = cfg_rmw_mmio_rd_tensix_wr(
        ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32,
        ALU_FORMAT_SPEC_REG0_SrcA_SHAMT,
        ALU_FORMAT_SPEC_REG0_SrcA_MASK,
        unpack_dst_format,
        alu_config_data);

    reconfig_unpacker_data_format(
        unpack_src_format,
        unpack_dst_format,
        THCON_SEC0_REG0_TileDescriptor_ADDR32,
        THCON_SEC0_REG2_Out_data_format_ADDR32,
        UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
}

inline void _llk_unpack_reconfig_data_format_srcb_impl_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK1);

    uint32_t alu_config_data = gl_alu_format_spec_reg;

    gl_alu_format_spec_reg = cfg_rmw_mmio_rd_tensix_wr(
        ALU_FORMAT_SPEC_REG_SrcB_val_ADDR32,
        ALU_FORMAT_SPEC_REG1_SrcB_SHAMT,
        ALU_FORMAT_SPEC_REG1_SrcB_MASK,
        unpack_dst_format,
        alu_config_data);

    reconfig_unpacker_data_format(
        unpack_src_format,
        unpack_dst_format,
        THCON_SEC1_REG0_TileDescriptor_ADDR32,
        THCON_SEC1_REG2_Out_data_format_ADDR32,
        UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
}

inline void _llk_unpack_reconfig_data_format_impl_(
    const std::uint32_t unpA_src_format, const std::uint32_t unpB_src_format,
    const std::uint32_t unpA_dst_format, const std::uint32_t unpB_dst_format) {
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK);

    uint alu_src_format = (unpB_dst_format << ALU_FORMAT_SPEC_REG1_SrcB_SHAMT) |
                          (unpA_dst_format << ALU_FORMAT_SPEC_REG0_SrcA_SHAMT);
    uint alu_src_mask = ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_FORMAT_SPEC_REG1_SrcB_MASK;
    uint32_t alu_config_data = gl_alu_format_spec_reg;

    gl_alu_format_spec_reg = cfg_rmw_mmio_rd_tensix_wr(
        ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, 0, alu_src_mask, alu_src_format, alu_config_data);

    reconfig_unpacker_data_format(
        unpA_src_format,
        unpA_dst_format,
        THCON_SEC0_REG0_TileDescriptor_ADDR32,
        THCON_SEC0_REG2_Out_data_format_ADDR32,
        UNP0_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
    reconfig_unpacker_data_format(
        unpB_src_format,
        unpB_dst_format,
        THCON_SEC1_REG0_TileDescriptor_ADDR32,
        THCON_SEC1_REG2_Out_data_format_ADDR32,
        UNP1_ADDR_CTRL_ZW_REG_1_Zstride_ADDR32);
}

inline void _llk_unpack_dbg_feature_disable_(){
    TT_LLK_DUMP("llk_unpack_dbg_feature_disable()");
    //TBD
}
