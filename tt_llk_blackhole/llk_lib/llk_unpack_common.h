// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
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

void _llk_zero_buffer_(const std::uint32_t base_address, const std::uint32_t size) {

    TT_SETDMAREG(0, 0, 0, LO_16(p_gpr_unpack::OPERAND_OFFSET_ADDR));
    TT_SETDMAREG(0, 0, 0, HI_16(p_gpr_unpack::OPERAND_OFFSET_ADDR));

    TT_SETDMAREG(0, LOWER_HALFWORD(base_address), 0, LO_16(p_gpr_unpack::p_gpr_unpack::OPERAND_BASE_ADDR));
    TT_SETDMAREG(0, UPPER_HALFWORD(base_address), 0, HI_16(p_gpr_unpack::p_gpr_unpack::OPERAND_BASE_ADDR));

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
inline void _llk_unpack_get_tile_(std::uint32_t address, std::uint32_t *p_tile) {
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

inline void _llk_unpack_config_tile_dim_srca_impl_(const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32+1, 16, 0xffff0000>(num_faces);
    config_unpacker_0_face_dim<true, p_setadc::UNP_A>(face_r_dim);
}

inline void _llk_unpack_config_tile_dim_srcb_impl_(const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    const uint face_dim = face_r_dim*FACE_C_DIM;
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 16, 0xffff0000>(face_dim);
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32+1, 16, 0xffff0000>(num_faces);
}

inline void _llk_unpack_reconfig_data_format_srca_impl_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t tile_size)
{
    alu_config_u alu_payload = {.val = 0};
    alu_payload.f.ALU_FORMAT_SPEC_REG0_SrcA = unpack_dst_format;
    if ((uint)unpack_src_format == (uint)DataFormat::UInt8) {
        alu_payload.f.ALU_FORMAT_SPEC_REG0_SrcAUnsigned = 1;
    }
    alu_payload.f.ALU_ACC_CTRL_INT8_math_enabled = ((uint)(unpack_dst_format & 0xF) == (uint)DataFormat::Int8) ||
                                                   ((uint)unpack_dst_format == (uint)DataFormat::Int32);
    constexpr uint alu_mask = ALU_FORMAT_SPEC_REG0_SrcA_MASK | ALU_FORMAT_SPEC_REG0_SrcAUnsigned_MASK | ALU_ACC_CTRL_INT8_math_enabled_MASK;
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_ADDR32, 0, alu_mask>(alu_payload.val);
    
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(unpack_src_format);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(unpack_dst_format);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A)); // update gpr which holds tile size A
}

inline void _llk_unpack_reconfig_data_format_srcb_impl_(const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t tile_size)
{
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0x0f>(unpack_src_format);
    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(unpack_dst_format);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B)); // update gpr which holds tile size B
}

inline void _llk_unpack_dbg_feature_disable_(){
    reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1<<11); // Set debug feature disable bit 11
                                                           // workaround for bug tenstorrent/budabackend#1372
}

inline void _llk_unpack_clear_dbg_feature_disable_(){
    reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 0);     // Unset debug feature disable
}

inline void _llk_enable_int8_fpu_math_() {
    enable_int8_fpu_math();
}

inline void _llk_unpack_set_srcb_dummy_valid_() {
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK);
    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
}
