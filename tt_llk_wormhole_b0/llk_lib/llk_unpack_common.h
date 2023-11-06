#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "fw_debug.h"
#include "cunpack_common.h"
#include "llk_param_structs.h"

#ifdef PERF_DUMP
#include "ckernel_perf_api.h"
#endif

using namespace ckernel;
using namespace ckernel::unpacker;

void llk_zero_operand(std::uint32_t operand) {
    std::uint32_t input = get_operand_id(operand);

    TT_SETDMAREG(0, 0, 0, LO_16(p_gpr_unpack::OPERAND_OFFSET_ADDR));
    TT_SETDMAREG(0, 0, 0, HI_16(p_gpr_unpack::OPERAND_OFFSET_ADDR));

    std::uint32_t fifo_base_addr = (operands[input].f.fifo_limit + 1) - operands[input].f.fifo_size;
    TT_SETDMAREG(0, LOWER_HALFWORD(fifo_base_addr), 0, LO_16(p_gpr_unpack::p_gpr_unpack::OPERAND_BASE_ADDR));
    TT_SETDMAREG(0, UPPER_HALFWORD(fifo_base_addr), 0, HI_16(p_gpr_unpack::p_gpr_unpack::OPERAND_BASE_ADDR));

    for (std::uint32_t i = 0; i < operands[input].f.fifo_size; i++) {
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
inline void llk_unpack_get_tile(std::uint32_t operand, std::uint32_t tile_index, std::uint32_t *p_tile) {
    TT_LLK_DUMP("llk_unpack_get_tile<{}, {}>({}, {}, tile_pointer)", mail2math, mail2pack, operand, tile_index);
    std::uint32_t input = get_operand_id(operand);
    std::uint32_t base_address = operands[input].f.fifo_rd_ptr;
    std::uint32_t offset_address = operands[input].f.tile_size_words * tile_index;
    std::uint32_t byte_address = (base_address + offset_address + TILE_HEADER_SIZE)<<4;

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
inline void llk_unpack_release_tile(std::uint32_t operand) {
    TT_LLK_DUMP("llk_unpack_release_tile<{}, {}>({})", mail2math, mail2pack, operand);
    while (semaphore_read(semaphore::UNPACK_OPERAND_SYNC) > 0);
}

inline void llk_unpack_debug_dump(std::uint8_t *data, std::uint32_t byte_size) {
    TT_LLK_DUMP("llk_unpack_debug_dump(ptr, {})", byte_size);
    debug_dump(data, byte_size);
}

inline void llk_unpack_debug_dump_seek(std::uint8_t offset) {
    debug_dump_seek(offset);
}

template <bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format_srca_impl(std::uint32_t srca_operand_id)
{
    if constexpr(is_tile_dim_reconfig_en) {
        const uint unpA_num_faces = get_num_faces(srca_operand_id);
        const uint unpA_face_r_dim = get_face_r_dim(srca_operand_id);
        const uint face_dim = unpA_face_r_dim*FACE_C_DIM;

        cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32+1, 16, 0xffff0000>(unpA_num_faces);
        cfg_reg_rmw_tensix<THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32, 0, 0xffffffff>(face_dim | face_dim << 16);
    }

    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(unpack_src_format[srca_operand_id]);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(unpack_dst_format[srca_operand_id]);
    TT_SETDMAREG(0, LOWER_HALFWORD(operands[srca_operand_id].f.tile_size_words), 0, LO_16(p_gpr_unpack::TILE_SIZE_A)); // update gpr which holds tile size A
}

template <bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format_srcb_impl(std::uint32_t srcb_operand_id)
{
    if constexpr(is_tile_dim_reconfig_en) {
        const uint unpB_num_faces = get_num_faces(srcb_operand_id);
        const uint unpB_face_r_dim = get_face_r_dim(srcb_operand_id);
        const uint face_dim = unpB_face_r_dim*FACE_C_DIM;

        cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 16, 0xffff0000>(unpB_face_r_dim*FACE_C_DIM);
        cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32+1, 16, 0xffff0000>(unpB_num_faces);
    }

    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0x0f>(unpack_src_format[srcb_operand_id]);
    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(unpack_dst_format[srcb_operand_id]);
    
    TT_SETDMAREG(0, LOWER_HALFWORD(operands[srcb_operand_id].f.tile_size_words), 0, LO_16(p_gpr_unpack::TILE_SIZE_B)); // update gpr which holds tile size B
}

template <bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    TT_LLK_DUMP("llk_unpack_reconfig_data_format_srca<{}>({})", is_tile_dim_reconfig_en, srca_new_operand);
    llk_unpack_reconfig_data_format_srca_impl<is_tile_dim_reconfig_en>(get_operand_id(srca_new_operand));
}

template <bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    TT_LLK_DUMP("llk_unpack_reconfig_data_format_srcb<{}>({})", is_tile_dim_reconfig_en, srcb_new_operand);
    llk_unpack_reconfig_data_format_srcb_impl<is_tile_dim_reconfig_en>(get_operand_id(srcb_new_operand));
}

template <bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    TT_LLK_DUMP("llk_unpack_reconfig_data_format_srca<{}>({}, {})", is_tile_dim_reconfig_en, srca_old_operand, srca_new_operand);
    std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);

    if((unpack_src_format[old_srca_operand_id] != unpack_src_format[new_srca_operand_id])) {
        llk_unpack_reconfig_data_format_srca_impl<is_tile_dim_reconfig_en>(new_srca_operand_id);
    }
}

template <bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format_srcb(const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    TT_LLK_DUMP("llk_unpack_reconfig_data_format_srcb<{}>({}, {})", is_tile_dim_reconfig_en, srcb_old_operand, srcb_new_operand);
    std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if((unpack_src_format[old_srcb_operand_id] != unpack_src_format[new_srcb_operand_id])) {
        llk_unpack_reconfig_data_format_srcb_impl<is_tile_dim_reconfig_en>(new_srcb_operand_id);
    }
}

template <bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format(const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    TT_LLK_DUMP("llk_unpack_reconfig_data_format<{}>({}, {})", is_tile_dim_reconfig_en, srca_new_operand, srcb_new_operand);
    llk_unpack_reconfig_data_format_srca<is_tile_dim_reconfig_en>(srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<is_tile_dim_reconfig_en>(srcb_new_operand);
}

template <bool is_tile_dim_reconfig_en = false>
inline void llk_unpack_reconfig_data_format(const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand, const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    TT_LLK_DUMP("llk_unpack_reconfig_data_format<{}>({}, {}, {}, {})", is_tile_dim_reconfig_en, srca_old_operand, srca_new_operand, srcb_old_operand, srcb_new_operand);
    llk_unpack_reconfig_data_format_srca<is_tile_dim_reconfig_en>(srca_old_operand, srca_new_operand);
    llk_unpack_reconfig_data_format_srcb<is_tile_dim_reconfig_en>(srcb_old_operand, srcb_new_operand);
}

inline void llk_unpack_dbg_feature_disable(){
    TT_LLK_DUMP("llk_unpack_dbg_feature_disable()");
    reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1<<11); // Set debug feature disable bit 11
                                                           // workaround for bug tenstorrent/budabackend#1372
}

inline void llk_enable_int8_fpu_math() {
    enalbe_int8_fpu_math();
}