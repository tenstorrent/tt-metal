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
    std::uint32_t input = get_operand_id(operand);
    std::uint32_t base_address = operands[input].f.fifo_rd_ptr;
    std::uint32_t offset_address = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[input], tile_index);
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
    while (semaphore_read(semaphore::UNPACK_OPERAND_SYNC) > 0);
}

inline void llk_unpack_debug_dump(std::uint8_t *data, std::uint32_t byte_size) {
    debug_dump(data, byte_size);
}

inline void llk_unpack_debug_dump_seek(std::uint8_t offset) {
    debug_dump_seek(offset);
}

inline void llk_unpack_reconfig_data_format_srca_impl(std::uint32_t srca_operand_id)
{
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(unpack_src_format[srca_operand_id]);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(unpack_dst_format[srca_operand_id]);
    TT_SETDMAREG(0, LOWER_HALFWORD(GET_L1_TILE_SIZE((uint)unpack_src_format[srca_operand_id])), 0, LO_16(p_gpr_unpack::TILE_SIZE_A)); // update gpr which holds tile size A
}

inline void llk_unpack_reconfig_data_format_srcb_impl(std::uint32_t srcb_operand_id)
{
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0x0f>(unpack_src_format[srcb_operand_id]);
    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(unpack_dst_format[srcb_operand_id]);
    TT_SETDMAREG(0, LOWER_HALFWORD(GET_L1_TILE_SIZE((uint)unpack_src_format[srcb_operand_id])), 0, LO_16(p_gpr_unpack::TILE_SIZE_B)); // update gpr which holds tile size B
}

inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_new_operand) {
    llk_unpack_reconfig_data_format_srca_impl(get_operand_id(srca_new_operand));
}

inline void llk_unpack_reconfig_data_format_srcb(const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srcb_impl(get_operand_id(srcb_new_operand));
}

inline void llk_unpack_reconfig_data_format_srca(const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand) {
    std::uint32_t old_srca_operand_id = get_operand_id(srca_old_operand);
    std::uint32_t new_srca_operand_id = get_operand_id(srca_new_operand);

    if((unpack_src_format[old_srca_operand_id] != unpack_src_format[new_srca_operand_id])) {
        llk_unpack_reconfig_data_format_srca_impl(new_srca_operand_id);
    }
}

inline void llk_unpack_reconfig_data_format_srcb(const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    std::uint32_t old_srcb_operand_id = get_operand_id(srcb_old_operand);
    std::uint32_t new_srcb_operand_id = get_operand_id(srcb_new_operand);

    if((unpack_src_format[old_srcb_operand_id] != unpack_src_format[new_srcb_operand_id])) {
        llk_unpack_reconfig_data_format_srcb_impl(new_srcb_operand_id);
    }
}

inline void llk_unpack_reconfig_data_format(const std::uint32_t srca_new_operand, const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srca(srca_new_operand);
    llk_unpack_reconfig_data_format_srcb(srcb_new_operand);
}

inline void llk_unpack_reconfig_data_format(const std::uint32_t srca_old_operand, const std::uint32_t srca_new_operand, const std::uint32_t srcb_old_operand, const std::uint32_t srcb_new_operand) {
    llk_unpack_reconfig_data_format_srca(srca_old_operand, srca_new_operand);
    llk_unpack_reconfig_data_format_srcb(srcb_old_operand, srcb_new_operand);
}