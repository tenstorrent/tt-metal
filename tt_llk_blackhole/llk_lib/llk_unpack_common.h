// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_ops.h"
#include "cunpack_common.h"
#include "fw_debug.h"

#ifdef PERF_DUMP
#include "ckernel_perf_api.h"
#endif

using namespace ckernel;
using namespace ckernel::unpacker;

// If `x` is the result of loading from memory, placing `consume_discard(x)` somewhere
// will ensure that code after `consume_discard(x)` doesn't start until the load is complete.
#define consume_discard(x) __asm volatile("andi x0, %0, 0" : : "r"((x)) : "memory")

// This function stores a value to memory, and then immediately reads it back.
// The load result will not be available until the store has completed.
// This will make sure any subsequent instruction will see the store as complete.
static inline __attribute__((always_inline)) uint32_t store_then_load(volatile uint32_t *addr, uint32_t to_store)
{
    uint32_t result;
    __asm volatile("sw %2, %1; lw %0, %1" : "=r"(result) : "m"(*addr), "r"(to_store));
    return result;
}

void _llk_zero_buffer_(const std::uint32_t base_address, const std::uint32_t size)
{
    TT_SETDMAREG(0, 0, 0, LO_16(p_gpr_unpack::OPERAND_OFFSET_ADDR));
    TT_SETDMAREG(0, 0, 0, HI_16(p_gpr_unpack::OPERAND_OFFSET_ADDR));

    TT_SETDMAREG(0, LOWER_HALFWORD(base_address), 0, LO_16(p_gpr_unpack::p_gpr_unpack::OPERAND_BASE_ADDR));
    TT_SETDMAREG(0, UPPER_HALFWORD(base_address), 0, HI_16(p_gpr_unpack::p_gpr_unpack::OPERAND_BASE_ADDR));

    for (std::uint32_t i = 0; i < size; i++)
    {
        TTI_STOREIND(1, 0, p_ind::LD_16B, LO_16(p_gpr_unpack::OPERAND_OFFSET_ADDR), p_ind::INC_16B, p_gpr_unpack::ZERO_0, p_gpr_unpack::OPERAND_BASE_ADDR);
    }
}

template <bool mail2math = true, bool mail2pack = true>
inline void _llk_unpack_get_tile_(std::uint32_t address, std::uint32_t *p_tile)
{
    std::uint32_t byte_address = (address) << 4;

    if constexpr (mail2math || mail2pack)
    {
        // We only need to make sure the last semaphore_post happens before mailbox write.
        // When we need to do 2 semaphore posts the first one is a general one, the second
        // one is done using store_then_load. When we need only one, this one is skipped.
        if constexpr (mail2math && mail2pack)
        {
            semaphore_post(semaphore::UNPACK_OPERAND_SYNC);
        }

        uint32_t sem_tmp = store_then_load(&pc_buf_base[PC_BUF_SEMAPHORE_BASE + semaphore::UNPACK_OPERAND_SYNC], 0);
        consume_discard(sem_tmp);

        if constexpr (mail2math)
        {
            mailbox_write(ThreadId::MathThreadId, byte_address);
        }

        if constexpr (mail2pack)
        {
            mailbox_write(ThreadId::PackThreadId, byte_address);
        }
    }

    *p_tile = byte_address;
}

template <bool mail2math = true, bool mail2pack = true>
inline void _llk_unpack_release_tile_()
{
    while (semaphore_read(semaphore::UNPACK_OPERAND_SYNC) > 0)
        ;
}

inline void _llk_unpack_debug_dump_(std::uint8_t *data, std::uint32_t byte_size)
{
    debug_dump(data, byte_size);
}

inline void _llk_unpack_debug_dump_seek_(std::uint8_t offset)
{
    debug_dump_seek(offset);
}

inline void _llk_unpack_config_tile_dim_srca_impl_(const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32 + 1, 16, 0xffff0000>(num_faces);
    config_unpacker_0_face_dim<true, p_setadc::UNP_A>(face_r_dim);
}

inline void _llk_unpack_config_tile_dim_srcb_impl_(const std::uint32_t face_r_dim = FACE_R_DIM, const std::uint32_t num_faces = 4)
{
    const uint face_dim = face_r_dim * FACE_C_DIM;
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 16, 0xffff0000>(face_dim);
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32 + 1, 16, 0xffff0000>(num_faces);
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void _llk_unpack_reconfig_data_format_srca_impl_(
    const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t tile_size)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK0);
    if constexpr (to_from_int8)
    {
        static_assert(is_fp32_dest_acc_en, "Reconfiguring unpack to/from Int8 formats requires FP32 Dest mode enabled");
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcAUnsigned_RMW>(((uint)unpack_src_format == (uint)DataFormat::UInt8) ? 1 : 0);
    }
    cfg_reg_rmw_tensix<THCON_SEC0_REG0_TileDescriptor_ADDR32, 0, 0x0f>(unpack_src_format);
    cfg_reg_rmw_tensix<THCON_SEC0_REG2_Out_data_format_RMW>(unpack_dst_format);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_A)); // update gpr which holds tile size A
}

template <bool is_fp32_dest_acc_en, bool to_from_int8 = false>
inline void _llk_unpack_reconfig_data_format_srcb_impl_(
    const std::uint32_t unpack_src_format, const std::uint32_t unpack_dst_format, const std::uint32_t tile_size)
{
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK1);
    if constexpr (to_from_int8)
    {
        static_assert(is_fp32_dest_acc_en, "Reconfiguring unpack to/from Int8 formats requires FP32 Dest mode enabled");
        cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcBUnsigned_RMW>(((uint)unpack_src_format == (uint)DataFormat::UInt8) ? 1 : 0);
    }
    cfg_reg_rmw_tensix<THCON_SEC1_REG0_TileDescriptor_ADDR32, 0, 0x0f>(unpack_src_format);
    cfg_reg_rmw_tensix<THCON_SEC1_REG2_Out_data_format_RMW>(unpack_dst_format);
    TT_SETDMAREG(0, LOWER_HALFWORD(tile_size), 0, LO_16(p_gpr_unpack::TILE_SIZE_B)); // update gpr which holds tile size B
}

inline void _llk_unpack_dbg_feature_disable_()
{
    reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 1 << 11); // Set debug feature disable bit 11
                                                             // workaround for bug tenstorrent/budabackend#1372
}

inline void _llk_unpack_clear_dbg_feature_disable_()
{
    reg_write(RISCV_DEBUG_REG_DBG_FEATURE_DISABLE, 0); // Unset debug feature disable
}

inline void _llk_enable_int8_fpu_math_()
{
    enable_int8_fpu_math();
}

inline void _llk_unpack_set_srcb_dummy_valid_()
{
    TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::UNPACK);
    TTI_UNPACR_NOP(SrcB, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
    TTI_UNPACR_NOP(SrcA, 0, 0, p_unpacr_nop::SET_DVALID, 0, 0, 0, 0, p_unpacr_nop::UNP_ZEROSRC);
}
