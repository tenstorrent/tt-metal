// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

//#include "kernel_types.h"
#include "ckernel.h"
#include "ckernel_template.h"
#include "ckernel_sfpu.h"
#include "ckernel_globals.h"
#include "fw_debug.h"
#include "llk_defs.h"

#ifndef SFPU_OP_PARAM
#define SFPU_OP_PARAM 0
#endif

#ifndef FUSE_SQRT_RECIP
#define FUSE_SQRT_RECIP 0
#endif

using namespace ckernel;

namespace ckernel::math
{

constexpr uint DstTileSize[3] = {
    64,     // 32x32 tile shape
    32,     // 32x16, 16x32 tile shape
    16      // 16x16 tile shape
};
constexpr uint DstTileSizeLog2[3] = {
    6,     // 32x32 tile shape
    5,     // 32x16, 16x32 tile shape
    4      // 16x16 tile shape
};

constexpr uint8_t MATH_HALO_ROWS = 4;

inline void reset_counters(const uint setrwc)
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, setrwc);
}

inline void incr_counters(const uint incr_a, const uint incr_b, const uint incr_d, const uint incr_cr)
{
    FWASSERT("Value exceeds RWC_A width of 4 bits", incr_a < 16);
    FWASSERT("Value exceeds RWC_B width of 4 bits", incr_b < 16);
    FWASSERT("Value exceeds RWC_D width of 4 bits", incr_d < 16);
    FWASSERT("Value exceeds RWC_CR width of 4 bits", incr_cr < 16);
    TT_INCRWC(incr_cr, incr_d, incr_b, incr_a);
}

inline void move_d2a_fixed_face(const uint8_t addrmod)
{
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD); // MOVD2A for a whole face assumes unpacker will set a dummy data_valid, so we want to wait on that
    TTI_MOVD2A(p_movd2a::MOV_4_ROWS, addrmod, MATH_HALO_ROWS + 0, 0);
    TTI_MOVD2A(p_movd2a::MOV_4_ROWS, addrmod, MATH_HALO_ROWS + 4, 4);
    TTI_MOVD2A(p_movd2a::MOV_4_ROWS, addrmod, MATH_HALO_ROWS + 8, 8);
    TTI_MOVD2A(p_movd2a::MOV_4_ROWS, addrmod, MATH_HALO_ROWS + 12, 12);
}

inline void move_d2a_row_broadcast_fixed_face(const uint8_t addrmod)
{

    // // Seems to make things 200 clocks slower. Really shouldn't though.
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 0, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 1, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 2, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 3, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 4, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 5, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 6, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 7, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 8, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 9, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 10, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 11, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 12, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 13, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 14, 0);
    TTI_MOVD2A(p_movd2a::MOV_1_ROW, addrmod, MATH_HALO_ROWS + 15, 0);

}

inline void move_a2d_fixed_face(const uint8_t addrmod)
{

    TTI_MOVA2D(p_mova2d::MOV_8_ROWS, addrmod, MATH_HALO_ROWS, 0);
    TTI_MOVA2D(p_mova2d::MOV_8_ROWS, addrmod, MATH_HALO_ROWS, 0);
}

template <uint SrcReg>
inline void wait_bank_valid()
{
    if constexpr (SrcReg == Srcs::SrcA)
    {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCA_VLD);
    }
    else
    {
        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SRCB_VLD);
    }
}

template <uint SrcReg>
inline void clear_bank_valid()
{
    if constexpr (SrcReg == Srcs::SrcA)
    {
        TTI_SETRWC(p_setrwc::CLR_A, 0, 0, 0, 0, p_setrwc::SET_A);
    }
    else
    {
        TTI_SETRWC(p_setrwc::CLR_B, 0, 0, 0, 0, p_setrwc::SET_B);
    }
}

inline void update_dest_offset_id()
{
    //ping-pong between 0 and 1
    dest_offset_id = 1 - dest_offset_id;
}

inline uint32_t get_dest_buffer_base()
{
    return (0 != dest_offset_id) ? DEST_REGISTER_HALF_SIZE : 0x0;
}

inline void wait_math_semaphores()
{
    // wait while math semaphore is on max, no room to write math results
    TTI_SEMWAIT(p_stall::STALL_MATH, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_MAX);
}

inline void set_math_semaphores()
{
    // Tell packer that it has something to pack
    t6_semaphore_post<p_stall::MATH>(semaphore::MATH_PACK);
}

template <DstTileLayout layout, DstTileShape tile_shape>
inline void set_dst_write_addr(uint32_t tile_index)
{
    if constexpr (layout == DstTileLayout::Default) {
        uint dst_index = tile_index << DstTileSizeLog2[tile_shape];
        dst_index = dst_index + get_dest_buffer_base();
        TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, dst_index);
    } else {
        // FIXME MT: add this mapping for other layout
    }

}

// Programming a dst write addr offset that gets added to base
//
inline void clear_dst_reg_addr()
{
    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_D);
}

template <uint num_rows=8>
inline void inc_dst_addr()
{
    static_assert(num_rows <= 15, "num_rows must be <= 15");
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, num_rows, 0, 0, p_setrwc::SET_D);
}

inline void math_dest_wait()
{
    FWLOG0("XX math_full_dest_sync()->wait for whole dest available");
    TTI_SEMWAIT(p_stall::STALL_MATH, semaphore::t6_sem(semaphore::MATH_PACK), p_stall::STALL_ON_MAX);
}

inline void dest_section_flip()
{
    update_dest_offset_id();
    uint base_addr = get_dest_buffer_base();
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, base_addr);
}

template <DstStart Dst>
inline void set_dest_section_base()
{
    uint base_addr;
    if constexpr (Dst == DstStart::StartZero) {
        base_addr = 0;
    } else {
        base_addr = DEST_REGISTER_HALF_SIZE;
    }
    TT_SETC16(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32, base_addr);
}

inline constexpr int get_math_num_fidelity_phases(const int math_fidelity_desc)
{
    return (math_fidelity_desc & 0x7);
}

inline constexpr int get_math_fidelity_increment(const int math_fidelity_desc)
{
    return ((math_fidelity_desc >> 3) & 0x1) + 1;
}


} // namespace ckernel::math
