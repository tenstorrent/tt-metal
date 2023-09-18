#pragma once
#include "llk_param_structs.h"

#include "ckernel_include.h"
#include "ckernel_template.h"
#include <type_traits>

#include "cmath_common.h"
#include "llk_math_common.h"
#include "llk_format_conversions.h"
#include "ckernel_globals.h"
#include "ckernel_sfpu.h"

using namespace ckernel;
// local function declarations
template <SfpuType sfpu_op>
inline void eltwise_binary_sfpu_configure_addrmod(){
    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!

    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }.set(ADDR_MOD_7);

}
inline void eltwise_binary_sfpu_configure_mop();

template <SfpuType sfpu_op, bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_binary_sfpu(
    const uint operand,
    uint dst_index_a, 
    uint dst_index_b, 
    int vector_mode = (int)Dim::RC,
    uint param0 = 0,
    uint param1 = 0,
    uint param2 = 0,
    uint param3 = 0,
    uint param4 = 0,
    uint param5 = 0) {
    constexpr int ITERATIONS = 8;
    uint dst_index = (dst_index_a <= dst_index_b) ? dst_index_a : dst_index_b;
    param0 = (dst_index_a > dst_index_b) ? dst_index_a-dst_index_b : dst_index_b-dst_index_a;
    if constexpr ((Dst == DstSync::SyncTile16) || (Dst == DstSync::SyncTile2)) {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(math_sync_tile_dst_index);
    } else {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    }
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    if (vector_mode == (int)Dim::R) {
        const std::uint32_t operand_id = get_operand_id(operand);
        // Do a row vector, Face0 + Face1 -- first iteration (first row)
        const int iterations = (math::get_num_faces(operand_id) < 4) ? 
                                    ((math::get_face_r_dim(operand_id) <= 2) ? 2 : math::get_face_r_dim(operand_id)/2) : 2; // At least 2 iterations for odd and even columns
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++) {
            sfpu::calculate_sfpu<sfpu_op, APPROXIMATE, 0, ITERATIONS>(iterations, param0, param1, param2, param3, param4, param5);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    } else if (vector_mode == (int)Dim::C) {
        // Do a column vector, Face0 + Face2 if tile is 32x32 or Face0+Face1 if tiles is 32x16 -- All iterations for full face
        const std::uint32_t operand_id = get_operand_id(operand);
#pragma GCC unroll 0
        for (int face = 0; face < 2; face++) {
            sfpu::calculate_sfpu<sfpu_op, APPROXIMATE, 0, ITERATIONS>(ITERATIONS, param0, param1, param2, param3, param4, param5);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            if (math::get_num_faces(operand_id)>2) { // Skip next 2 faces if tile is 32x32
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
                TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            }
        }
        if (math::get_num_faces(operand_id)<=2) { 
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }    
    } else {
        // Do all four faces, and iterate through all 4 blocks of 4 rows each
#pragma GCC unroll 0
        for (int face = 0; face < 4; face++) {
            sfpu::calculate_sfpu<sfpu_op, APPROXIMATE, 0, ITERATIONS>(ITERATIONS, param0, param1, param2, param3, param4, param5);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
            TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        }
    }
    math::clear_dst_reg_addr(); 

    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    math::clear_addr_mod_base();
}

template <SfpuType sfpu_op, bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_init(
    const uint operand, uint param0 = 0, uint param1 = 0, uint param2 = 0, uint param3 = 0, uint param4 = 0, uint param5 = 0) {
    TT_LLK_DUMP("llk_math_eltwise_binary_sfpu_init<{}, {}>({}, {}, {}, {}, {}, {})", sfpu_op, APPROXIMATE, param0, param1, param2, param3, param4, param5);
    eltwise_binary_sfpu_configure_addrmod< sfpu_op >();
    if constexpr (sfpu_op == SfpuType::requant_int32) {
        sfpu::sfpu_init<APPROXIMATE>(sfpu_op, param0);
    } else if constexpr (sfpu_op == SfpuType::dequant_int32) {
        sfpu::sfpu_init<APPROXIMATE>(sfpu_op, param0);
    } else {
        sfpu::sfpu_init<APPROXIMATE>(sfpu_op);
    }
    math::reset_counters(p_setrwc::SET_ABD_F);
}

// New LLK SFPU APIs


template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_binary_sfpu_requant_int32(const uint operand, uint dst_index_a, uint dst_index_b, int vector_mode = (int)Dim::RC) {
    llk_math_eltwise_binary_sfpu<SfpuType::requant_int32, APPROXIMATE, dst_sync>(operand, dst_index_a, dst_index_b, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_requant_int32_init(const uint operand, const uint zero_point) {
    llk_math_eltwise_binary_sfpu_init<SfpuType::requant_int32, APPROXIMATE>(operand, zero_point);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_binary_sfpu_dequant_int32(const uint operand, uint dst_index_a, uint dst_index_b, int vector_mode = (int)Dim::RC) {
    llk_math_eltwise_binary_sfpu<SfpuType::dequant_int32, APPROXIMATE, dst_sync>(operand, dst_index_a, dst_index_b, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_binary_sfpu_dequant_int32_init(const uint operand, const uint zero_point) {
    llk_math_eltwise_binary_sfpu_init<SfpuType::dequant_int32, APPROXIMATE>(operand, zero_point);
}
