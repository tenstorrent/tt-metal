/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */


#include <type_traits>

#include "ckernel_globals.h"
#include "ckernel_include.h"
#include "ckernel_sfpu.h"
#include "ckernel_template.h"
#include "cmath_common.h"
#include "llk_format_conversions.h"
#include "llk_math_common.h"
#include "llk_param_structs.h"

using namespace ckernel;
template <SfpuType sfpu_type>
void static_assert_sfpu_type_dependent() {
    static_assert(sfpu_type == SfpuType::unused, "sfpu_type exception");
}
// local function declarations
inline void eltwise_unary_sfpu_configure_addrmod() {
    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }
        .set(ADDR_MOD_3);
}
inline void eltwise_unary_sfpu_configure_mop();

template <SfpuType sfpu_op, bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu(
    uint dst_index,
    uint param0 = 0,
    uint param1 = 0,
    uint param2 = 0,
    uint param3 = 0,
    uint param4 = 0,
    uint param5 = 0) {
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    if constexpr ((Dst == DstSync::SyncTile16) || (Dst == DstSync::SyncTile2)) {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(math_sync_tile_dst_index);
    } else {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    }
#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
        sfpu::calculate_sfpu<sfpu_op, APPROXIMATE>(param0, param1, param2, param3, param4, param5);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }
    math::clear_dst_reg_addr();
}

template <SfpuType sfpu_op, bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_init(
    uint param0 = 0, uint param1 = 0, uint param2 = 0, uint param3 = 0, uint param4 = 0, uint param5 = 0) {
    eltwise_unary_sfpu_configure_addrmod();
    if constexpr (sfpu_op == SfpuType::dropout) {
        sfpu::sfpu_init<APPROXIMATE>(sfpu_op, param2);
    } else {
        sfpu::sfpu_init<APPROXIMATE>(sfpu_op);
    }
    math::reset_counters(p_setrwc::SET_ABD_F);
}

// New LLK SFPU APIs

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_exponential(uint dst_index) {
    llk_math_eltwise_unary_sfpu<SfpuType::exponential, APPROXIMATE, dst_sync>(dst_index);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_exponential_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::exponential, APPROXIMATE>();
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_sqrt(uint dst_index) {
    llk_math_eltwise_unary_sfpu<SfpuType::sqrt, APPROXIMATE, dst_sync>(dst_index);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sqrt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sqrt, APPROXIMATE>();
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_gelu(uint dst_index) {
    llk_math_eltwise_unary_sfpu<SfpuType::gelu, APPROXIMATE, dst_sync>(dst_index);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::gelu, APPROXIMATE>();
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_gelu_derivative(uint dst_index) {
    llk_math_eltwise_unary_sfpu<SfpuType::gelu_derivative, APPROXIMATE, dst_sync>(dst_index);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_gelu_derivative_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::gelu_derivative, APPROXIMATE>();
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_reciprocal(uint dst_index) {
    llk_math_eltwise_unary_sfpu<SfpuType::reciprocal, APPROXIMATE, dst_sync>(dst_index);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_reciprocal_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::reciprocal, APPROXIMATE>();
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_log(uint dst_index) {
    llk_math_eltwise_unary_sfpu<SfpuType::log, APPROXIMATE, dst_sync>(dst_index);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log, APPROXIMATE>();
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_log_with_base(uint dst_index,uint base) {
    llk_math_eltwise_unary_sfpu<SfpuType::log_with_base, APPROXIMATE, dst_sync>(dst_index,base);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_log_with_base_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::log_with_base, APPROXIMATE>();
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_tanh(uint dst_index) {
    llk_math_eltwise_unary_sfpu<SfpuType::tanh, APPROXIMATE, dst_sync>(dst_index);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_tanh_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::tanh, APPROXIMATE>();
}

template <DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_dropout(uint dst_index, int integer_dropout, int scale_factor) {
    constexpr bool dont_care = false;
    llk_math_eltwise_unary_sfpu<SfpuType::dropout, dont_care, dst_sync>(dst_index, integer_dropout, scale_factor);
}

inline void llk_math_eltwise_unary_sfpu_dropout_init(uint seed = 0) {
    constexpr bool dont_care = false;
    constexpr uint dont_care_param = 0;

    llk_math_eltwise_unary_sfpu_init<SfpuType::dropout, dont_care>(dont_care_param, dont_care_param, seed);
}

template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_sigmoid(uint dst_index) {
    llk_math_eltwise_unary_sfpu<SfpuType::sigmoid, APPROXIMATE, dst_sync>(dst_index);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_sigmoid_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::sigmoid, APPROXIMATE>();
}

// Negative
template <bool APPROXIMATE, DstSync dst_sync = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_negative(uint dst_index, int vector_mode = Dim::RC) {
    llk_math_eltwise_unary_sfpu<SfpuType::negative, APPROXIMATE, dst_sync>(dst_index,vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_negative_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::negative, APPROXIMATE>();
}
