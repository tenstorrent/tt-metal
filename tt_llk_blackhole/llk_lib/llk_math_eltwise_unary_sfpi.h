// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0


#include "ckernel_include.h"
#include "ckernel_template.h"
#include <type_traits>

#include "cmath_common.h"
#include "llk_math_common.h"
#include "ckernel_globals.h"
#include "ckernel_sfpi.h"

using namespace ckernel;
template <SfpiTestType sfpu_type>
void static_assert_sfpi_type_dependent() {
    static_assert(sfpu_type == SfpiTestType::unused_test, "sfpu_type exception");
}
// local function declarations
inline void eltwise_unary_sfpi_configure_addrmod(){
    // NOTE: this kernel is typically used in conjunction with
    //       A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one
    //       that doesn't conflict!
    addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 0},
    }.set(ADDR_MOD_3);
}
inline void eltwise_unary_sfpi_configure_mop();

template <SfpiTestType sfpu_op, DstSync Dst>
inline void llk_math_eltwise_unary_sfpi(
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

    int face = 0;
    sfpi_test::calculate_sfpi<sfpu_op>(param0, param1, param2, param3, param4, param5);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);

    math::clear_dst_reg_addr();
}

inline void llk_math_eltwise_unary_sfpi_init(
    uint param0 = 0, uint param1 = 0, uint param2 = 0, uint param3 = 0, uint param4 = 0, uint param5 = 0) {
    sfpu::_init_sfpu_config_reg();
    eltwise_unary_sfpi_configure_addrmod();
    math::reset_counters(p_setrwc::SET_ABD_F);
}

// New LLK SFPU APIs

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test1(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test1, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test2(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test2, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test3(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test3, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test4(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test4, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test5(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test5, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test6(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test6, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test7(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test7, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test8(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test8, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test9(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test9, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test10(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test10, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test11(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test11, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test12(uint dst_index, uint param0) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test12, dst_sync>(dst_index, param0);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test13(uint dst_index, uint param0) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test13, dst_sync>(dst_index, param0);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test14(uint dst_index, uint param0) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test14, dst_sync>(dst_index, param0);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test15(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test15, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test16(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test16, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test17(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test17, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test18(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test18, dst_sync>(dst_index);
}

template <DstSync dst_sync>
inline void llk_math_eltwise_unary_sfpi_test19(uint dst_index) {
    llk_math_eltwise_unary_sfpi<SfpiTestType::test19, dst_sync>(dst_index);
}
