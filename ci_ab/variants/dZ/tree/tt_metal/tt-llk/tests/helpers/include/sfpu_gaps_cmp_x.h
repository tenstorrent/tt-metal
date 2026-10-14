// SPDX-License-Identifier: Apache-2.0
// Shadow only (regression audit headroom round): the ordered int32 scalar compares in the compute API form
// (unary_<op>_tile_int32: _calculate_comp_unary_int_, 8 rows, VectorMode::RC; init SFPU_UNARY_INIT(unary_<op>)).
#pragma once

#include <cstdint>

#include "ckernel_sfpu.h"
#include "llk_sfpu_types.h"

#define DST_ACCUM_MODE is_fp32_dest_acc_en

#include "llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h"
#include "llk_sfpu/ckernel_sfpu_unary_comp.h"
#include "sfpu/ckernel_sfpu_comp.h"

namespace ckernel
{
namespace gaps_x
{

enum class GapOp
{
    unary_lt_int32,
    unary_gt_int32,
    unary_le_int32,
    unary_ge_int32,
    unary_lt_int32_n,
    unary_gt_int32_n,
    unary_le_int32_n,
    unary_ge_int32_n,
};

constexpr std::uint32_t I_FIVE = 5u;
constexpr std::uint32_t I_NEG5 = 0xFFFFFFFBu;

template <GapOp OP>
constexpr SfpuType cmp_type()
{
    return (OP == GapOp::unary_lt_int32 || OP == GapOp::unary_lt_int32_n)   ? SfpuType::unary_lt
           : (OP == GapOp::unary_gt_int32 || OP == GapOp::unary_gt_int32_n) ? SfpuType::unary_gt
           : (OP == GapOp::unary_le_int32 || OP == GapOp::unary_le_int32_n) ? SfpuType::unary_le
                                                                            : SfpuType::unary_ge;
}

template <GapOp OP>
constexpr std::uint32_t cmp_scalar()
{
    return (OP == GapOp::unary_lt_int32 || OP == GapOp::unary_gt_int32 || OP == GapOp::unary_le_int32 || OP == GapOp::unary_ge_int32) ? I_FIVE
                                                                                                                                          : I_NEG5;
}

template <GapOp OP, bool APPROX, DataFormat FMT>
inline void gaps_init()
{
    ::ckernel::llk_math_eltwise_unary_sfpu_init<cmp_type<OP>(), DST_ACCUM_MODE>();
}

template <GapOp OP, bool APPROX, DataFormat FMT>
inline void gaps_call(std::uint32_t t0, std::uint32_t, std::uint32_t, std::uint32_t)
{
    constexpr DstSync DST_SYNC_MODE = DstSync::SyncHalf;
#if GAP_ALL32
    SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_comp_unary_int_, (APPROX, cmp_type<OP>(), 32), t0, VectorMode::None, cmp_scalar<OP>());
#else
    SFPU_UNARY_CALL(DST_SYNC_MODE, is_fp32_dest_acc_en, _calculate_comp_unary_int_, (APPROX, cmp_type<OP>(), 8), t0, VectorMode::RC, cmp_scalar<OP>());
#endif
}

} // namespace gaps_x
} // namespace ckernel
