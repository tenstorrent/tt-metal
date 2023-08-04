#include "llk_math_eltwise_unary_sfpu.h"

template <SfpuType sfpu_op, bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu(
    void (*function_ptr)(uint),
    uint dst_index,
    uint param0 = 0) {
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    if constexpr ((Dst == DstSync::SyncTile16) || (Dst == DstSync::SyncTile2)) {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(math_sync_tile_dst_index);
    } else {
        math::set_dst_write_addr<DstTileLayout::Default, DstTileShape::Tile32x32>(dst_index);
    }
#pragma GCC unroll 0
    for (int face = 0; face < 4; face++) {
        function_ptr(param0);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D);
    }
    math::clear_dst_reg_addr();
}
