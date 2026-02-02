// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#define EXP_APPROX_MODE false

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/custom_mm.h"

namespace NAMESPACE {

void MAIN {
    // CB indices passed as compile-time args
    constexpr uint32_t cb_s = get_compile_time_arg_val(0);         // q input
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);         // k input
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);       // output CB
    constexpr uint32_t chunk_size = get_compile_time_arg_val(3);   // chunk size
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(4);  // number of tiles in k
    constexpr uint32_t num_tiles_v = get_compile_time_arg_val(5);  // number of tiles in v

    static_assert(chunk_size + num_tiles_v <= 8, "chunk_size + num_tiles_v must be less than or equal to 8");
    custom_mm_block_init(cb_s, cb_k, cb_out, false, num_tiles_k);
    cb_wait_front(cb_s, chunk_size);
    cb_wait_front(cb_k, num_tiles_k * chunk_size);
    cb_reserve_back(cb_out, num_tiles_v);
    tile_regs_acquire();
    // Reconfigure Unpack srcA, TODO: Cleanup
    UNPACK({ TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK); })
    UNPACK({ TTI_WRCFG(p_gpr_unpack::FACE_DIM_8x16, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32); })
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, false, true>(cb_s)));
    // Only for testing, this should already be in dest in full sdpa
    copy_tile_to_dst_init_short(cb_k, cb_s);
    for (uint32_t i = 0; i < chunk_size; i++) {
        copy_tile(cb_s, i, i);
    }
    // Reconfigure Unpack srcA, TODO: Cleanup
    UNPACK({ TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::UNPACK); })
    UNPACK({ TTI_WRCFG(p_gpr_unpack::FACE_DIM_16x16, p_cfg::WRCFG_32b, THCON_SEC0_REG5_Tile_x_dim_cntx0_ADDR32); })
    UNPACK((llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE, false, true>(cb_k)));
    sdpa_custom_mm_reuse_dest_srcb_block_init_short(cb_s, cb_k, false, num_tiles_k);

    sdpa_custom_mm_reuse_dest_srcb_block(cb_s, cb_k, 0, 0, 0, chunk_size, false, chunk_size, num_tiles_v, num_tiles_k);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile_block(chunk_size, cb_out, num_tiles_v);
    tile_regs_release();
    cb_pop_front(cb_k, num_tiles_k * chunk_size);
    cb_push_back(cb_out, num_tiles_v);
    cb_pop_front(cb_s, chunk_size);
}

}  // namespace NAMESPACE
