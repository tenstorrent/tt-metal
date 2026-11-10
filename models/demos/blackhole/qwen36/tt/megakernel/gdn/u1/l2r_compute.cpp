// L2-norm root compute: gdn_spec_step's l2norm_rows<N> op sequence on the group's N tiles:
// tmp = src * src; stats = row-sum(tmp) (REDUCE_ROW SUM over the N tiles into one DST); scratch = rsqrt(stats + eps)
// * post; inv = scratch * mask_T; dst = src * col0(inv) (bcast cols).
#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/bcast.h"
#include "api/compute/reduce.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reg_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "l2r_common.hpp"

void kernel_main() {
    constexpr uint32_t N = get_compile_time_arg_val(0);
    const uint32_t post_bits = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(L_SRC, L_SRC, L_TMP);
    DeviceZoneScopedN("L2R-C");
    cb_wait_front(L_SRC, N);
    // square
    cb_reserve_back(L_TMP, N);
    pack_reconfig_data_format(L_TMP);
    reconfig_data_format(L_SRC, L_SRC);
    mul_init(L_SRC, L_SRC);
    for (uint32_t i = 0; i < N; ++i) {
        tile_regs_acquire();
        mul_tiles(L_SRC, L_SRC, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, L_TMP, i);
        tile_regs_release();
    }
    cb_push_back(L_TMP, N);
    // row reduce over the N tiles
    cb_wait_front(L_TMP, N);
    cb_wait_front(L_SCALER, 1);
    cb_reserve_back(L_STATS, 1);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(L_TMP, L_SCALER, L_STATS);
    tile_regs_acquire();
    for (uint32_t i = 0; i < N; ++i) reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(L_TMP, L_SCALER, i, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, L_STATS);
    tile_regs_release();
    reduce_uninit();
    cb_push_back(L_STATS, 1);
    cb_pop_front(L_TMP, N);
    // inverse_l2
    cb_wait_front(L_STATS, 1);
    cb_wait_front(L_EPS, 1);
    cb_reserve_back(L_SCRATCH, 1);
    pack_reconfig_data_format(L_SCRATCH);
    reconfig_data_format(L_STATS, L_EPS);
    add_init(L_STATS, L_EPS);
    tile_regs_acquire();
    add_tiles(L_STATS, L_EPS, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    binop_with_scalar_tile_init();
    mul_unary_tile(0, post_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, L_SCRATCH, 0);
    tile_regs_release();
    cb_push_back(L_SCRATCH, 1);
    cb_wait_front(L_SCRATCH, 1);
    cb_wait_front(L_MASK, 1);
    cb_reserve_back(L_INV, 1);
    pack_reconfig_data_format(L_INV);
    reconfig_data_format(L_SCRATCH, L_MASK);
    mul_init(L_SCRATCH, L_MASK);
    tile_regs_acquire();
    mul_tiles(L_SCRATCH, L_MASK, 0, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, L_INV, 0);
    tile_regs_release();
    cb_push_back(L_INV, 1);
    cb_pop_front(L_SCRATCH, 1);
    cb_pop_front(L_STATS, 1);
    // scale rows
    cb_wait_front(L_INV, 1);
    cb_reserve_back(L_DST, N);
    pack_reconfig_data_format(L_DST);
    reconfig_data_format(L_SRC, L_INV);
    mul_bcast_cols_init(L_SRC, L_INV);
    for (uint32_t i = 0; i < N; ++i) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(L_SRC, L_INV, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, L_DST, i);
        tile_regs_release();
    }
    cb_push_back(L_DST, N);
    cb_pop_front(L_INV, 1);
    cb_pop_front(L_SRC, N);
}
