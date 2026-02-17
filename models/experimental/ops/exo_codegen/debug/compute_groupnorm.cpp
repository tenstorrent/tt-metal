#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t block_hw = get_compile_time_arg_val(0);

    // CB indices
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_eps = tt::CBIndex::c_3;
    constexpr uint32_t cb_scaler_global = tt::CBIndex::c_4;
    constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;
    constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;
    constexpr uint32_t cb_ex2_partial = tt::CBIndex::c_21;
    constexpr uint32_t cb_ex2_global = tt::CBIndex::c_14;
    constexpr uint32_t cb_ex2pe = tt::CBIndex::c_27;
    constexpr uint32_t cb_x = tt::CBIndex::c_24;
    constexpr uint32_t cb_xmm = tt::CBIndex::c_25;
    constexpr uint32_t cb_input_mask = tt::CBIndex::c_28;
    constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;
    constexpr bool FP32_DEST_ACC = false;

    // Init
    binary_op_init_common(cb_in0, cb_input_mask, cb_x);

    // Wait for all persistent data
    cb_wait_front(cb_in0, block_hw);
    cb_wait_front(cb_input_mask, 1);
    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_scaler_global, 1);
    cb_wait_front(cb_eps, 1);

    // =====================================================================
    // Pass 1: Mean — E[x]
    // =====================================================================

    // Step 1.1: Mask input (input * mask -> cb_x)
    mul_tiles_init(cb_in0, cb_input_mask);
    cb_reserve_back(cb_x, block_hw);
    for (uint32_t i = 0; i < block_hw; ++i) {
        tile_regs_acquire();
        mul_tiles(cb_in0, cb_input_mask, (i), 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_x);
        tile_regs_release();
    }
    cb_push_back(cb_x, block_hw);

    // Step 1.2: Local reduce (sum of masked tiles -> cb_ex_partial)
    reconfig_data_format_srcb(cb_input_mask, cb_scaler);
    reduce_init<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_x, cb_scaler, cb_ex_partial);
    cb_reserve_back(cb_ex_partial, 1);
    tile_regs_acquire();
    cb_wait_front(cb_x, block_hw);
    for (uint32_t i = 0; i < block_hw; ++i) {
        reduce_tile<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_x, cb_scaler, (i), 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, cb_ex_partial);
    tile_regs_release();
    cb_pop_front(cb_x, block_hw);
    cb_push_back(cb_ex_partial, 1);
    reduce_uninit<FP32_DEST_ACC>();

    // Step 1.3: Global reduce (partial * 1/N -> mean in cb_ex_global)
    reduce_init<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_ex_partial, cb_scaler_global, cb_ex_global);
    cb_reserve_back(cb_ex_global, 1);
    tile_regs_acquire();
    cb_wait_front(cb_ex_partial, 1);
    reduce_tile<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_ex_partial, cb_scaler_global, 0, scaler0, dst0);
    cb_pop_front(cb_ex_partial, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, cb_ex_global);
    tile_regs_release();
    cb_push_back(cb_ex_global, 1);
    reduce_uninit<FP32_DEST_ACC>();

    // =====================================================================
    // Pass 2: Variance — E[(x - E[x])^2]
    // =====================================================================

    // Step 2.1: Subtract mean (input - mean -> cb_xmm)
    sub_tiles_bcast_scalar_init_short(cb_in0, cb_ex_global);
    cb_reserve_back(cb_xmm, block_hw);
    cb_wait_front(cb_ex_global, 1);
    for (uint32_t i = 0; i < block_hw; ++i) {
        tile_regs_acquire();
        sub_tiles_bcast_scalar(cb_in0, cb_ex_global, (i), 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_xmm);
        tile_regs_release();
    }
    cb_push_back(cb_xmm, block_hw);

    // Step 2.2: Mask residual (residual * mask -> cb_x)
    reconfig_data_format_srcb(cb_ex_global, cb_input_mask);
    mul_tiles_init(cb_xmm, cb_input_mask);
    cb_reserve_back(cb_x, block_hw);
    cb_wait_front(cb_xmm, block_hw);
    for (uint32_t i = 0; i < block_hw; ++i) {
        tile_regs_acquire();
        mul_tiles(cb_xmm, cb_input_mask, (i), 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_x);
        tile_regs_release();
    }
    cb_pop_front(cb_xmm, block_hw);
    cb_push_back(cb_x, block_hw);

    // Step 2.3: Square ((x - E[x])^2 -> cb_xmm)
    reconfig_data_format_srcb(cb_input_mask, cb_x);
    mul_tiles_init(cb_x, cb_x);
    cb_reserve_back(cb_xmm, block_hw);
    cb_wait_front(cb_x, block_hw);
    for (uint32_t i = 0; i < block_hw; ++i) {
        tile_regs_acquire();
        mul_tiles(cb_x, cb_x, (i), (i), 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_xmm);
        tile_regs_release();
    }
    cb_pop_front(cb_x, block_hw);
    cb_push_back(cb_xmm, block_hw);

    // Step 2.4: Local reduce (sum of squares -> cb_ex2_partial)
    reconfig_data_format_srcb(cb_x, cb_scaler);
    reduce_init<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_xmm, cb_scaler, cb_ex2_partial);
    cb_reserve_back(cb_ex2_partial, 1);
    tile_regs_acquire();
    cb_wait_front(cb_xmm, block_hw);
    for (uint32_t i = 0; i < block_hw; ++i) {
        reduce_tile<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_xmm, cb_scaler, (i), 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, cb_ex2_partial);
    tile_regs_release();
    cb_pop_front(cb_xmm, block_hw);
    cb_push_back(cb_ex2_partial, 1);
    reduce_uninit<FP32_DEST_ACC>();

    // Step 2.5: Global reduce (partial variance * 1/N -> variance in cb_ex2_global)
    reduce_init<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_ex2_partial, cb_scaler_global, cb_ex2_global);
    cb_reserve_back(cb_ex2_global, 1);
    tile_regs_acquire();
    cb_wait_front(cb_ex2_partial, 1);
    reduce_tile<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_ex2_partial, cb_scaler_global, 0, scaler0, dst0);
    cb_pop_front(cb_ex2_partial, 1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, cb_ex2_global);
    tile_regs_release();
    cb_push_back(cb_ex2_global, 1);
    reduce_uninit<FP32_DEST_ACC>();

    // Step 2.6: inv_std = 1/sqrt(var + eps)
    cb_wait_front(cb_ex2_global, 1);
    cb_reserve_back(cb_ex2pe, 1);
    tile_regs_acquire();
    add_tiles_init(cb_ex2_global, cb_eps);
    add_tiles(cb_ex2_global, cb_eps, 0, 0, dst0);
    tile_regs_wait();
    rsqrt_tile_init<true>();
    rsqrt_tile<true>(dst0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(dst0, cb_ex2pe);
    tile_regs_release();
    cb_push_back(cb_ex2pe, 1);
    cb_pop_front(cb_ex2_global, 1);

    // =====================================================================
    // Pass 3: Normalize — (x - E[x]) * inv_std
    // =====================================================================

    // Step 3.1: Subtract mean (input - mean -> cb_xmm)
    sub_tiles_bcast_scalar_init_short(cb_in0, cb_ex_global);
    cb_reserve_back(cb_xmm, block_hw);
    for (uint32_t i = 0; i < block_hw; ++i) {
        tile_regs_acquire();
        sub_tiles_bcast_scalar(cb_in0, cb_ex_global, (i), 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_xmm);
        tile_regs_release();
    }
    cb_push_back(cb_xmm, block_hw);

    // Step 3.2: Mask residual (residual * mask -> cb_x)
    reconfig_data_format_srcb(cb_ex_global, cb_input_mask);
    mul_tiles_init(cb_xmm, cb_input_mask);
    cb_reserve_back(cb_x, block_hw);
    cb_wait_front(cb_xmm, block_hw);
    for (uint32_t i = 0; i < block_hw; ++i) {
        tile_regs_acquire();
        mul_tiles(cb_xmm, cb_input_mask, (i), 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_x);
        tile_regs_release();
    }
    cb_pop_front(cb_xmm, block_hw);
    cb_push_back(cb_x, block_hw);

    // Step 3.3: Multiply by inv_std (masked_residual * inv_std -> cb_out0)
    reconfig_data_format_srcb(cb_input_mask, cb_ex2pe);
    mul_tiles_bcast_scalar_init_short(cb_x, cb_ex2pe);
    cb_reserve_back(cb_out0, block_hw);
    cb_wait_front(cb_x, block_hw);
    cb_wait_front(cb_ex2pe, 1);
    for (uint32_t i = 0; i < block_hw; ++i) {
        tile_regs_acquire();
        mul_tiles_bcast_scalar(cb_x, cb_ex2pe, (i), 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_out0);
        tile_regs_release();
    }
    cb_pop_front(cb_x, block_hw);
    cb_push_back(cb_out0, block_hw);

    // Cleanup: pop persistent CBs
    cb_pop_front(cb_in0, block_hw);
    cb_pop_front(cb_input_mask, 1);
    cb_pop_front(cb_ex_global, 1);
    cb_pop_front(cb_ex2pe, 1);
}
