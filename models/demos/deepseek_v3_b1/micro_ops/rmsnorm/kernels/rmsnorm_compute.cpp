// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_SCALAR

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/add_rsqrt.h"

template <
    uint32_t input_cb,
    uint32_t scalars_cb,
    uint32_t interm_cb,
    uint32_t weight_cb,
    uint32_t output_cb,
    bool fp32_acc,
    uint32_t num_tiles,
    bool rsqrt_fast_approx,
    bool pop_input>
void compute_rmsnorm(uint32_t epsilon) {
    // TODO: #32998: Fuse this without having to spill output of square to interm cb
    {
        // Square the input
        mul_tiles_init(input_cb, input_cb);
        cb_wait_front(input_cb, num_tiles);
        cb_reserve_back(interm_cb, num_tiles + 1);  // Plus 1 for the RMS tile
        tile_regs_acquire();
        for (uint32_t i = 0; i < num_tiles; i++) {
            mul_tiles(input_cb, input_cb, i, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile_block(0, interm_cb, num_tiles);
        tile_regs_release();
        cb_push_back(interm_cb, num_tiles);

        // Calculate the avg of the sum of the squares
        cb_wait_front(interm_cb, num_tiles);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR, fp32_acc>(interm_cb, scalars_cb, interm_cb);
        tile_regs_acquire();
        // TODO: #32998: Instead of accumulating to index 0, accumulate to num_tiles + 1 once bcast reuse is supported
        for (uint32_t i = 0; i < num_tiles; i++) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR, fp32_acc>(interm_cb, scalars_cb, i, 0, 0);
        }
    }
    // TODO: #32998: Avoid having to spill 1/RMS to interm cb
    {
        // TODO: We can move this to the beginning of the function since this is the only sfpu op we use
        add_rsqrt_tile_init();
        add_rsqrt_tile<rsqrt_fast_approx, VectorMode::RC_custom, 1>(0, epsilon);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, interm_cb);
        tile_regs_release();
        reduce_uninit();
        cb_pop_front(interm_cb, num_tiles);
        cb_push_back(interm_cb, 1);  // 1/RMS tile should now be index 0
    }
    {
        // Multiply input by 1/RMS
        cb_wait_front(interm_cb, 1);
        cb_reserve_back(output_cb, num_tiles);
        mul_tiles_bcast_scalar_init_short(input_cb, interm_cb);
        tile_regs_acquire();
        for (uint32_t i = 0; i < num_tiles; i++) {
            // TODO: #32998: Once we have bcast reuse, we will use input_cb index i, reuse dst index num_tiles + 1,
            // output dst index i
            mul_tiles_bcast_scalar(input_cb, interm_cb, i, 0, i);
        }
        if constexpr (pop_input) {
            cb_pop_front(input_cb, num_tiles);
        }
    }
    {
        // Multiply by the weight
        binary_dest_reuse_tiles_init<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(weight_cb);
        for (uint32_t i = 0; i < num_tiles; i++) {
            binary_dest_reuse_tiles<ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(weight_cb, i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile_block(0, output_cb, num_tiles);
        tile_regs_release();
        cb_pop_front(interm_cb, 1);
        cb_push_back(output_cb, num_tiles);
    }
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t scalars_cb = get_compile_time_arg_val(1);
    constexpr uint32_t interm_cb = get_compile_time_arg_val(2);
    constexpr uint32_t gamma_cb = get_compile_time_arg_val(3);
    constexpr uint32_t output_cb = get_compile_time_arg_val(4);
    constexpr bool fp32_acc = get_compile_time_arg_val(5);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(6);
    constexpr bool rsqrt_fast_approx = get_compile_time_arg_val(7);

    uint32_t epsilon = get_arg_val<uint32_t>(0);

    // Init block done only once
    binary_op_init_common(input_cb, input_cb, output_cb);
    cb_wait_front(scalars_cb, 1);
    cb_wait_front(gamma_cb, num_tiles);  // we don't pop, only wait once and reuse
    compute_rmsnorm<input_cb, scalars_cb, interm_cb, gamma_cb, output_cb, fp32_acc, num_tiles, rsqrt_fast_approx, true>(
        epsilon);
}
}  // namespace NAMESPACE
