// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Isolated test compute kernel for add_bias_bcast_rows helper
// (ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp).
//
// CB layout:
//   c_24 (partials) — synthetic matmul output (host pre-fills)
//   c_2  (bias)     — row-broadcast bias vector (one tile per N-subblock column)
//   c_16 (out)      — biased output
//
// Compile-time args:
//   [0] num_invocations    — how many outer iterations (bh * batch loops)
//   [1] in0_num_subblocks
//   [2] in1_num_subblocks
//   [3] out_subblock_h
//   [4] out_subblock_w
//   [5] bias_ntiles        — total bias tiles (= in1_num_subblocks * out_subblock_w)
//
// Defines:
//   BIAS_ONE_TIME_FRONT  — reader pushes bias once; caller waits once, never pops
//   BIAS_PER_ITER_PUSH   — reader pushes bias per iteration; caller waits+pops per iter
//   HELPER_POST_BIAS_RELU — PostBiasFn applies relu via SFPU

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp"
#include "api/compute/matmul.h"
#include "api/compute/bcast.h"

#ifdef HELPER_POST_BIAS_RELU
#include "api/compute/eltwise_unary/relu.h"

struct ReluPostBias {
    ALWI void operator()(uint32_t num_tiles) const {
        relu_tile_init();
        for (uint32_t i = 0; i < num_tiles; i++) {
            relu_tile(i);
        }
    }
};
#endif

void kernel_main() {
    constexpr uint32_t num_invocations = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(2);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(3);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(4);
    constexpr uint32_t bias_ntiles = get_compile_time_arg_val(5);

    constexpr uint32_t partials_cb = tt::CBIndex::c_24;
    constexpr uint32_t bias_cb = tt::CBIndex::c_2;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;

    // init_bcast expects all three CBs — mm_init style is insufficient because
    // the helper uses add_tiles_bcast_rows via add_bcast_rows_init_short().
    // init_bcast sets the global data formats correctly for the row-bcast add.
    init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::ROW>(partials_cb, bias_cb, out_cb);

    for (uint32_t iter = 0; iter < num_invocations; ++iter) {
#if defined(BIAS_ONE_TIME_FRONT)
        // Production path A (num_blocks_w_dim==1): wait on first iter only, never pop.
        if (iter == 0) {
            cb_wait_front(bias_cb, bias_ntiles);
        }
#elif defined(BIAS_PER_ITER_PUSH)
        // Production path B (num_blocks_w_dim>1): wait + pop every iter.
        cb_wait_front(bias_cb, bias_ntiles);
#endif

#ifdef HELPER_POST_BIAS_RELU
        compute_kernel_lib::add_bias_bcast_rows<partials_cb, bias_cb, out_cb, ReluPostBias>(
            in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w, ReluPostBias{});
#else
        compute_kernel_lib::add_bias_bcast_rows<partials_cb, bias_cb, out_cb>(
            in0_num_subblocks, in1_num_subblocks, out_subblock_h, out_subblock_w);
#endif

#ifdef BIAS_PER_ITER_PUSH
        cb_pop_front(bias_cb, bias_ntiles);
#endif
    }
}
