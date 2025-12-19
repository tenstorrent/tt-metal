// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

namespace NAMESPACE {

template <VectorMode vec_mode = VectorMode::RC, bool fast_and_approx = false>
void sigmoid(uint32_t input_cb, uint32_t output_cb, uint32_t num_tiles) {
    copy_tile_init(input_cb);
    sigmoid_tile_init<fast_and_approx>();

    cb_wait_front(input_cb, num_tiles);
    cb_reserve_back(output_cb, num_tiles);

    tile_regs_acquire();
    for (uint32_t i = 0; i < num_tiles; i++) {
        // Copy input tile to destination register
        copy_tile(input_cb, i, i);
        // Apply sigmoid
        sigmoid_tile<vec_mode, fast_and_approx>(i);
    }
    tile_regs_commit();
    tile_regs_wait();

    // Pack result to output
    pack_tile_block(0, output_cb, num_tiles);

    tile_regs_release();

    cb_pop_front(input_cb, num_tiles);
    cb_push_back(output_cb, num_tiles);
}

template <bool pop_input = true>
void add_bias(uint32_t input0_cb, uint32_t input1_cb, uint32_t output_cb, uint32_t num_tiles) {
    // Perform add bias on sigmoid scores
    add_tiles_init(input0_cb, input1_cb);
    cb_wait_front(input0_cb, num_tiles);
    cb_wait_front(input1_cb, num_tiles);
    cb_reserve_back(output_cb, num_tiles);
    tile_regs_acquire();
    for (uint32_t i = 0; i < num_tiles; i++) {
        add_tiles(input0_cb, input1_cb, i, i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile_block(0, output_cb, num_tiles);
    tile_regs_release();
    if constexpr (pop_input) {
        cb_pop_front(input0_cb, num_tiles);
    }
    cb_pop_front(input1_cb, num_tiles);
    cb_push_back(output_cb, num_tiles);
}

template <VectorMode vec_mode = VectorMode::RC, bool fast_and_approx = false>
void sigmoid_add_bias(uint32_t input_cb, uint32_t bias_cb, uint32_t output_cb, uint32_t num_tiles) {
    // We output both sigmoid and add bias results
    const uint32_t num_output_tiles = num_tiles * 2;
    copy_tile_init(input_cb);
    sigmoid_tile_init<fast_and_approx>();

    cb_wait_front(input_cb, num_tiles);
    cb_wait_front(bias_cb, num_tiles);
    cb_reserve_back(output_cb, num_output_tiles);

    tile_regs_acquire();
    for (uint32_t i = 0; i < num_tiles; i++) {
        // Copy input tile to destination register
        copy_tile(input_cb, i, i);
        // Apply sigmoid
        sigmoid_tile<vec_mode, fast_and_approx>(i);
    }
    binary_dest_reuse_tiles_init<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(bias_cb);
    for (uint32_t i = 0; i < num_tiles; i++) {
        binary_dest_reuse_tiles<ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(bias_cb, i, i, num_tiles + i);
    }
    tile_regs_commit();
    tile_regs_wait();

    // Pack result to output
    pack_tile_block(0, output_cb, num_output_tiles);

    tile_regs_release();

    cb_pop_front(input_cb, num_tiles);
    cb_push_back(output_cb, num_output_tiles);
}

void MAIN {
    constexpr uint32_t input_cb = get_compile_time_arg_val(0);
    constexpr uint32_t bias_cb = get_compile_time_arg_val(1);
    // constexpr uint32_t sigmoid_cb = get_compile_time_arg_val(2);
    constexpr uint32_t output_cb = get_compile_time_arg_val(3);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(4);
    constexpr VectorMode vec_mode = get_compile_time_arg_val(5) ? VectorMode::R : VectorMode::RC;
    constexpr bool fast_and_approx = get_compile_time_arg_val(6) ? true : false;

    binary_op_init_common(input_cb, bias_cb, sigmoid_cb);

    // Wait once and don't pop
    cb_wait_front(bias_cb, num_tiles);

    // sigmoid<vec_mode, fast_and_approx>(input_cb, sigmoid_cb, num_tiles);

    // add_bias<false>(sigmoid_cb, bias_cb, output_cb, num_tiles);

    // cb_pop_front(sigmoid_cb, num_tiles);

    sigmoid_add_bias<vec_mode, fast_and_approx>(input_cb, bias_cb, output_cb, num_tiles);
}
}  // namespace NAMESPACE
