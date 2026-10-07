// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/tilize.h"
#include "tt-train/sources/ttml/metal/common/compute_utils.hpp"

constexpr uint32_t work_count = get_compile_time_arg_val(0);
constexpr uint32_t block_ct = get_compile_time_arg_val(1);
constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

constexpr uint32_t cb_act_rm = tt::CBIndex::c_0;
constexpr uint32_t cb_act_tile = tt::CBIndex::c_1;
constexpr uint32_t cb_weights = tt::CBIndex::c_2;
constexpr uint32_t cb_partial = tt::CBIndex::c_3;
constexpr uint32_t cb_output = tt::CBIndex::c_4;
constexpr uint32_t cb_grad = tt::CBIndex::c_5;
constexpr uint32_t cb_conv = tt::CBIndex::c_6;
constexpr uint32_t cb_sigmoid = tt::CBIndex::c_7;
constexpr uint32_t cb_scratch_a = tt::CBIndex::c_8;
constexpr uint32_t cb_scratch_b = tt::CBIndex::c_9;

constexpr uint32_t tap_count = 4U;
constexpr uint32_t one_fp32 = 0x3F800000U;

#ifdef SILU_GRAD
constexpr uint32_t cb_conv_result = cb_conv;
#else
constexpr uint32_t cb_conv_result = cb_output;
#endif

FORCE_INLINE void tilize_window() {
    tilize_init(cb_act_rm, block_ct, cb_act_tile);
    cb_wait_front(cb_act_rm, block_ct);
    cb_reserve_back(cb_act_tile, block_ct);
    tilize_block(cb_act_rm, block_ct, cb_act_tile);
    cb_push_back(cb_act_tile, block_ct);
    cb_pop_front(cb_act_rm, block_ct);
    tilize_uninit(cb_act_rm, cb_act_tile);
}

// conv = sum_tap act_tap * w_tap, accumulated through cb_partial one tap at a time.
FORCE_INLINE void convolve_block() {
    for (uint32_t tap = 0; tap < tap_count; ++tap) {
        tilize_window();
        cb_wait_front(cb_act_tile, block_ct);

        const bool is_final_tap = tap + 1 == tap_count;
        const uint32_t destination = is_final_tap ? cb_conv_result : cb_partial;
        if (tap != 0) {
            cb_wait_front(cb_partial, block_ct);
        }

        reconfig_data_format(cb_act_tile, cb_weights);
        pack_reconfig_data_format(destination);
        mul_bcast_rows_init(cb_act_tile, cb_weights);
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            cb_reserve_back(destination, 1);
            tile_regs_acquire();
            if (tap != 0) {
                mul_bcast_rows_init(cb_act_tile, cb_weights);
            }
            mul_tiles_bcast_rows(cb_act_tile, cb_weights, ct, tap * block_ct + ct, 0);
            if (tap != 0) {
                reconfig_data_format_srca(cb_partial);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_partial);
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_partial, 0, 0);
                reconfig_data_format_srca(cb_act_tile);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, destination);
            tile_regs_release();
            cb_push_back(destination, 1);
            if (tap != 0) {
                cb_pop_front(cb_partial, 1);
            }
        }
        cb_pop_front(cb_act_tile, block_ct);
    }
}

#ifdef SILU_GRAD
// out = grad * sigmoid(u) * (1 + u * (1 - sigmoid(u))), the SiLU derivative at u = conv, one tile at a time.
FORCE_INLINE void silu_grad_tile() {
    cb_wait_front(cb_conv, 1);

    tile_regs_acquire();
    reconfig_data_format_srca(cb_conv);
    copy_init(cb_conv);
    copy_tile(cb_conv, 0, 0);
    sigmoid_tile_init();
    sigmoid_tile(0);
    tile_regs_commit();
    pack_and_push(0, cb_sigmoid);

    cb_wait_front(cb_sigmoid, 1);
    tile_regs_acquire();
    reconfig_data_format_srca(cb_sigmoid);
    copy_init(cb_sigmoid);
    copy_tile(cb_sigmoid, 0, 0);
    binop_with_scalar_tile_init();
    rsub_unary_tile(0, one_fp32);
    tile_regs_commit();
    pack_and_push(0, cb_scratch_a);

    cb_wait_front(cb_scratch_a, 1);
    tile_regs_acquire();
    reconfig_data_format(cb_conv, cb_scratch_a);
    mul_init(cb_conv, cb_scratch_a);
    mul_tiles(cb_conv, cb_scratch_a, 0, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, one_fp32);
    tile_regs_commit();
    pack_and_push(0, cb_scratch_b);
    cb_pop_front(cb_scratch_a, 1);

    cb_wait_front(cb_scratch_b, 1);
    tile_regs_acquire();
    reconfig_data_format(cb_sigmoid, cb_scratch_b);
    mul_init(cb_sigmoid, cb_scratch_b);
    mul_tiles(cb_sigmoid, cb_scratch_b, 0, 0, 0);
    tile_regs_commit();
    pack_and_push(0, cb_scratch_a);
    cb_pop_front(cb_scratch_b, 1);
    cb_pop_front(cb_sigmoid, 1);

    cb_wait_front(cb_scratch_a, 1);
    tile_regs_acquire();
    reconfig_data_format(cb_scratch_a, cb_grad);
    mul_init(cb_scratch_a, cb_grad);
    mul_tiles(cb_scratch_a, cb_grad, 0, 0, 0);
    tile_regs_commit();
    pack_and_push(0, cb_output);
    cb_pop_front(cb_scratch_a, 1);
    cb_pop_front(cb_grad, 1);
    cb_pop_front(cb_conv, 1);
}
#endif

void kernel_main() {
    compute_kernel_hw_startup(cb_act_rm, cb_weights, cb_act_tile);

    if constexpr (num_blocks == 1) {
        cb_wait_front(cb_weights, tap_count * block_ct);
    }
    for (uint32_t item = 0; item < work_count; ++item) {
        if constexpr (num_blocks > 1) {
            cb_wait_front(cb_weights, tap_count * block_ct);
        }

        convolve_block();

#ifdef SILU_GRAD
        cb_wait_front(cb_grad, block_ct);
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            silu_grad_tile();
        }
#endif

        if constexpr (num_blocks > 1) {
            cb_pop_front(cb_weights, tap_count * block_ct);
        }
    }
}
