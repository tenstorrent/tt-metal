// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/tile_move_copy.h"

constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
constexpr uint32_t block_size = get_compile_time_arg_val(1);
constexpr uint32_t num_inner = get_compile_time_arg_val(2);

constexpr auto cb_input_pass_1 = tt::CBIndex::c_0;
constexpr auto cb_input_pass_2 = tt::CBIndex::c_1;
constexpr auto cb_input_pass_3 = tt::CBIndex::c_19;
constexpr auto cb_scaler = tt::CBIndex::c_2;
constexpr auto cb_eps = tt::CBIndex::c_3;
constexpr auto cb_w0 = tt::CBIndex::c_4;
constexpr auto cb_w1 = tt::CBIndex::c_5;
constexpr auto cb_w2 = tt::CBIndex::c_6;
constexpr auto cb_bias = tt::CBIndex::c_7;
constexpr auto cb_sum_x2 = tt::CBIndex::c_8;
constexpr auto cb_sum_x4 = tt::CBIndex::c_9;
constexpr auto cb_sum_x6 = tt::CBIndex::c_10;
constexpr auto cb_inv_rms_x = tt::CBIndex::c_11;
constexpr auto cb_inv_rms_x2 = tt::CBIndex::c_12;
constexpr auto cb_inv_rms_x3 = tt::CBIndex::c_13;
constexpr auto cb_output = tt::CBIndex::c_14;
constexpr auto cb_ones = tt::CBIndex::c_15;
constexpr auto cb_zero = tt::CBIndex::c_17;
constexpr auto cb_debug = tt::CBIndex::c_18;

constexpr uint32_t one_tile = 1U;

#ifndef POLYNORM_STAGE
#define POLYNORM_STAGE 0
#endif

void reduce_sum_to_inv_rms(uint32_t cb_sum, uint32_t cb_inv_rms);

#if POLYNORM_STAGE == 1
void run_stage_passthrough() {
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_1, block_size);
            cb_wait_front(cb_input_pass_2, block_size);
            cb_reserve_back(cb_output, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                tile_regs_acquire();
                copy_tile_init(cb_input_pass_2);
                copy_tile(cb_input_pass_2, block_idx, 0U);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_output);
                pack_tile(0U, cb_output);
                tile_regs_release();
            }
            cb_push_back(cb_output, block_size);
            cb_pop_front(cb_input_pass_1, block_size);
            cb_pop_front(cb_input_pass_2, block_size);
        }
    }
}
#elif POLYNORM_STAGE == 2
void run_stage_square_only() {
    init_sfpu(cb_input_pass_2, cb_output);
    binary_op_init_common(cb_input_pass_2, cb_input_pass_2, cb_output);
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_1, block_size);
            cb_wait_front(cb_input_pass_2, block_size);
            cb_reserve_back(cb_output, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                tile_regs_acquire();
                copy_tile_init(cb_input_pass_2);
                copy_tile(cb_input_pass_2, block_idx, 0U);
                mul_binary_tile_init();
                mul_binary_tile(0U, 0U, 0U);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_output);
                pack_tile(0U, cb_output);
                tile_regs_release();
            }
            cb_push_back(cb_output, block_size);
            cb_pop_front(cb_input_pass_1, block_size);
            cb_pop_front(cb_input_pass_2, block_size);
        }
    }
}
#elif POLYNORM_STAGE == 3
void run_stage_cube_only() {
    init_sfpu(cb_input_pass_2, cb_output);
    binary_op_init_common(cb_input_pass_2, cb_input_pass_2, cb_output);
    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_1, block_size);
            cb_wait_front(cb_input_pass_2, block_size);
            cb_reserve_back(cb_output, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                tile_regs_acquire();
                copy_tile_init(cb_input_pass_2);
                copy_tile(cb_input_pass_2, block_idx, 0U);
                mul_binary_tile_init();
                mul_binary_tile(0U, 0U, 1U);  // x^2
                mul_binary_tile(1U, 0U, 0U);  // x^3
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_output);
                pack_tile(0U, cb_output);
                tile_regs_release();
            }
            cb_push_back(cb_output, block_size);
            cb_pop_front(cb_input_pass_1, block_size);
            cb_pop_front(cb_input_pass_2, block_size);
        }
    }
}
#elif POLYNORM_STAGE == 4
void run_stage_norm_x_only() {
    init_sfpu(cb_input_pass_1, cb_output);
    binary_op_init_common(cb_input_pass_1, cb_input_pass_1, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        bool first_tile = true;

        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_1, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile) {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 0U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 0U);  // x^2 accumulator
                    first_tile = false;
                } else {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 1U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(1U, 1U, 1U);  // x^2
                    add_binary_tile_init();
                    add_binary_tile(0U, 1U, 0U);
                }
            }
            cb_pop_front(cb_input_pass_1, block_size);
        }

        cb_reserve_back(cb_sum_x2, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x2);
        pack_tile(0U, cb_sum_x2);
        tile_regs_release();
        cb_push_back(cb_sum_x2, one_tile);

        reduce_sum_to_inv_rms(cb_sum_x2, cb_inv_rms_x);

        cb_wait_front(cb_inv_rms_x, one_tile);
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_2, block_size);
            cb_reserve_back(cb_output, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                tile_regs_acquire();
                unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x, cb_inv_rms_x);
                unary_bcast<BroadcastType::COL>(cb_inv_rms_x, 0, 1U);

                copy_tile_init(cb_input_pass_2);
                copy_tile(cb_input_pass_2, block_idx, 0U);  // x
                reconfig_data_format(cb_input_pass_2, cb_inv_rms_x);
                mul_binary_tile_init();
                mul_binary_tile(0U, 1U, 0U);

                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_output);
                pack_tile(0U, cb_output);
                tile_regs_release();
            }
            cb_push_back(cb_output, block_size);
            cb_pop_front(cb_input_pass_2, block_size);
        }
        cb_pop_front(cb_inv_rms_x, one_tile);
    }
}
#elif POLYNORM_STAGE == 5
void run_stage_inv_rms_broadcast_only() {
    init_sfpu(cb_input_pass_1, cb_output);
    binary_op_init_common(cb_input_pass_1, cb_input_pass_1, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        bool first_tile = true;

        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_1, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile) {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 0U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 0U);  // x^2 accumulator
                    first_tile = false;
                } else {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 1U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(1U, 1U, 1U);  // x^2
                    add_binary_tile_init();
                    add_binary_tile(0U, 1U, 0U);
                }
            }
            cb_pop_front(cb_input_pass_1, block_size);
        }

        cb_reserve_back(cb_sum_x2, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x2);
        pack_tile(0U, cb_sum_x2);
        tile_regs_release();
        cb_push_back(cb_sum_x2, one_tile);

        reduce_sum_to_inv_rms(cb_sum_x2, cb_inv_rms_x);

        cb_wait_front(cb_inv_rms_x, one_tile);
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_2, block_size);
            cb_reserve_back(cb_output, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                (void)block_idx;
                tile_regs_acquire();
                unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x, cb_inv_rms_x);
                unary_bcast<BroadcastType::COL>(cb_inv_rms_x, 0, 0U);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_output);
                pack_tile(0U, cb_output);
                tile_regs_release();
            }
            cb_push_back(cb_output, block_size);
            cb_pop_front(cb_input_pass_2, block_size);
        }
        cb_pop_front(cb_inv_rms_x, one_tile);
    }
}
#elif POLYNORM_STAGE == 6
void run_stage_weighted_norm_x_with_full_pass1() {
    init_sfpu(cb_input_pass_1, cb_output);
    binary_op_init_common(cb_input_pass_1, cb_input_pass_1, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // pass1a: sum(x^2) using only cb_input_pass_1
        bool first_tile_x2 = true;
        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_1, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile_x2) {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 0U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 0U);  // x^2 accumulator
                    first_tile_x2 = false;
                } else {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 1U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(1U, 1U, 1U);  // x^2
                    add_binary_tile_init();
                    add_binary_tile(0U, 1U, 0U);
                }
            }
            cb_pop_front(cb_input_pass_1, block_size);
        }
        cb_reserve_back(cb_sum_x2, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x2);
        pack_tile(0U, cb_sum_x2);
        tile_regs_release();
        cb_push_back(cb_sum_x2, one_tile);

        // pass1b: sum(x^4), sum(x^6) using cb_input_pass_3
        bool first_tile_x4x6 = true;
        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_3, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile_x4x6) {
                    copy_tile_init(cb_input_pass_3);
                    copy_tile(cb_input_pass_3, block_idx, 0U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 3U);  // x^2
                    mul_binary_tile(3U, 3U, 1U);  // x^4 accumulator
                    mul_binary_tile(1U, 3U, 2U);  // x^6 accumulator
                    first_tile_x4x6 = false;
                } else {
                    copy_tile_init(cb_input_pass_3);
                    copy_tile(cb_input_pass_3, block_idx, 3U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(3U, 3U, 4U);  // x^2
                    mul_binary_tile(4U, 4U, 5U);  // x^4
                    mul_binary_tile(5U, 4U, 6U);  // x^6
                    add_binary_tile_init();
                    add_binary_tile(1U, 5U, 1U);
                    add_binary_tile(2U, 6U, 2U);
                }
            }
            cb_pop_front(cb_input_pass_3, block_size);
        }
        cb_reserve_back(cb_sum_x4, one_tile);
        cb_reserve_back(cb_sum_x6, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x4);
        pack_tile(1U, cb_sum_x4);
        pack_reconfig_data_format(cb_sum_x6);
        pack_tile(2U, cb_sum_x6);
        tile_regs_release();
        cb_push_back(cb_sum_x4, one_tile);
        cb_push_back(cb_sum_x6, one_tile);

        reduce_sum_to_inv_rms(cb_sum_x2, cb_inv_rms_x);

        cb_wait_front(cb_inv_rms_x, one_tile);
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_2, block_size);
            cb_reserve_back(cb_output, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                tile_regs_acquire();
                unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x, cb_inv_rms_x);
                unary_bcast<BroadcastType::COL>(cb_inv_rms_x, 0, 1U);
                copy_tile_init(cb_input_pass_2);
                copy_tile(cb_input_pass_2, block_idx, 0U);  // x
                reconfig_data_format(cb_input_pass_2, cb_inv_rms_x);
                mul_binary_tile_init();
                mul_binary_tile(0U, 1U, 0U);  // norm(x)
                copy_tile_init(cb_w2);
                copy_tile(cb_w2, 0, 2U);
                mul_binary_tile(0U, 2U, 0U);  // w2 * norm(x)
                copy_tile_init(cb_bias);
                copy_tile(cb_bias, 0, 2U);
                add_binary_tile_init();
                add_binary_tile(0U, 2U, 0U);  // + bias
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_output);
                pack_tile(0U, cb_output);
                tile_regs_release();
            }
            cb_push_back(cb_output, block_size);
            cb_pop_front(cb_input_pass_2, block_size);
        }
        cb_pop_front(cb_inv_rms_x, one_tile);
    }
}
#elif POLYNORM_STAGE == 7
void run_stage_weighted_norm_x2_with_full_pass1() {
    init_sfpu(cb_input_pass_1, cb_output);
    binary_op_init_common(cb_input_pass_1, cb_input_pass_1, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // pass1a: consume pass_1 as in full stage.
        bool first_tile_x2 = true;
        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_1, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile_x2) {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 0U);
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 0U);
                    first_tile_x2 = false;
                } else {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 1U);
                    mul_binary_tile_init();
                    mul_binary_tile(1U, 1U, 1U);
                    add_binary_tile_init();
                    add_binary_tile(0U, 1U, 0U);
                }
            }
            cb_pop_front(cb_input_pass_1, block_size);
        }
        cb_reserve_back(cb_sum_x2, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x2);
        pack_tile(0U, cb_sum_x2);
        tile_regs_release();
        cb_push_back(cb_sum_x2, one_tile);

        // pass1b: compute x^4 on pass_3.
        bool first_tile_x4 = true;
        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_3, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile_x4) {
                    copy_tile_init(cb_input_pass_3);
                    copy_tile(cb_input_pass_3, block_idx, 0U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 3U);  // x^2
                    mul_binary_tile(3U, 3U, 1U);  // x^4 accumulator
                    first_tile_x4 = false;
                } else {
                    copy_tile_init(cb_input_pass_3);
                    copy_tile(cb_input_pass_3, block_idx, 3U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(3U, 3U, 4U);  // x^2
                    mul_binary_tile(4U, 4U, 5U);  // x^4
                    add_binary_tile_init();
                    add_binary_tile(1U, 5U, 1U);
                }
            }
            cb_pop_front(cb_input_pass_3, block_size);
        }
        cb_reserve_back(cb_sum_x4, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x4);
        pack_tile(1U, cb_sum_x4);
        tile_regs_release();
        cb_push_back(cb_sum_x4, one_tile);

        reduce_sum_to_inv_rms(cb_sum_x4, cb_inv_rms_x2);

        cb_wait_front(cb_inv_rms_x2, one_tile);
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_2, block_size);
            cb_reserve_back(cb_output, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                tile_regs_acquire();
                unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x2, cb_inv_rms_x2);
                unary_bcast<BroadcastType::COL>(cb_inv_rms_x2, 0, 1U);
                copy_tile_init(cb_input_pass_2);
                copy_tile(cb_input_pass_2, block_idx, 0U);  // x
                mul_binary_tile_init();
                mul_binary_tile(0U, 0U, 0U);  // x^2
                reconfig_data_format(cb_input_pass_2, cb_inv_rms_x2);
                mul_binary_tile_init();
                mul_binary_tile(0U, 1U, 0U);  // norm(x^2)
                copy_tile_init(cb_w1);
                copy_tile(cb_w1, 0, 2U);
                mul_binary_tile(0U, 2U, 0U);  // w1 * norm(x^2)
                copy_tile_init(cb_bias);
                copy_tile(cb_bias, 0, 2U);
                add_binary_tile_init();
                add_binary_tile(0U, 2U, 0U);  // + bias
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_output);
                pack_tile(0U, cb_output);
                tile_regs_release();
            }
            cb_push_back(cb_output, block_size);
            cb_pop_front(cb_input_pass_2, block_size);
        }
        cb_pop_front(cb_inv_rms_x2, one_tile);
    }
}
#elif POLYNORM_STAGE == 8
void run_stage_weighted_norm_x3_with_full_pass1() {
    init_sfpu(cb_input_pass_1, cb_output);
    binary_op_init_common(cb_input_pass_1, cb_input_pass_1, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // pass1a: consume pass_1 as in full stage.
        bool first_tile_x2 = true;
        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_1, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile_x2) {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 0U);
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 0U);
                    first_tile_x2 = false;
                } else {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 1U);
                    mul_binary_tile_init();
                    mul_binary_tile(1U, 1U, 1U);
                    add_binary_tile_init();
                    add_binary_tile(0U, 1U, 0U);
                }
            }
            cb_pop_front(cb_input_pass_1, block_size);
        }
        cb_reserve_back(cb_sum_x2, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x2);
        pack_tile(0U, cb_sum_x2);
        tile_regs_release();
        cb_push_back(cb_sum_x2, one_tile);

        // pass1b: compute x^6 on pass_3.
        bool first_tile_x6 = true;
        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_3, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile_x6) {
                    copy_tile_init(cb_input_pass_3);
                    copy_tile(cb_input_pass_3, block_idx, 0U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 3U);  // x^2
                    mul_binary_tile(3U, 3U, 1U);  // x^4
                    mul_binary_tile(1U, 3U, 2U);  // x^6 accumulator
                    first_tile_x6 = false;
                } else {
                    copy_tile_init(cb_input_pass_3);
                    copy_tile(cb_input_pass_3, block_idx, 3U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(3U, 3U, 4U);  // x^2
                    mul_binary_tile(4U, 4U, 5U);  // x^4
                    mul_binary_tile(5U, 4U, 6U);  // x^6
                    add_binary_tile_init();
                    add_binary_tile(2U, 6U, 2U);
                }
            }
            cb_pop_front(cb_input_pass_3, block_size);
        }
        cb_reserve_back(cb_sum_x6, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x6);
        pack_tile(2U, cb_sum_x6);
        tile_regs_release();
        cb_push_back(cb_sum_x6, one_tile);

        reduce_sum_to_inv_rms(cb_sum_x6, cb_inv_rms_x3);

        cb_wait_front(cb_inv_rms_x3, one_tile);
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_2, block_size);
            cb_reserve_back(cb_output, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                tile_regs_acquire();
                unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x3, cb_inv_rms_x3);
                unary_bcast<BroadcastType::COL>(cb_inv_rms_x3, 0, 1U);
                copy_tile_init(cb_input_pass_2);
                copy_tile(cb_input_pass_2, block_idx, 0U);  // x
                mul_binary_tile_init();
                mul_binary_tile(0U, 0U, 0U);  // x^2
                mul_binary_tile(0U, 0U, 2U);  // x^3
                reconfig_data_format(cb_input_pass_2, cb_inv_rms_x3);
                mul_binary_tile_init();
                mul_binary_tile(2U, 1U, 2U);  // norm(x^3)
                copy_tile_init(cb_w0);
                copy_tile(cb_w0, 0, 3U);
                mul_binary_tile(2U, 3U, 2U);  // w0 * norm(x^3)
                copy_tile_init(cb_bias);
                copy_tile(cb_bias, 0, 3U);
                add_binary_tile_init();
                add_binary_tile(2U, 3U, 2U);  // + bias
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_output);
                pack_tile(2U, cb_output);
                tile_regs_release();
            }
            cb_push_back(cb_output, block_size);
            cb_pop_front(cb_input_pass_2, block_size);
        }
        cb_pop_front(cb_inv_rms_x3, one_tile);
    }
}
#elif POLYNORM_STAGE == 9
void run_stage_inv_rms_x2_broadcast_with_full_pass1() {
    init_sfpu(cb_input_pass_1, cb_output);
    binary_op_init_common(cb_input_pass_1, cb_input_pass_1, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // Consume pass_1 exactly once per row.
        bool first_tile_x2 = true;
        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_1, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile_x2) {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 0U);
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 0U);
                    first_tile_x2 = false;
                } else {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 1U);
                    mul_binary_tile_init();
                    mul_binary_tile(1U, 1U, 1U);
                    add_binary_tile_init();
                    add_binary_tile(0U, 1U, 0U);
                }
            }
            cb_pop_front(cb_input_pass_1, block_size);
        }
        cb_reserve_back(cb_sum_x2, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x2);
        pack_tile(0U, cb_sum_x2);
        tile_regs_release();
        cb_push_back(cb_sum_x2, one_tile);

        // Compute x^4 row sum from pass_3.
        bool first_tile_x4 = true;
        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_3, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile_x4) {
                    copy_tile_init(cb_input_pass_3);
                    copy_tile(cb_input_pass_3, block_idx, 0U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 3U);  // x^2
                    mul_binary_tile(3U, 3U, 1U);  // x^4 accumulator
                    first_tile_x4 = false;
                } else {
                    copy_tile_init(cb_input_pass_3);
                    copy_tile(cb_input_pass_3, block_idx, 3U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(3U, 3U, 4U);  // x^2
                    mul_binary_tile(4U, 4U, 5U);  // x^4
                    add_binary_tile_init();
                    add_binary_tile(1U, 5U, 1U);
                }
            }
            cb_pop_front(cb_input_pass_3, block_size);
        }
        cb_reserve_back(cb_sum_x4, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x4);
        pack_tile(1U, cb_sum_x4);
        tile_regs_release();
        cb_push_back(cb_sum_x4, one_tile);

        reduce_sum_to_inv_rms(cb_sum_x4, cb_inv_rms_x2);

        cb_wait_front(cb_inv_rms_x2, one_tile);
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_2, block_size);
            cb_reserve_back(cb_output, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                (void)block_idx;
                tile_regs_acquire();
                unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x2, cb_inv_rms_x2);
                unary_bcast<BroadcastType::COL>(cb_inv_rms_x2, 0, 0U);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_output);
                pack_tile(0U, cb_output);
                tile_regs_release();
            }
            cb_push_back(cb_output, block_size);
            cb_pop_front(cb_input_pass_2, block_size);
        }
        cb_pop_front(cb_inv_rms_x2, one_tile);
    }
}
#elif POLYNORM_STAGE == 10
void run_stage_inv_rms_x3_broadcast_with_full_pass1() {
    init_sfpu(cb_input_pass_1, cb_output);
    binary_op_init_common(cb_input_pass_1, cb_input_pass_1, cb_output);

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // Consume pass_1 exactly once per row.
        bool first_tile_x2 = true;
        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_1, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile_x2) {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 0U);
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 0U);
                    first_tile_x2 = false;
                } else {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 1U);
                    mul_binary_tile_init();
                    mul_binary_tile(1U, 1U, 1U);
                    add_binary_tile_init();
                    add_binary_tile(0U, 1U, 0U);
                }
            }
            cb_pop_front(cb_input_pass_1, block_size);
        }
        cb_reserve_back(cb_sum_x2, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x2);
        pack_tile(0U, cb_sum_x2);
        tile_regs_release();
        cb_push_back(cb_sum_x2, one_tile);

        // Compute x^6 row sum from pass_3.
        bool first_tile_x6 = true;
        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_3, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile_x6) {
                    copy_tile_init(cb_input_pass_3);
                    copy_tile(cb_input_pass_3, block_idx, 0U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 3U);  // x^2
                    mul_binary_tile(3U, 3U, 1U);  // x^4
                    mul_binary_tile(1U, 3U, 2U);  // x^6 accumulator
                    first_tile_x6 = false;
                } else {
                    copy_tile_init(cb_input_pass_3);
                    copy_tile(cb_input_pass_3, block_idx, 3U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(3U, 3U, 4U);  // x^2
                    mul_binary_tile(4U, 4U, 5U);  // x^4
                    mul_binary_tile(5U, 4U, 6U);  // x^6
                    add_binary_tile_init();
                    add_binary_tile(2U, 6U, 2U);
                }
            }
            cb_pop_front(cb_input_pass_3, block_size);
        }
        cb_reserve_back(cb_sum_x6, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x6);
        pack_tile(2U, cb_sum_x6);
        tile_regs_release();
        cb_push_back(cb_sum_x6, one_tile);

        reduce_sum_to_inv_rms(cb_sum_x6, cb_inv_rms_x3);

        cb_wait_front(cb_inv_rms_x3, one_tile);
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_2, block_size);
            cb_reserve_back(cb_output, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                (void)block_idx;
                tile_regs_acquire();
                unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x3, cb_inv_rms_x3);
                unary_bcast<BroadcastType::COL>(cb_inv_rms_x3, 0, 0U);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_output);
                pack_tile(0U, cb_output);
                tile_regs_release();
            }
            cb_push_back(cb_output, block_size);
            cb_pop_front(cb_input_pass_2, block_size);
        }
        cb_pop_front(cb_inv_rms_x3, one_tile);
    }
}
#endif

void reduce_sum_to_inv_rms(uint32_t cb_sum, uint32_t cb_inv_rms) {
    cb_wait_front(cb_sum, one_tile);
    cb_wait_front(cb_scaler, one_tile);
    cb_wait_front(cb_eps, one_tile);
    cb_reserve_back(cb_inv_rms, one_tile);

    tile_regs_acquire();
    constexpr uint32_t acc_reg = 0U;
    constexpr uint32_t eps_reg = 1U;
    reconfig_data_format(cb_sum, cb_scaler);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_scaler, cb_inv_rms);
    reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sum, cb_scaler, 0, 0, acc_reg);
    reduce_uninit();

    reconfig_data_format_srca(cb_eps);
    copy_tile_init(cb_eps);
    copy_tile(cb_eps, 0, eps_reg);

    reconfig_data_format(cb_sum, cb_eps);
    add_binary_tile_init();
    add_binary_tile(acc_reg, eps_reg, acc_reg);

    sqrt_tile_init();
    sqrt_tile(acc_reg);
    recip_tile_init();
    recip_tile(acc_reg);

    tile_regs_commit();
    tile_regs_wait();
    pack_reconfig_data_format(cb_inv_rms);
    pack_tile(acc_reg, cb_inv_rms);
    tile_regs_release();

    cb_push_back(cb_inv_rms, one_tile);
    cb_pop_front(cb_sum, one_tile);
}

void kernel_main() {
    cb_wait_front(cb_scaler, one_tile);
    cb_wait_front(cb_eps, one_tile);
    cb_wait_front(cb_w0, one_tile);
    cb_wait_front(cb_w1, one_tile);
    cb_wait_front(cb_w2, one_tile);
    cb_wait_front(cb_bias, one_tile);
    cb_wait_front(cb_ones, one_tile);
    cb_wait_front(cb_zero, one_tile);

    init_sfpu(cb_input_pass_1, cb_output);
    binary_op_init_common(cb_input_pass_1, cb_input_pass_1, cb_output);

#if POLYNORM_STAGE == 1
    run_stage_passthrough();
#elif POLYNORM_STAGE == 2
    run_stage_square_only();
#elif POLYNORM_STAGE == 3
    run_stage_cube_only();
#elif POLYNORM_STAGE == 4
    run_stage_norm_x_only();
#elif POLYNORM_STAGE == 5
    run_stage_inv_rms_broadcast_only();
#elif POLYNORM_STAGE == 6
    run_stage_weighted_norm_x_with_full_pass1();
#elif POLYNORM_STAGE == 7
    run_stage_weighted_norm_x2_with_full_pass1();
#elif POLYNORM_STAGE == 8
    run_stage_weighted_norm_x3_with_full_pass1();
#elif POLYNORM_STAGE == 9
    run_stage_inv_rms_x2_broadcast_with_full_pass1();
#elif POLYNORM_STAGE == 10
    run_stage_inv_rms_x3_broadcast_with_full_pass1();
#else

    for (uint32_t row = 0; row < num_rows_per_core; ++row) {
        // Pass 1a: accumulate sum(x^2) only.
        bool first_tile_x2 = true;
        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_1, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile_x2) {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 0U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 0U);  // x^2 accumulator
                    first_tile_x2 = false;
                } else {
                    copy_tile_init(cb_input_pass_1);
                    copy_tile(cb_input_pass_1, block_idx, 1U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(1U, 1U, 1U);  // x^2
                    add_binary_tile_init();
                    add_binary_tile(0U, 1U, 0U);
                }
            }
            cb_pop_front(cb_input_pass_1, block_size);
        }
        cb_reserve_back(cb_sum_x2, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x2);
        pack_tile(0U, cb_sum_x2);
        tile_regs_release();
        cb_push_back(cb_sum_x2, one_tile);

        // Pass 1b: accumulate sum(x^4) and sum(x^6) independently on pass_3.
        bool first_tile_x4x6 = true;
        tile_regs_acquire();
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_3, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                if (first_tile_x4x6) {
                    copy_tile_init(cb_input_pass_3);
                    copy_tile(cb_input_pass_3, block_idx, 0U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 3U);  // x^2
                    mul_binary_tile(3U, 3U, 1U);  // x^4 accumulator
                    mul_binary_tile(1U, 3U, 2U);  // x^6 accumulator
                    first_tile_x4x6 = false;
                } else {
                    copy_tile_init(cb_input_pass_3);
                    copy_tile(cb_input_pass_3, block_idx, 0U);  // x
                    mul_binary_tile_init();
                    mul_binary_tile(0U, 0U, 3U);  // x^2
                    mul_binary_tile(3U, 3U, 4U);  // x^4
                    mul_binary_tile(4U, 3U, 5U);  // x^6
                    add_binary_tile_init();
                    add_binary_tile(1U, 4U, 1U);
                    add_binary_tile(2U, 5U, 2U);
                }
            }
            cb_pop_front(cb_input_pass_3, block_size);
        }
        cb_reserve_back(cb_sum_x4, one_tile);
        cb_reserve_back(cb_sum_x6, one_tile);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_sum_x4);
        pack_tile(1U, cb_sum_x4);
        pack_reconfig_data_format(cb_sum_x6);
        pack_tile(2U, cb_sum_x6);
        tile_regs_release();
        cb_push_back(cb_sum_x4, one_tile);
        cb_push_back(cb_sum_x6, one_tile);

#ifdef POLYNORM_DEBUG
        if (row == 0U) {
            cb_reserve_back(cb_debug, 3U);
            tile_regs_acquire();
            copy_tile_init(cb_sum_x2);
            copy_tile(cb_sum_x2, 0, 0U);
            copy_tile_init(cb_sum_x4);
            copy_tile(cb_sum_x4, 0, 1U);
            copy_tile_init(cb_sum_x6);
            copy_tile(cb_sum_x6, 0, 2U);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_debug);
            pack_tile(0U, cb_debug);
            pack_tile(1U, cb_debug);
            pack_tile(2U, cb_debug);
            tile_regs_release();
            cb_push_back(cb_debug, 3U);
        }
#endif

        reduce_sum_to_inv_rms(cb_sum_x2, cb_inv_rms_x);
        reduce_sum_to_inv_rms(cb_sum_x4, cb_inv_rms_x2);
        reduce_sum_to_inv_rms(cb_sum_x6, cb_inv_rms_x3);

#ifdef POLYNORM_DEBUG
        // Export first-row inv_rms intermediates to writer thread for deterministic printing.
        if (row == 0U) {
            cb_reserve_back(cb_debug, 3U);
            tile_regs_acquire();
            copy_tile_init(cb_inv_rms_x);
            copy_tile(cb_inv_rms_x, 0, 0U);
            copy_tile_init(cb_inv_rms_x2);
            copy_tile(cb_inv_rms_x2, 0, 1U);
            copy_tile_init(cb_inv_rms_x3);
            copy_tile(cb_inv_rms_x3, 0, 2U);
            tile_regs_commit();
            tile_regs_wait();
            pack_reconfig_data_format(cb_debug);
            pack_tile(0U, cb_debug);
            pack_tile(1U, cb_debug);
            pack_tile(2U, cb_debug);
            tile_regs_release();
            cb_push_back(cb_debug, 3U);
        }
#endif

        // Pass 2: recompute x/x^2/x^3, normalize each, apply weights, then add bias.
        cb_wait_front(cb_inv_rms_x, one_tile);
        cb_wait_front(cb_inv_rms_x2, one_tile);
        cb_wait_front(cb_inv_rms_x3, one_tile);
        for (uint32_t col = 0; col < num_inner; col += block_size) {
            cb_wait_front(cb_input_pass_2, block_size);
            cb_reserve_back(cb_output, block_size);
            for (uint32_t block_idx = 0; block_idx < block_size; ++block_idx) {
                // term 1: w0 * norm(x^3) -> scratch cb_sum_x2
                cb_reserve_back(cb_sum_x2, one_tile);
                tile_regs_acquire();
                unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x3, cb_inv_rms_x3);
                unary_bcast<BroadcastType::COL>(cb_inv_rms_x3, 0, 5U);
                copy_tile_init(cb_input_pass_2);
                copy_tile(cb_input_pass_2, block_idx, 0U);  // x
                mul_binary_tile_init();
                mul_binary_tile(0U, 0U, 1U);  // x^2
                mul_binary_tile(1U, 0U, 2U);  // x^3
                reconfig_data_format(cb_input_pass_2, cb_inv_rms_x3);
                mul_binary_tile_init();
                mul_binary_tile(2U, 5U, 2U);  // norm(x^3)
                copy_tile_init(cb_w0);
                copy_tile(cb_w0, 0, 4U);
                mul_binary_tile(2U, 4U, 2U);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_sum_x2);
                pack_tile(2U, cb_sum_x2);
                tile_regs_release();
                cb_push_back(cb_sum_x2, one_tile);

                // term 2: w1 * norm(x^2) -> scratch cb_sum_x4
                cb_reserve_back(cb_sum_x4, one_tile);
                tile_regs_acquire();
                unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x2, cb_inv_rms_x2);
                unary_bcast<BroadcastType::COL>(cb_inv_rms_x2, 0, 5U);
                copy_tile_init(cb_input_pass_2);
                copy_tile(cb_input_pass_2, block_idx, 0U);  // x
                mul_binary_tile_init();
                mul_binary_tile(0U, 0U, 1U);  // x^2
                reconfig_data_format(cb_input_pass_2, cb_inv_rms_x2);
                mul_binary_tile_init();
                mul_binary_tile(1U, 5U, 1U);  // norm(x^2)
                copy_tile_init(cb_w1);
                copy_tile(cb_w1, 0, 4U);
                mul_binary_tile(1U, 4U, 1U);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_sum_x4);
                pack_tile(1U, cb_sum_x4);
                tile_regs_release();
                cb_push_back(cb_sum_x4, one_tile);

                // term 3: w2 * norm(x) -> scratch cb_sum_x6
                cb_reserve_back(cb_sum_x6, one_tile);
                tile_regs_acquire();
                unary_bcast_init<BroadcastType::COL>(cb_inv_rms_x, cb_inv_rms_x);
                unary_bcast<BroadcastType::COL>(cb_inv_rms_x, 0, 5U);
                copy_tile_init(cb_input_pass_2);
                copy_tile(cb_input_pass_2, block_idx, 0U);  // x
                reconfig_data_format(cb_input_pass_2, cb_inv_rms_x);
                mul_binary_tile_init();
                mul_binary_tile(0U, 5U, 0U);  // norm(x)
                copy_tile_init(cb_w2);
                copy_tile(cb_w2, 0, 4U);
                mul_binary_tile(0U, 4U, 0U);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_sum_x6);
                pack_tile(0U, cb_sum_x6);
                tile_regs_release();
                cb_push_back(cb_sum_x6, one_tile);

                // final combine from scratch terms + bias
                cb_wait_front(cb_sum_x2, one_tile);
                cb_wait_front(cb_sum_x4, one_tile);
                cb_wait_front(cb_sum_x6, one_tile);
                tile_regs_acquire();
                copy_tile_init(cb_sum_x2);
                copy_tile(cb_sum_x2, 0, 2U);
                copy_tile_init(cb_sum_x4);
                copy_tile(cb_sum_x4, 0, 1U);
                add_binary_tile_init();
                add_binary_tile(2U, 1U, 2U);
                copy_tile_init(cb_sum_x6);
                copy_tile(cb_sum_x6, 0, 0U);
                add_binary_tile(2U, 0U, 2U);
                copy_tile_init(cb_bias);
                copy_tile(cb_bias, 0, 4U);
                add_binary_tile(2U, 4U, 2U);
                tile_regs_commit();
                tile_regs_wait();
                pack_reconfig_data_format(cb_output);
                pack_tile(2U, cb_output);
                tile_regs_release();
                cb_pop_front(cb_sum_x2, one_tile);
                cb_pop_front(cb_sum_x4, one_tile);
                cb_pop_front(cb_sum_x6, one_tile);
            }

            cb_push_back(cb_output, block_size);
            cb_pop_front(cb_input_pass_2, block_size);
        }
        cb_pop_front(cb_inv_rms_x, one_tile);
        cb_pop_front(cb_inv_rms_x2, one_tile);
        cb_pop_front(cb_inv_rms_x3, one_tile);
    }
#endif

    cb_pop_front(cb_scaler, one_tile);
    cb_pop_front(cb_eps, one_tile);
    cb_pop_front(cb_w0, one_tile);
    cb_pop_front(cb_w1, one_tile);
    cb_pop_front(cb_w2, one_tile);
    cb_pop_front(cb_bias, one_tile);
    cb_pop_front(cb_ones, one_tile);
    cb_pop_front(cb_zero, one_tile);
}
