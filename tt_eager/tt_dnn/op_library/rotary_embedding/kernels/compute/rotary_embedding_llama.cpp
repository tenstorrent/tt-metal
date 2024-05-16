// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

ALWI void MUL_TILES(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles, uint32_t in1_idx) {
    // Multiply input by cos
    cb_wait_front(in0_cb, num_tiles);
    // TODO: Modify cos/sin index
    cb_wait_front(in1_cb, in1_idx + 1);
    cb_reserve_back(out_cb, num_tiles);

    ACQ();
    mul_tiles_init();
    mul_tiles(in0_cb, in1_cb, 0, 0, 0);
    pack_tile(0, out_cb);
    REL();
    cb_push_back(out_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
    cb_pop_front(in1_cb, num_tiles);
}

ALWI void UNTILIZE_TILES(uint32_t in0_cb, uint32_t out_cb, uint32_t num_tiles) {
    untilize_init_short(in0_cb);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    untilize_block(in0_cb, num_tiles, out_cb);
    cb_push_back(out_cb, num_tiles);
    cb_pop_front(in0_cb, num_tiles);
    untilize_uninit(in0_cb);
}

ALWI void TILIZE_ROWS(uint32_t in0_cb, uint32_t sync_cb, uint32_t out_cb, uint32_t num_tiles) {
    tilize_init_short(in0_cb, num_tiles);
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(sync_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);
    tilize_block(in0_cb, num_tiles, out_cb);
    cb_push_back(out_cb, num_tiles);

    // Pop shared cbs after tilize
    cb_pop_front(in0_cb, num_tiles);
    cb_pop_front(sync_cb, num_tiles);
    tilize_uninit(in0_cb);
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t rotated_in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(2);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(3);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(4);

    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(6);
    constexpr uint32_t num_rows = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);

    mm_init(in_cb, trans_mat_cb, rotated_in_cb);

    binary_op_init_common(rotated_in_cb, cos_cb);

    uint32_t updated_cos_cb = cos_cb;
    uint32_t updated_sin_cb = sin_cb;

    uint32_t in1_idx = 0;

    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
            // Transmation Matrix

            unpack_reconfig_data_format(rotated_in_cb, updated_sin_cb);
            pack_reconfig_data_format(out_cb, sin_interm_cb);

            // Multiply rotated input by sin
            MUL_TILES(rotated_in_cb, updated_sin_cb, sin_interm_cb, onetile, in1_idx);

            // Multiply input by cos
            MUL_TILES(rotated_in_cb, updated_cos_cb, cos_interm_cb, onetile, in1_idx);

            // Add applied sin/cos tensors
            cb_wait_front(cos_interm_cb, onetile);
            cb_wait_front(sin_interm_cb, onetile);
            cb_reserve_back(out_cb, onetile);

            unpack_reconfig_data_format_srca(in_cb, cos_interm_cb);
            pack_reconfig_data_format(cos_interm_cb, out_cb);
            ACQ();
            add_tiles_init();
            add_tiles(cos_interm_cb, sin_interm_cb, 0, 0, 0);
            pack_tile(0, out_cb);
            REL();

            cb_push_back(out_cb, onetile);
            cb_pop_front(cos_interm_cb, onetile);
            cb_pop_front(sin_interm_cb, onetile);

        }
    }
}
} // NAMESPACE
