// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/matmul.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);

    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t num_rows = get_compile_time_arg_val(8);
    constexpr uint32_t Wt = get_compile_time_arg_val(9);

    mm_init();


    uint32_t in0_index = 0;
    uint32_t in1_index = 0;
    uint32_t interm_index = 0;

    DPRINT << "Start of loop" << ENDL();
    DPRINT << "Num rows:  " << num_rows << ENDL();
    DPRINT << "Wt:    " << Wt << ENDL();

    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {

            DPRINT << "Index:" << i << j << ENDL();
            // Transmation Matrix
            ACQ();
            mm_init_short(in_cb, trans_mat_cb);
            cb_wait_front(in_cb, onetile);
            cb_wait_front(trans_mat_cb, onetile);
            cb_reserve_back(rotated_in_interm_cb, onetile);

            DPRINT << "Hang 1" << ENDL();

            matmul_tiles(in_cb, trans_mat_cb, in0_index, in1_index, interm_index, false);
            pack_tile(0, rotated_in_interm_cb);
            REL();

            DPRINT << "Hang 2" << ENDL();
            cb_push_back(rotated_in_interm_cb, onetile);
            cb_pop_front(in_cb, onetile);
            cb_pop_front(trans_mat_cb, onetile);

            DPRINT << "Hang 3" << ENDL();

            cb_wait_front(rotated_in_interm_cb, onetile);
            // Multiply interim_transformation by sin
            cb_wait_front(sin_cb, onetile);
            cb_reserve_back(sin_interm_cb, onetile);

            binary_op_init_common(rotated_in_interm_cb, cos_cb);

            DPRINT << "Hang 4" << ENDL();
            ACQ();
            mul_tiles_init();
            DPRINT << "Hang 4.25" << ENDL();
            mul_tiles(rotated_in_interm_cb, sin_cb, 0, 0, 0);
            DPRINT << "Hang 4.5" << ENDL();
            pack_tile(0, sin_interm_cb);
            REL();

            DPRINT << "Hang 5" << ENDL();
            cb_push_back(sin_interm_cb, onetile);
            cb_pop_front(sin_cb, onetile);

            // Multiply interim_transformation by cos
            cb_wait_front(cos_cb, onetile);
            cb_reserve_back(cos_interm_cb, onetile);
            binary_op_init_common(rotated_in_interm_cb, sin_cb);

            DPRINT << "Hang 6" << ENDL();
            ACQ();
            mul_tiles_init();
            mul_tiles(rotated_in_interm_cb, cos_cb, 0, 0, 0);
            pack_tile(0, cos_interm_cb);
            REL();
            cb_push_back(cos_interm_cb, onetile);
            cb_pop_front(cos_cb, onetile);

            cb_pop_front(rotated_in_interm_cb, onetile);

            // Add applied sin/cos tensors
            cb_wait_front(cos_interm_cb, onetile);
            cb_wait_front(sin_interm_cb, onetile);
            cb_reserve_back(out_cb, onetile);
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
