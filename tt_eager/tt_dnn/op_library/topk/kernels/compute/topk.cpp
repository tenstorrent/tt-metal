// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/unpack.h"
#include "compute_kernel_api/pack.h"

// topk llk needs a global variable atm
// this can only be removed once that's fixed
int32_t topk_replay_init = 0;

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t index_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t input_transposed_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_transposed_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t values_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t output_ind_cb_index = get_compile_time_arg_val(5);
    constexpr uint32_t Ht = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t K = get_compile_time_arg_val(8);
    constexpr uint32_t logk = get_compile_time_arg_val(9);
    constexpr uint32_t logWt = get_compile_time_arg_val(10);

    // init pack, compute and unpack
    ckernel::topk_tile_init();
    transpose_wh_init(input_cb_index, input_transposed_cb_index);
    for(uint32_t ht = 0; ht < Ht; ++ht) {
        bool ascending = false;
        cb_reserve_back(input_transposed_cb_index, Wt);
        cb_reserve_back(index_transposed_cb_index, Wt);

        for (uint32_t wt = 0; wt < Wt; wt+=2) {
            acquire_dst(tt::DstMode::Half);
            // local sort into k groups
            cb_wait_front(input_cb_index, 2);
            cb_wait_front(index_cb_index, 2);

            unpack_reconfig_data_format_srca(input_cb_index);
            transpose_wh_init_short(input_cb_index);
            transpose_wh_tile(input_cb_index, wt, 0);
            transpose_wh_tile(input_cb_index, wt+1, 1);

            unpack_reconfig_data_format_srca(index_cb_index);
            transpose_wh_init_short(index_cb_index);
            transpose_wh_tile(index_cb_index, wt, 2);
            transpose_wh_tile(index_cb_index, wt+1, 3);

            // llk_topk_sort -> inplace
            ckernel::topk_local_sort(0, (int) ascending, logk - 1);

            // pack value tiles into cb_intermed1
            pack_reconfig_data_format(input_transposed_cb_index);
            pack_tile(0, input_transposed_cb_index, wt);
            pack_tile(1, input_transposed_cb_index, wt+1);

            // pack index tiles into cb_intermed2
            pack_reconfig_data_format(index_transposed_cb_index);
            pack_tile(2, index_transposed_cb_index, wt);
            pack_tile(3, index_transposed_cb_index, wt+1);

            cb_pop_front(input_cb_index, 2);
            cb_pop_front(index_cb_index, 2);

            release_dst(tt::DstMode::Half);
            ascending = !ascending;
        }

        cb_push_back(input_transposed_cb_index, Wt);
        cb_push_back(index_transposed_cb_index, Wt);

        cb_wait_front(input_transposed_cb_index, Wt);
        cb_wait_front(index_transposed_cb_index, Wt);

        cb_reserve_back(input_transposed_cb_index, Wt);
        cb_reserve_back(index_transposed_cb_index, Wt);


        // iterative merge and rebuild on pairs of tiles
        constexpr uint32_t num_iterations = logWt;
        for (uint32_t m_iter = 0; m_iter < num_iterations; ++m_iter) {
            bool a = false;
            for (uint32_t left_ind = 0; left_ind < Wt / 2; left_ind += 1 << m_iter) {
                acquire_dst(tt::DstMode::Half);

                uint32_t right_ind = left_ind + (1 << m_iter);

                // unpack values into dest
                copy_tile_to_dst_init_short_with_dt(index_transposed_cb_index, input_transposed_cb_index);
                copy_tile(input_transposed_cb_index, left_ind, 0);
                copy_tile(input_transposed_cb_index, right_ind, 1);

                // unpack indices into dest
                copy_tile_to_dst_init_short_with_dt(input_transposed_cb_index, index_transposed_cb_index);
                copy_tile(index_transposed_cb_index, left_ind, 2);
                copy_tile(index_transposed_cb_index, right_ind, 3);

                // merge values - move larger 32 values into 0th dest and lower 32 values into 1st dest
                ckernel::topk_merge(0, m_iter, K);
                // sort within the larger 32 values
                ckernel::topk_rebuild(0, (uint32_t) a, m_iter, K, logk, true);

                // pack value tiles
                pack_reconfig_data_format(input_transposed_cb_index);
                pack_tile(0, input_transposed_cb_index, left_ind);
                pack_tile(1, input_transposed_cb_index, right_ind);

                // pack index tiles
                pack_reconfig_data_format(index_transposed_cb_index);
                pack_tile(2, index_transposed_cb_index, left_ind);
                pack_tile(3, index_transposed_cb_index, right_ind);

                release_dst(tt::DstMode::Half);
                a = !a;
            }
        }

        cb_push_back(input_transposed_cb_index, Wt);
        cb_push_back(index_transposed_cb_index, Wt);

        cb_pop_front(input_transposed_cb_index, Wt);
        cb_pop_front(index_transposed_cb_index, Wt);


        constexpr uint32_t Kt =  K % TILE_WIDTH == 0 ? K/TILE_WIDTH : K/TILE_WIDTH + 1;

        // transpose value tiles and pack into output buffer
        unpack_reconfig_data_format_srca(input_transposed_cb_index);
        transpose_wh_init_short(input_transposed_cb_index);
        pack_reconfig_data_format(input_transposed_cb_index);
        cb_wait_front(input_transposed_cb_index, Kt);
        for (uint32_t i = 0; i < Kt; ++i) {
            acquire_dst(tt::DstMode::Half);
            cb_reserve_back(values_cb_index, 1);

            transpose_wh_tile(input_transposed_cb_index, i, 0);
            pack_tile(0, values_cb_index);

            cb_push_back(values_cb_index, 1);
            release_dst(tt::DstMode::Half);
        }
        cb_pop_front(input_transposed_cb_index, Kt);

        // transpose index tiles and pack into output buffer
        unpack_reconfig_data_format_srca(index_transposed_cb_index);
        transpose_wh_init_short(index_transposed_cb_index);
        pack_reconfig_data_format(index_transposed_cb_index);
        cb_wait_front(index_transposed_cb_index, Kt);
        for (uint32_t i = 0; i < Kt; ++i) {
            acquire_dst(tt::DstMode::Half);
            cb_reserve_back(output_ind_cb_index, 1);

            transpose_wh_tile(index_transposed_cb_index, i, 0);
            pack_tile(0, output_ind_cb_index);

            cb_push_back(output_ind_cb_index, 1);
            release_dst(tt::DstMode::Half);
        }
        cb_pop_front(index_transposed_cb_index, Kt);
    }
}
}
