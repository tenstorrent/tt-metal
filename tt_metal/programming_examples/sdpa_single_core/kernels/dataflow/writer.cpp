// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(1);
    constexpr uint32_t Sv_chunk_t = get_compile_time_arg_val(2);
    constexpr uint32_t head_dim_t = get_compile_time_arg_val(3);
    constexpr uint32_t num_iter = get_compile_time_arg_val(4);
    constexpr uint32_t identity_scalar_packed = get_compile_time_arg_val(5);

    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    generate_reduce_scaler(cb_identity_scale_in, identity_scalar_packed);

    // // Dummy produce of prev max (used in sdpa reduce max)
    // constexpr uint32_t cb_prev_max_im = tt::CBIndex::c_27;
    // DPRINT << "Producing " << Sq_chunk_t << " tiles in prev_max" << ENDL();
    // cb_reserve_back(cb_prev_max_im, Sq_chunk_t);
    // cb_push_back(cb_prev_max_im, Sq_chunk_t);

    // // Pre-populate prev_sum so iter 0 has uniform structure (no special-casing)
    // constexpr uint32_t cb_prev_sum_im = tt::CBIndex::c_29;
    // cb_reserve_back(cb_prev_sum_im, Sq_chunk_t);
    // cb_push_back(cb_prev_sum_im, Sq_chunk_t);

    // // Pre-populate prev_out so iter 0 has uniform structure (no special-casing)
    // constexpr uint32_t cb_prev_out_im = tt::CBIndex::c_25;
    // cb_reserve_back(cb_prev_out_im, Sq_chunk_t * head_dim_t);
    // cb_push_back(cb_prev_out_im, Sq_chunk_t * head_dim_t);
}
