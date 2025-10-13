// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t qk_im_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cur_max_cb = get_compile_time_arg_val(1);
    constexpr uint32_t cur_sum_cb = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(5);

    init_bcast<EltwiseBinaryType::ELWSUB, BroadcastType::ROW>(qk_im_cb, cur_max_cb, qk_im_cb);

    // Push back buffers which point to sharded inputs
    cb_push_back(qk_im_cb, Sq_chunk_t * Sk_chunk_t);
    cb_push_back(cur_max_cb, Sq_chunk_t);

    sub_exp_rows_transposed<qk_im_cb, Sq_chunk_t, Sk_chunk_t, scale_fp32>(cur_max_cb, cur_sum_cb);

    // Wait for result
    cb_wait_front(qk_im_cb, Sq_chunk_t * Sk_chunk_t);
    cb_wait_front(cur_sum_cb, Sq_chunk_t);
}
}  // namespace NAMESPACE
