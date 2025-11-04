// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t qk_im_cb = get_compile_time_arg_val(0);
    constexpr uint32_t prev_max_cb = get_compile_time_arg_val(1);
    constexpr uint32_t out_max_cb = get_compile_time_arg_val(2);
    constexpr uint32_t scale_cb = get_compile_time_arg_val(3);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t do_eltwise = get_compile_time_arg_val(6);

    // Init compute?
    mm_init(qk_im_cb, qk_im_cb, qk_im_cb);

    // Inputs: qk_im (rows * cols tiles), prev_max (rows tiles if used), scale (1 tile)
    cb_push_back(qk_im_cb, Sq_chunk_t * Sk_chunk_t);
    cb_push_back(scale_cb, 1);
    if (do_eltwise) {
        cb_push_back(prev_max_cb, Sq_chunk_t);
    }

    reconfig_data_format(qk_im_cb, scale_cb);
    pack_reconfig_data_format(out_max_cb);

    reduce_c_transposed<qk_im_cb, scale_cb, Sq_chunk_t, Sk_chunk_t>(out_max_cb, prev_max_cb, do_eltwise);

    DPRINT << "FINISHED REDUCE\n"
           << "Compile-time args:\n"
           << "  qk_im_cb=" << qk_im_cb << "\n"
           << "  prev_max_cb=" << prev_max_cb << "\n"
           << "  out_max_cb=" << out_max_cb << "\n"
           << "  scale_cb=" << scale_cb << "\n"
           << "  Sq_chunk_t=" << Sq_chunk_t << "\n"
           << "  Sk_chunk_t=" << Sk_chunk_t << "\n"
           << "  do_eltwise=" << do_eltwise << ENDL();

    // Ensure outputs are produced before exiting
    cb_wait_front(out_max_cb, Sk_chunk_t);
}
}  // namespace NAMESPACE
