// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Round 3 eltwise binary twin of SDPA decode's output rescale (#58723 third review): sdpa_flash_decode.cpp:551-554, OUT_ACC *=
// EXP_MAX_DIFF through compute_common.hpp's mul_block_bcast_cols<Sq_chunk_t, vDHt, true, false> (mode 0), on the half
// (16x32) im and stats tiles of causal decode with up to 16 q heads; mode 1 is the DHT_GRANULARITY form
// (mul_block_bcast_cols<..., false, false>, DHT_GRANULARITY tiles per acquire) that the tree reduction's and the root's
// in-place rescales use (mul_block_bcast_cols_inplace). Compile args: cb_in, cb_stats, cb_out, Sq_chunk_t, vDHt, mode,
// iterations.
#define LLK_ZEROFLAG_OUTLINE 1
#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#ifndef EXP_APPROX_MODE
#define EXP_APPROX_MODE false
#endif

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stats = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);
    constexpr uint32_t mode = get_compile_time_arg_val(5);
    constexpr uint32_t twin_iters = get_compile_time_arg_val(6);

    compute_kernel_hw_startup(cb_in, cb_stats, cb_out);
    for (uint32_t it = 0; it < twin_iters; ++it) {
        reconfig_data_format(cb_in, cb_stats);
        pack_reconfig_data_format(cb_out);
        if constexpr (mode == 0) {
            mul_block_bcast_cols<Sq_chunk_t, vDHt, true, false>(cb_in, cb_stats, cb_out);
        } else {
            mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(cb_in, cb_stats, cb_out);
        }
    }
}
