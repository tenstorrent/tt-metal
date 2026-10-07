// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Round 3 eltwise binary twin of deepseek_v3_b1 post_sdpa's SDPA reduce multiply (SdpaReduceWorker, round 1 with both sides
// valid): sdpa_tail_streaming verbatim (models/demos/deepseek_v3_b1/unified_kernels/sdpa_reduce_worker.hpp:220-277) called as compute_impl (:819-888) does after
// post_sdpa_kernel.cpp:313's deepseek_compute_kernel_init, with the CBs the other RISCs pop popped here; run TWIN_ITERS
// times. Compile args: cb_local_l, cb_local_ms, cb_neighbor_l, cb_neighbor_ms, cb_r1_result_l, cb_r1_result_ms,
// scale_fp32, block_size, num_l_blocks, iterations.

// REDUCE_OP and REDUCE_DIM must be defined before including compute headers
#ifndef REDUCE_OP
#define REDUCE_OP (PoolType::MAX)
#endif
#ifndef REDUCE_DIM
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#endif
#ifndef EXP_APPROX_MODE
#define EXP_APPROX_MODE false
#endif

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include <cstdint>

// Include SDPA LLK APIs for srcB reuse pattern and sdpa_tail reduction
#include "api/compute/experimental/sdpa.h"
#include "api/compute/experimental/deepseek_compute_kernel_hw_startup.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/pack_untilize.h"

/**
 * Streaming SDPA tail reduction that processes L tiles in chunks.
 */
template <
    bool SDPA_EXP_APPROX_MODE,
    bool normalize,
    bool untilize,
    std::uint32_t block_size,
    std::uint32_t scale_fp32,
    std::uint32_t num_l_chunks,
    VectorMode vector_mode = VectorMode::C>
ALWI void sdpa_tail_streaming(
    std::uint32_t cb_worker_max_sum,
    std::uint32_t cb_prev_max_sum,
    std::uint32_t cb_cur_max_sum,
    std::uint32_t cb_l1,
    std::uint32_t cb_l2,
    std::uint32_t cb_l_out) {
    constexpr bool dense = untilize;
    constexpr std::uint32_t total_size = num_l_chunks * block_size;
    ckernel::sdpa_tail_ms_reduce<
        SDPA_EXP_APPROX_MODE,
        normalize,
        untilize ? total_size : block_size,
        scale_fp32,
        vector_mode,
        false,
        dense>(cb_worker_max_sum, cb_prev_max_sum, cb_cur_max_sum, cb_l1);

    // TODO: Unit test perf seemed better if we operated on all chunks
    // Retest in streaming context since unit test doesn't need to wait for input
    if constexpr (untilize) {
        pack_untilize_dest_init<total_size, total_size, false, TILE_C_DIM, dense, false>(cb_l_out);
        cb_wait_front(cb_l1, total_size);
        cb_wait_front(cb_l2, total_size);
        cb_reserve_back(cb_l_out, total_size);
        ckernel::sdpa_tail_l_block<total_size, 1, untilize, dense, false>(cb_l1, cb_l2, cb_l_out, 0, 0, false);
        cb_push_back(cb_l_out, total_size);
        pack_untilize_uninit(cb_l_out);
    } else {
        bool acquire_regs = !normalize;
        pack_block_contiguous_init(cb_l_out);
        for (std::uint32_t chunk = 0; chunk < num_l_chunks; chunk++) {
            cb_wait_front(cb_l1, (chunk + 1) * block_size);
            cb_wait_front(cb_l2, (chunk + 1) * block_size);
            cb_reserve_back(cb_l_out, block_size);
            std::uint32_t tile_index = chunk * block_size;
            ckernel::sdpa_tail_l_block<block_size, 1, untilize, dense, false>(
                cb_l1, cb_l2, cb_l_out, tile_index, 0, acquire_regs);
            acquire_regs = true;
            cb_push_back(cb_l_out, block_size);
        }
    }

    // Postamble only — caller handles MS pops based on round context
    // (R1 inputs are TRISC-owned, R2 prev MS is BRISC-owned)
    ckernel::sdpa_bcast_col_reuse_postamble();
}

void kernel_main() {
    constexpr uint32_t cb_local_l = get_compile_time_arg_val(0);
    constexpr uint32_t cb_local_ms = get_compile_time_arg_val(1);
    constexpr uint32_t cb_neighbor_l = get_compile_time_arg_val(2);
    constexpr uint32_t cb_neighbor_ms = get_compile_time_arg_val(3);
    constexpr uint32_t cb_r1_result_l = get_compile_time_arg_val(4);
    constexpr uint32_t cb_r1_result_ms = get_compile_time_arg_val(5);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(6);
    constexpr uint32_t block_size = get_compile_time_arg_val(7);
    constexpr uint32_t num_l_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t twin_iters = get_compile_time_arg_val(9);
    constexpr uint32_t total_l_tiles = block_size * num_l_blocks;

    deepseek_compute_kernel_init();

    for (uint32_t it = 0; it < twin_iters; ++it) {
        constexpr VectorMode vector_mode = VectorMode::RC_custom;

        reconfig_full_operand(cb_local_l, cb_local_l);
        pack_reconfig_data_format<true>(cb_r1_result_l);
        exp_tile_init<EXP_APPROX_MODE>();

        // ROUND 1: reduce(local, r1_neighbor) -> r1_result (unnormalized)
        sdpa_tail_streaming<
            EXP_APPROX_MODE,
            false /* no normalize - R1 doesn't normalize */,
            false /* untilize - R1 doesn't untilize */,
            block_size,
            scale_fp32,
            num_l_blocks,
            vector_mode>(cb_neighbor_ms, cb_local_ms, cb_r1_result_ms, cb_neighbor_l, cb_local_l, cb_r1_result_l);

        cb_pop_front(cb_neighbor_ms, 1);
        cb_pop_front(cb_neighbor_l, total_l_tiles);
        // Popped by the worker's data movement RISCs in the op
        cb_pop_front(cb_local_ms, 1);
        cb_pop_front(cb_local_l, total_l_tiles);
    }
}
