// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#define EXP_APPROX_MODE false

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
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h"

// =============================================================================
// Compile-time arguments
// =============================================================================
static constexpr uint32_t cb_local_l = get_compile_time_arg_val(0);
static constexpr uint32_t cb_local_ms = get_compile_time_arg_val(1);
static constexpr uint32_t cb_r1_neighbor_l = get_compile_time_arg_val(2);
static constexpr uint32_t cb_r1_neighbor_ms = get_compile_time_arg_val(3);
static constexpr uint32_t cb_r1_result_l = get_compile_time_arg_val(4);
static constexpr uint32_t cb_r1_result_ms = get_compile_time_arg_val(5);
static constexpr uint32_t cb_r2_neighbor_l = get_compile_time_arg_val(6);
static constexpr uint32_t cb_r2_neighbor_ms = get_compile_time_arg_val(7);
static constexpr uint32_t cb_l_out = get_compile_time_arg_val(8);
static constexpr uint32_t cb_ms_out = get_compile_time_arg_val(9);  // Only used for R1 intermediate

// Compute parameters
static constexpr uint32_t scale_fp32 = get_compile_time_arg_val(10);
static constexpr uint32_t tiles_per_l_chunk = get_compile_time_arg_val(11);  // tiles per L chunk (max 8)
static constexpr uint32_t num_l_chunks = get_compile_time_arg_val(12);       // number of L chunks

// Position mask parameters (for conditional reduction)
static constexpr uint32_t cb_position = get_compile_time_arg_val(13);
static constexpr uint32_t position_enabled = get_compile_time_arg_val(14);
static constexpr bool final_reduction = get_compile_time_arg_val(15);

// SDPA uses "block_size" terminology - alias for clarity
static constexpr uint32_t block_size = tiles_per_l_chunk;

/**
 * Streaming SDPA tail reduction that processes L tiles in chunks.
 *
 * This function implements the same reduction as sdpa_tail but processes
 * L tiles in chunks to enable streaming from fabric transfer.
 *
 * Flow:
 * 1. MS reduction: computes P1/P2, sets up SRCB
 * 2. For each chunk (chunk = block, since block_size = tiles_per_chunk):
 *    - Cumulative wait for L tiles (ensures chunk i data is available)
 *    - Process L tiles using P1/P2 from SRCB
 *    - Push output tiles (cb_push_back)
 *    - NO pop of input L CBs (allows multiple readers: compute + writer)
 * 3. Finalize: postamble + pop MS tiles
 *
 * Key design: L input CBs are NOT popped. This allows:
 * - Writer to read R1 result L tiles for R2 send (cumulative waits)
 * - Compute R2 to read R1 result L tiles (global indexing)
 *
 * Note: block_size = tiles_per_chunk (each chunk = one SDPA block)
 */
template <
    bool SDPA_EXP_APPROX_MODE,
    bool normalize,
    uint32_t block_size,
    uint32_t scale_fp32,
    uint32_t num_l_chunks,
    int vector_mode = (int)VectorMode::C>
ALWI void sdpa_tail_streaming(
    uint32_t cb_worker_max_sum,
    uint32_t cb_prev_max_sum,
    uint32_t cb_cur_max_sum,
    uint32_t cb_l1,
    uint32_t cb_l2,
    uint32_t cb_l_out) {
    // Phase 1: MS reduction - computes P1/P2, sets up SRCB
    // This also reserves regs if normalize=true for first L block
    ckernel::sdpa_tail_ms_reduce<SDPA_EXP_APPROX_MODE, normalize, block_size, scale_fp32, vector_mode>(
        cb_worker_max_sum, cb_prev_max_sum, cb_cur_max_sum, cb_l1);

    // Phase 2: Process L chunks as they arrive
    // Each chunk = one block (block_size = tiles_per_chunk)
    for (uint32_t chunk = 0; chunk < num_l_chunks; chunk++) {
        // Cumulative wait: ensure at least (chunk+1) * block_size tiles are available
        cb_wait_front(cb_l1, (chunk + 1) * block_size);
        cb_wait_front(cb_l2, (chunk + 1) * block_size);
        cb_reserve_back(cb_l_out, block_size);

        // Process this chunk (one block)
        uint32_t tile_index = chunk * block_size;
        // For normalize=true, first chunk uses regs still held from MS phase
        bool acquire_regs = !(normalize && chunk == 0);
        ckernel::sdpa_tail_l_block<block_size>(cb_l1, cb_l2, cb_l_out, tile_index, acquire_regs);

        cb_push_back(cb_l_out, block_size);
        // NOTE: No cb_pop_front for cb_l1/cb_l2 - tiles remain for other readers
    }

    // Phase 3: Finalize (postamble + pop MS)
    ckernel::sdpa_tail_finalize(cb_worker_max_sum, cb_prev_max_sum);
}

ALWI void forward_data(
    uint32_t cb_prev_max_sum,
    uint32_t cb_cur_max_sum,
    uint32_t num_l_chunks,
    uint32_t cb_l1,
    uint32_t cb_l_out,
    uint32_t block_size) {
    cb_wait_front(cb_prev_max_sum, 1);
    cb_reserve_back(cb_cur_max_sum, 1);

    tile_regs_acquire();
    copy_tile_init(cb_prev_max_sum);
    copy_tile(cb_prev_max_sum, 0, 0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, cb_cur_max_sum);
    tile_regs_release();

    cb_push_back(cb_cur_max_sum, 1);

    // Copy neighbor L to output L
    for (uint32_t chunk = 0; chunk < num_l_chunks; chunk++) {
        cb_wait_front(cb_l1, (chunk + 1) * block_size);
        cb_reserve_back(cb_l_out, block_size);

        uint32_t tile_index = chunk * block_size;
        for (uint32_t i = 0; i < block_size; i++) {
            tile_regs_acquire();
            copy_tile_init(cb_l1);
            copy_tile(cb_l1, tile_index + i, i);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(i, cb_l_out);
            tile_regs_release();
        }

        cb_push_back(cb_l_out, block_size);
    }
}
template <
    bool SDPA_EXP_APPROX_MODE,
    bool normalize,
    uint32_t block_size,
    uint32_t scale_fp32,
    uint32_t num_l_chunks,
    int vector_mode = (int)VectorMode::C>
ALWI void sdpa_tail_streaming_conditional(
    uint32_t cb_worker_max_sum,
    uint32_t cb_prev_max_sum,
    uint32_t cb_cur_max_sum,
    uint32_t cb_l1,
    uint32_t cb_l2,
    uint32_t cb_l_out,
    bool neighbor_valid,
    bool local_valid) {
    // Only local valid - copy local data to output
    if (!neighbor_valid && local_valid) {
        if constexpr (normalize) {
            // Normalize local data: pass same CB twice to trigger single-input path
            sdpa_tail_streaming<SDPA_EXP_APPROX_MODE, normalize, block_size, scale_fp32, num_l_chunks, vector_mode>(
                cb_prev_max_sum, cb_prev_max_sum, cb_cur_max_sum, cb_l2, cb_l2, cb_l_out);
        } else {
            // No normalization - just copy
            forward_data(cb_prev_max_sum, cb_cur_max_sum, num_l_chunks, cb_l2, cb_l_out, block_size);
        }
        return;
    }

    // Only neighbor valid - copy neighbor data to output
    if (neighbor_valid && !local_valid) {
        if constexpr (normalize) {
            // Normalize neighbor data: pass same CB twice to trigger single-input path
            sdpa_tail_streaming<SDPA_EXP_APPROX_MODE, normalize, block_size, scale_fp32, num_l_chunks, vector_mode>(
                cb_worker_max_sum, cb_worker_max_sum, cb_cur_max_sum, cb_l1, cb_l1, cb_l_out);
        } else {
            // No normalization - just copy
            forward_data(cb_worker_max_sum, cb_cur_max_sum, num_l_chunks, cb_l1, cb_l_out, block_size);
        }
        return;
    }

    // Both valid - perform normal SDPA reduction
    sdpa_tail_streaming<SDPA_EXP_APPROX_MODE, normalize, block_size, scale_fp32, num_l_chunks, vector_mode>(
        cb_worker_max_sum, cb_prev_max_sum, cb_cur_max_sum, cb_l1, cb_l2, cb_l_out);
}

void kernel_main() {
    constexpr int vector_mode = VectorMode::RC_custom;

    binary_op_init_common(cb_local_l, cb_local_l, cb_l_out);
    exp_tile_init<EXP_APPROX_MODE, false>();

    // Runtime args: device indices for position lookup (when position_enabled)
    bool local_valid = true;
    bool r1_neighbor_valid = true;
    bool r2_neighbor_valid = true;

    uint32_t device_idx = 0;
    uint32_t r1_neighbor_device_idx = 0;
    uint32_t r2_neighbor_device_idx = 0;

    if constexpr (position_enabled) {
        // Read device indices from runtime args
        size_t arg_idx = 0;
        device_idx = get_arg_val<uint32_t>(arg_idx++);
        r1_neighbor_device_idx = get_arg_val<uint32_t>(arg_idx++);
        r2_neighbor_device_idx = get_arg_val<uint32_t>(arg_idx++);
        uint32_t r2_neighbor_r1_neighbor_idx = get_arg_val<uint32_t>(arg_idx++);

        cb_wait_front(cb_position, 1);
        uint32_t local_val = read_tile_value(cb_position, 0, device_idx);
        uint32_t r1_val = read_tile_value(cb_position, 0, r1_neighbor_device_idx);
        uint32_t r2_val = read_tile_value(cb_position, 0, r2_neighbor_device_idx);
        uint32_t r2_neighbor_r1_val = read_tile_value(cb_position, 0, r2_neighbor_r1_neighbor_idx);
        cb_pop_front(cb_position, 1);

        local_valid = (local_val != 0);
        r1_neighbor_valid = (r1_val != 0);
        r2_neighbor_valid =
            (r2_val != 0) || (r2_neighbor_r1_val != 0);  // R2 neighbor's R1 result is valid if either device was valid
    }

    // =========================================================================
    // ROUND 1: reduce(local, r1_neighbor) → r1_result (unnormalized)
    // =========================================================================
    sdpa_tail_streaming_conditional<
        EXP_APPROX_MODE,
        false /* no normalize - R1 doesn't normalize */,
        block_size,
        scale_fp32,
        num_l_chunks,
        vector_mode>(
        cb_r1_neighbor_ms,  // worker (neighbor)
        cb_local_ms,        // prev (local)
        cb_r1_result_ms,    // cur output MS
        cb_r1_neighbor_l,   // l1 (neighbor)
        cb_local_l,         // l2 (local)
        cb_r1_result_l,     // l_out
        r1_neighbor_valid,
        local_valid);

    // =========================================================================
    // ROUND 2: reduce(r1_result, r2_neighbor) → final output (normalized L)
    // =========================================================================
    // Compute R1 result validity: valid if at least one device in the pair was valid
    bool local_r1_valid = local_valid || r1_neighbor_valid;
    bool r2_neighbor_r1_valid = r2_neighbor_valid;

    sdpa_tail_streaming_conditional<
        EXP_APPROX_MODE,
        final_reduction, /* don't normalize if data only on a single device */
        block_size,
        scale_fp32,
        num_l_chunks,
        vector_mode>(
        cb_r2_neighbor_ms,     // worker (neighbor's R1 result)
        cb_r1_result_ms,       // prev (local R1 result)
        cb_ms_out,             // cur output MS
        cb_r2_neighbor_l,      // l1 (neighbor's R1 result)
        cb_r1_result_l,        // l2 (local R1 result)
        cb_l_out,              // l_out
        r2_neighbor_r1_valid,  // R2 neighbor R1 result validity
        local_r1_valid);       // Local R1 result validity
}
