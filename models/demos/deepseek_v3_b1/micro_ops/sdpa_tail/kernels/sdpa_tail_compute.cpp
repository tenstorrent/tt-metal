// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#define EXP_APPROX_MODE false

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h"

#ifdef TRISC_MATH
/**
 * The custom SFPI LLK function computes the following operation:
 * cur_max = max(prev_max, worker_max)
 * cur_sum = exp((worker_max - cur_max) * scale) * worker_sum + exp((prev_max - cur_max) * scale) * prev_sum
 * There are 4 results produced:
 * 1. exp_max_diff = exp((worker_max - cur_max) * scale), produced in dst_reg[prev_max_base_idx]
 * 2. exp_max_diff_2 = exp((prev_max - cur_max) * scale), produced in dst_reg[worker_max_base_idx]
 * 3. cur_sum produced in dst_reg[prev_sum_base_idx]
 * 4. cur_max produced in dst_reg[cur_max_base_idx]
 * If final_norm is true, the output is:
 * 1. exp_max_diff = exp((worker_max - cur_max) * scale) * recip(cur_sum), produced in dst_reg[prev_max_base_idx]
 * 2. exp_max_diff_2 = exp((prev_max - cur_max) * scale) * recip(cur_sum), produced in dst_reg[worker_max_base_idx]
 * fused_max_sub_exp_add_tile
 */
template <bool SDPA_EXP_APPROX_MODE, bool final_norm = false>
void calculate_fused_max_sub_exp_add_tile(int scale_bf16) {
    // Non-Approx mode for exp initializes recip for final normalization
    static_assert(!(final_norm && SDPA_EXP_APPROX_MODE), "Approx mode must be disabled when final_norm is true");
    // 8 rows
    constexpr int ITERATIONS_HALF_FACE = 2;
    constexpr uint32_t prev_max_base_idx = 0;     // dst_reg_0 (Tile 0 first col)
    constexpr uint32_t prev_sum_base_idx = 1;     // dst_reg_3 (Tile 0 second col)
    constexpr uint32_t worker_max_base_idx = 32;  // dst_reg_1 (Tile 1 first col)
    constexpr uint32_t worker_sum_base_idx = 33;  // dst_reg_4 (Tile 1 second col)
    constexpr uint32_t cur_max_base_idx = 64;     // dst_reg_2 (Tile 2 first col)
    constexpr uint32_t cur_sum_base_idx = 65;     // dst_reg_3 (Tile 2 second col)

    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        // Load inputs for this vector-slot into temporaries to avoid aliasing on dst_reg
        sfpi::vFloat prev_max_vec = sfpi::dst_reg[prev_max_base_idx];
        sfpi::vFloat worker_max_vec = sfpi::dst_reg[worker_max_base_idx];
        sfpi::vFloat prev_sum_vec = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::vFloat worker_sum_vec = sfpi::dst_reg[worker_sum_base_idx];
        sfpi::vFloat cur_max;
        v_if(prev_max_vec < worker_max_vec) { cur_max = worker_max_vec; }
        v_else { cur_max = prev_max_vec; }
        v_endif;
        if constexpr (!final_norm) {
            sfpi::dst_reg[cur_max_base_idx] = cur_max;
        }

        // Compute differences
        sfpi::vFloat diff_prev = prev_max_vec - cur_max;
        sfpi::vFloat diff_worker = worker_max_vec - cur_max;

        // Exponentials of differences
        sfpi::vFloat exp_prev = ckernel::sfpu::
            _calculate_exponential_piecewise_<EXP_APPROX_MODE, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
                diff_prev, scale_bf16);
        sfpi::vFloat exp_worker = ckernel::sfpu::
            _calculate_exponential_piecewise_<EXP_APPROX_MODE, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
                diff_worker, scale_bf16);

        if constexpr (!final_norm) {
            sfpi::dst_reg[cur_sum_base_idx] = exp_worker * worker_sum_vec + exp_prev * prev_sum_vec;
            sfpi::dst_reg[prev_max_base_idx] = exp_prev;
            sfpi::dst_reg[worker_max_base_idx] = exp_worker;
        } else {
            sfpi::vFloat curr_sum = exp_worker * worker_sum_vec + exp_prev * prev_sum_vec;
            sfpi::vFloat recip_sum = ckernel::sfpu::sfpu_reciprocal<SDPA_EXP_APPROX_MODE>(curr_sum);
            sfpi::dst_reg[prev_max_base_idx] = exp_prev * recip_sum;
            sfpi::dst_reg[worker_max_base_idx] = exp_worker * recip_sum;
        }
        sfpi::dst_reg += 2;
    }
}

template <bool SDPA_EXP_APPROX_MODE, int vector_mode = (int)VectorMode::C, bool final_norm = false>
void fused_max_sub_exp_add_tile(uint32_t idst, int scale_bf16) {
    _llk_math_eltwise_unary_sfpu_params_<false /*APPROXIMATE*/>(
        calculate_fused_max_sub_exp_add_tile<SDPA_EXP_APPROX_MODE, final_norm>, idst, vector_mode, scale_bf16);
}
#endif

template <
    bool normalize,
    uint32_t block_size,
    uint32_t num_blocks,
    uint32_t scale_fp32,
    int vector_mode = (int)VectorMode::C>
void sdpa_tail(
    uint32_t cb_worker_max_sum,
    uint32_t cb_prev_max_sum,
    uint32_t cb_cur_max_sum,
    uint32_t cb_l1,
    uint32_t cb_l2,
    uint32_t cb_l_out) {
    copy_tile_to_dst_init_short(cb_worker_max_sum);

    cb_wait_front(cb_worker_max_sum, 1);
    cb_wait_front(cb_prev_max_sum, 1);

    constexpr uint32_t dst_reg_0 = 0;  // dst_reg_0 is used for prev_max_sum
    constexpr uint32_t dst_reg_1 = 1;  // dst_reg_1 is used for worker_max_sum
    constexpr uint32_t dst_reg_2 = 2;  // dst_reg_2 is used for cur_max_sum

    // convert scale from fp32 to bf16
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    tile_regs_acquire();
    copy_tile(cb_prev_max_sum, 0, dst_reg_0);
    copy_tile(cb_worker_max_sum, 0, dst_reg_1);
    MATH((fused_max_sub_exp_add_tile<EXP_APPROX_MODE, vector_mode, normalize>(0, scale_bf16)));
    sdpa_bcast_col_reuse_preamble<normalize>();

    // Not final reduction, pack out stats
    if constexpr (!normalize) {
        tile_regs_commit();
        cb_reserve_back(cb_cur_max_sum, 1);
        tile_regs_wait();
        pack_tile(dst_reg_2, cb_cur_max_sum);
        cb_push_back(cb_cur_max_sum, 1);
        tile_regs_release();
    }
    sdpa_mul_bcast_col_reuse_tiles_init<block_size>(cb_l1);

    cb_wait_front(cb_l2, num_blocks * block_size);
    cb_wait_front(cb_l1, num_blocks * block_size);
    cb_reserve_back(cb_l_out, num_blocks * block_size);

    // Now compute l = l1 * P1 + l2 * P2
    // Following sdpa_flash_decode.cpp pattern:
    //   l2 *= P2 (cb_exp_diff_1)
    //   l1 *= P1 (cb_exp_diff_2)
    //   l = l2 + l1
    // Final reduction, we compute the first iteration without spilling
    if constexpr (normalize) {
        sdpa_mul_bcast_col_reuse_tiles<block_size>(cb_l2, cb_l1, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile_block(0, cb_l_out, block_size);
        tile_regs_release();
    }
    for (uint32_t i = (normalize ? 1 : 0); i < num_blocks; i++) {
        tile_regs_acquire();
        sdpa_mul_bcast_col_reuse_tiles<block_size>(cb_l2, cb_l1, i * block_size, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile_block(0, cb_l_out, block_size);
        tile_regs_release();
    }
    sdpa_bcast_col_reuse_postamble();
    cb_push_back(cb_l_out, num_blocks * block_size);
    cb_pop_front(cb_prev_max_sum, 1);
    cb_pop_front(cb_worker_max_sum, 1);
    cb_pop_front(cb_l2, num_blocks * block_size);
    cb_pop_front(cb_l1, num_blocks * block_size);
}

void kernel_main() {
    // CB indices passed as compile-time args
    constexpr uint32_t cb_l1 = get_compile_time_arg_val(0);       // l1 input
    constexpr uint32_t cb_l2 = get_compile_time_arg_val(1);       // l2 input
    constexpr uint32_t cb_ms1 = get_compile_time_arg_val(2);      // ms1 input (worker max)
    constexpr uint32_t cb_ms2 = get_compile_time_arg_val(3);      // ms2 input (prev max)
    constexpr uint32_t cb_l_out = get_compile_time_arg_val(4);    // l output
    constexpr uint32_t cb_ms_out = get_compile_time_arg_val(5);   // ms output (cur max)
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(6);  // scale as fp32 bits
    constexpr uint32_t block_size = get_compile_time_arg_val(7);  // number of row tiles
    constexpr uint32_t num_blocks = get_compile_time_arg_val(8);  // number of column tiles
    constexpr bool final_reduction = get_compile_time_arg_val(9);

    constexpr int vector_mode = VectorMode::RC_custom;

    binary_op_init_common(cb_l1, cb_l1, cb_l_out);
    exp_tile_init<EXP_APPROX_MODE, false>();

    // correction_block computes:
    // 1. cb_m_out = max(cb_m2, cb_m1)
    // 2. cb_exp_diff_2 = exp((cb_m1 - cb_m_out) * scale)  [P1]
    // 3. cb_s1 *= cb_exp_diff_2  (s1 * P1)
    // 4. cb_exp_diff_1 = exp((cb_m2 - cb_m_out) * scale)  [P2]
    // 5. cb_s2 *= cb_exp_diff_1  (s2 * P2)
    // 6. cb_s_out = cb_s1 + cb_s2  (s1*P1 + s2*P2)
    // 7. cb_l_out = cb_l1 * P1 + cb_l2 * P2
    sdpa_tail<final_reduction, block_size, num_blocks, scale_fp32, vector_mode>(
        cb_ms1,     // worker max (ms1)
        cb_ms2,     // prev max (m2)
        cb_ms_out,  // cur max output (m = max(m1, m2))
        cb_l1,      // l1 input
        cb_l2,      // l2 input
        cb_l_out    // l output
    );
}
