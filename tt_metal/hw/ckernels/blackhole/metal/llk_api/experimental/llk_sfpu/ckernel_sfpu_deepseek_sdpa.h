// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
#include "ckernel_sfpu_exp.h"
#include "ckernel_sfpu_recip.h"

namespace ckernel::sfpu {

/**
 * Fused max-sub-exp-add SFPI kernel for SDPA tail reduction.
 *
 * Computes:
 *   cur_max = max(prev_max, worker_max)
 *   cur_sum = exp((worker_max - cur_max) * scale) * worker_sum
 *           + exp((prev_max - cur_max) * scale) * prev_sum
 *
 * Results (final_norm=false):
 *   dst_reg[prev_max_base_idx]   = exp((prev_max - cur_max) * scale)
 *   dst_reg[worker_max_base_idx] = exp((worker_max - cur_max) * scale)
 *   dst_reg[cur_sum_base_idx]    = cur_sum
 *   dst_reg[cur_max_base_idx]    = cur_max
 *
 * Results (final_norm=true):
 *   dst_reg[prev_max_base_idx]   = exp((prev_max - cur_max) * scale) * recip(cur_sum)
 *   dst_reg[worker_max_base_idx] = exp((worker_max - cur_max) * scale) * recip(cur_sum)
 */
template <bool SDPA_EXP_APPROX_MODE, bool final_norm = false>
inline void calculate_fused_max_sub_exp_add_tile(int scale_bf16) {
    static_assert(!(final_norm && SDPA_EXP_APPROX_MODE), "Approx mode must be disabled when final_norm is true");

    constexpr int ITERATIONS_HALF_FACE = 2;
    constexpr std::uint32_t prev_max_base_idx = 0;
    constexpr std::uint32_t prev_sum_base_idx = 1;
    constexpr std::uint32_t worker_max_base_idx = 32;
    constexpr std::uint32_t worker_sum_base_idx = 33;
    constexpr std::uint32_t cur_max_base_idx = 64;
    constexpr std::uint32_t cur_sum_base_idx = 65;

    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
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

        sfpi::vFloat diff_prev = prev_max_vec - cur_max;
        sfpi::vFloat diff_worker = worker_max_vec - cur_max;

        sfpi::vFloat exp_prev =
            ckernel::sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
                diff_prev, scale_bf16);
        sfpi::vFloat exp_worker =
            ckernel::sfpu::_ckernel_sfpu_exp_accurate_<true /*SCALE_EN*/, DST_ACCUM_MODE /*is_fp32_dest_acc_en*/>(
                diff_worker, scale_bf16);

        if constexpr (!final_norm) {
            sfpi::dst_reg[cur_sum_base_idx] = exp_worker * worker_sum_vec + exp_prev * prev_sum_vec;
            sfpi::dst_reg[prev_max_base_idx] = exp_prev;
            sfpi::dst_reg[worker_max_base_idx] = exp_worker;
        } else {
            sfpi::vFloat curr_sum = exp_worker * worker_sum_vec + exp_prev * prev_sum_vec;
            ckernel::sfpu::sfpu_reciprocal_init<false>();
            sfpi::vFloat recip_sum = ckernel::sfpu::sfpu_reciprocal<SDPA_EXP_APPROX_MODE>(curr_sum);
            sfpi::dst_reg[prev_max_base_idx] = exp_prev * recip_sum;
            sfpi::dst_reg[worker_max_base_idx] = exp_worker * recip_sum;
        }
        sfpi::dst_reg += 2;
    }
}

}  // namespace ckernel::sfpu
