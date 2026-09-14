// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "experimental/llk_sfpu/ckernel_sfpu_sdpa_reduce_row.h"
#include "sanitizer/api.h"

namespace ckernel {

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, DataFormat format>
inline void llk_math_sfpu_sdpa_reduce_row_init() {
    SAN_HOOK(unsupported());
    sfpu::init_sdpa_reduce_row<format>();
}

template <
    bool APPROXIMATE,
    bool is_fp32_dest_acc_en,
    DataFormat format,
    std::uint32_t block_width,
    bool skip_signalling = false,
    std::uint32_t signal_granularity = 1>
inline void llk_math_sfpu_sdpa_reduce_max_row(std::uint32_t src_index, std::uint32_t dst_index, bool prev_max = false) {
    SAN_HOOK(unsupported());
    sfpu::calculate_sdpa_reduce_max_row<format, block_width, skip_signalling, signal_granularity>(
        src_index, dst_index, prev_max);
}

template <
    bool APPROXIMATE,
    bool is_fp32_dest_acc_en,
    DataFormat format,
    std::uint32_t block_width,
    bool skip_signalling = false>
inline void llk_math_sfpu_sdpa_reduce_sum_row(std::uint32_t src_index, std::uint32_t dst_index, bool prev_sum = false) {
    SAN_HOOK(unsupported());
    sfpu::calculate_sdpa_reduce_sum_row<format, block_width, skip_signalling>(src_index, dst_index, prev_sum);
}

// Signal that SFPU work for the chunk is complete. Waits for the FPU's post
// (p_stall::NONE — never blocks SFPU logic; only ensures the semaphore is
// non-zero before decrementing), then decrements so the QK matmul can reuse
// the space in the next iteration.
inline void llk_math_sdpa_sfpu_signal_chunk_done() {
    SAN_HOOK(unsupported());
    t6_semaphore_wait_on_zero<p_stall::NONE>(semaphore::FPU_SFPU);
    t6_semaphore_get<p_stall::WAIT_SFPU>(semaphore::FPU_SFPU);
}

}  // namespace ckernel
