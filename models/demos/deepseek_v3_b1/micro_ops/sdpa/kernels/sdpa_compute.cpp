// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

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
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa_custom_mm.h"
#include "../../../kernel_includes/tt_metal/include/compute_kernel_api/sdpa_custom_mm_reuse_dest_srcb.h"
#include "../../flash_mla/kernels/compute/compute_common.hpp"

#ifdef TRISC_PACK
#include "../../../kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/llk_math_sdpa_reduce_row.h"
#include "ckernel_sfpu_exp.h"
#include "llk_math_eltwise_unary_sfpu_macros.h"
#endif

void kernel_main() {
    // CB indices passed as compile-time args
    constexpr uint32_t cb_q = get_compile_time_arg_val(0);             // q input
    constexpr uint32_t cb_k = get_compile_time_arg_val(1);             // k input
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);           // output CB
    constexpr uint32_t cb_stats = get_compile_time_arg_val(3);         // stats CB
    constexpr uint32_t chunk_size = get_compile_time_arg_val(4);       // chunk size
    constexpr uint32_t num_chunks = get_compile_time_arg_val(5);       // number of chunks
    constexpr uint32_t num_tiles_k = get_compile_time_arg_val(6);      // number of tiles in k
    constexpr uint32_t num_tiles_v = get_compile_time_arg_val(7);      // number of tiles in v
    constexpr uint32_t num_tiles_stats = get_compile_time_arg_val(8);  // number of tiles in stats
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(9);       // scale factor in FP32 uint32 representation
    static_assert(num_tiles_stats == 1, "num_tiles_stats must be 1");
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    constexpr bool transpose_k = true;
    constexpr bool transpose_v = false;

    // 8 rows per face, 2 faces per tile
    // constexpr uint32_t max_dst_offset = 8 * 2 * chunk_size;
    constexpr uint32_t packed_tile_size = 8 * 2;
    constexpr uint32_t mm2_dst_offset = 0;
    constexpr uint32_t mm2_dst_tile_offset = mm2_dst_offset / packed_tile_size;
    constexpr uint32_t max_dst_offset = mm2_dst_offset + packed_tile_size * num_tiles_v;
    constexpr uint32_t max_dst_tile_offset = max_dst_offset / packed_tile_size;
    // Second col in the tile containing max
    constexpr uint32_t sum_dst_offset = max_dst_offset + 2;
    // Next tile after max/sum
    constexpr uint32_t corr_exp_dst_offset = max_dst_offset + packed_tile_size;
    constexpr uint32_t mm1_dst_offset = corr_exp_dst_offset + packed_tile_size;
    constexpr uint32_t mm1_dst_tile_offset = mm1_dst_offset / packed_tile_size;

    constexpr bool exp_approx_mode = false;

    PACK((llk_math_sfpu_sdpa_reduce_row_init<false, DST_ACCUM_MODE, DataFormat::Float16_b>()));
    PACK(SFPU_TEMPLATE_INIT_KERNEL(exponential, sfpu::exp_init, true, true, scale_fp32, true));
    sdpa_custom_mm_block_init<transpose_k>(cb_q, cb_k, cb_out, chunk_size);

    // TODO: Init ahead of time
    MATH(ckernel::t6_semaphore_init(ckernel::semaphore::FPU_SFPU, 0, 1));
    PACK(ckernel::t6_semaphore_init(SFPU_FPU, 0, 1));

    cb_wait_front(cb_q, num_tiles_k);
    cb_reserve_back(cb_out, num_tiles_v);
    cb_reserve_back(cb_stats, num_tiles_stats);
    tile_regs_acquire();
    for (uint32_t chunk = 0; chunk < num_chunks; chunk++) {
        compute_sdpa_chunk<
            chunk_size,
            num_tiles_k,
            num_tiles_v,
            scale_fp32,
            scale_bf16,
            transpose_k,
            transpose_v,
            packed_tile_size,
            exp_approx_mode>(
            cb_q,
            cb_k,
            cb_out,
            mm1_dst_offset,
            mm2_dst_offset,
            max_dst_offset,
            sum_dst_offset,
            corr_exp_dst_offset,
            chunk == 0,
            chunk == num_chunks - 1);
    }

    // Sem is incremented once per 2 tiles since sem can only go up to 15
    for (uint32_t i = 0; i < num_tiles_v; i += 2) {
        PACK(t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU));
        pack_tile(mm2_dst_tile_offset + i, cb_out);
        pack_tile(mm2_dst_tile_offset + i + 1, cb_out);
        PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));
    }
    // Stall for Red Sum to finish
    PACK(TTI_STALLWAIT(p_stall::STALL_PACK, p_stall::WAIT_SFPU));
    pack_tile(max_dst_tile_offset, cb_stats);

    // Validation that all counters are reset to 0
    // MATH(t6_semaphore_wait_on_max<p_stall::STALL_MATH>(SFPU_FPU));
    // MATH(t6_semaphore_wait_on_max<p_stall::STALL_MATH>(semaphore::FPU_SFPU));
    // PACK(t6_semaphore_wait_on_max<p_stall::STALL_PACK>(SFPU_FPU));
    // PACK(t6_semaphore_wait_on_max<p_stall::STALL_PACK>(semaphore::FPU_SFPU));

    cb_push_back(cb_out, num_tiles_v);
    cb_push_back(cb_stats, num_tiles_stats);
    cb_pop_front(cb_q, num_tiles_k);
    tile_regs_commit();
    // tile_regs_wait();
    tile_regs_release();
    sdpa_custom_mm_block_uninit();
}
