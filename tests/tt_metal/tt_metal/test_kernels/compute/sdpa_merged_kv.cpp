// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/experimental/sdpa.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr std::uint32_t rounds = get_compile_time_arg_val(0);
    constexpr std::uint32_t chunks = get_compile_time_arg_val(1);
    constexpr std::uint32_t layout = get_compile_time_arg_val(2);
    constexpr std::uint32_t row_tiles = get_compile_time_arg_val(3);
    constexpr std::uint32_t v_offset = get_compile_time_arg_val(4);
    constexpr std::uint32_t chunk_tiles = 2;
    constexpr std::uint32_t qk_tiles = 2;
    constexpr std::uint32_t v_tiles = 2;
    constexpr std::uint32_t scale = 0x3f000000;  // 0.5f
    constexpr auto cb_q = tt::CBIndex::c_0;
    constexpr auto cb_k = tt::CBIndex::c_1;
    constexpr auto cb_v = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr std::uint32_t packed_tile_size = 16;
    constexpr std::uint32_t output_offset = 0;
    constexpr std::uint32_t max_offset = packed_tile_size * v_tiles;
    constexpr std::uint32_t sum_offset = max_offset + 2;
    constexpr std::uint32_t correction_offset = max_offset + packed_tile_size;
    constexpr std::uint32_t scores_offset = correction_offset + packed_tile_size;

    deepseek_compute_kernel_init();
    for (std::uint32_t round = 0; round < rounds; ++round) {
        reconfig_full_operand(cb_k, cb_q);
        pack_reconfig_data_format(cb_out);
        PACK((llk_math_sfpu_sdpa_reduce_row_init<false, DST_ACCUM_MODE, DataFormat::Float16_b>()));
        exp_packthread_tile_init<true, scale>();
        sdpa_custom_mm_block_init_pack_short();
        pack_block_contiguous_init(cb_out);
        cb_wait_front(cb_q, qk_tiles);
        cb_reserve_back(cb_out, v_tiles);
        tile_regs_acquire();
        for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
            // The old layouts omit both new parameters to test compatibility.
            if constexpr (layout == 2) {
                compute_sdpa_chunk<
                    chunk_tiles,
                    row_tiles,
                    v_tiles,
                    scale,
                    true,
                    false,
                    packed_tile_size,
                    false,
                    1,
                    1,
                    v_tiles,
                    false,
                    false,
                    false,
                    v_offset,
                    qk_tiles>(
                    cb_q,
                    cb_k,
                    cb_v,
                    cb_q,
                    cb_out,
                    scores_offset,
                    output_offset,
                    max_offset,
                    sum_offset,
                    correction_offset,
                    chunk == 0,
                    false,
                    false);
            } else {
                compute_sdpa_chunk<
                    chunk_tiles,
                    row_tiles,
                    v_tiles,
                    scale,
                    true,
                    false,
                    packed_tile_size,
                    false,
                    1,
                    1,
                    v_tiles,
                    false,
                    layout == 1>(
                    cb_q,
                    cb_k,
                    cb_v,
                    cb_q,
                    cb_out,
                    scores_offset,
                    output_offset,
                    max_offset,
                    sum_offset,
                    correction_offset,
                    chunk == 0,
                    false,
                    false);
            }
        }
        compute_sdpa_recip<v_tiles, false, scale, v_tiles>(cb_q, sum_offset, correction_offset, output_offset);
        PACK((t6_semaphore_wait_on_zero<p_stall::STALL_PACK>(semaphore::FPU_SFPU)));
        pack_block_contiguous(0, cb_out, v_tiles);
        PACK((t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU)));
        cb_push_back(cb_out, v_tiles);
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
        sdpa_custom_mm_block_uninit();
        MATH((t6_semaphore_wait_on_max<p_stall::STALL_SFPU>(semaphore::FPU_SFPU)));
        cb_pop_front(cb_q, qk_tiles);
    }
}
