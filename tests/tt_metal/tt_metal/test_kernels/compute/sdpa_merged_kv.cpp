// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/experimental/sdpa.h"
#include "api/compute/reconfig_data_format.h"
#include "tests/tt_metal/tt_metal/test_kernels/compute/sdpa_chunk_test_helpers.hpp"

void kernel_main() {
    constexpr std::uint32_t rounds = get_compile_time_arg_val(0);
    constexpr std::uint32_t chunks = get_compile_time_arg_val(1);
    constexpr std::uint32_t layout = get_compile_time_arg_val(2);
    constexpr std::uint32_t row_tiles = get_compile_time_arg_val(3);
    constexpr std::uint32_t v_offset = get_compile_time_arg_val(4);
    constexpr bool test_correction_fidelity = get_compile_time_arg_val(5);
    constexpr std::uint32_t corr_fidelity = test_correction_fidelity ? 4 : SDPA_FIDELITY_PROGRAM_DEFAULT;
    constexpr std::uint32_t chunk_tiles = 2;
    constexpr std::uint32_t qk_tiles = 2;
    constexpr std::uint32_t v_tiles = 2;
    constexpr std::uint32_t scale = 0x3f000000;  // 0.5f
    constexpr auto cb_q = tt::CBIndex::c_0;
    constexpr auto cb_k = tt::CBIndex::c_1;
    constexpr auto cb_v = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr auto cb_stats = tt::CBIndex::c_17;
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
        ckernel::test_helpers::init_sdpa_chunk_pack_reduce();
        exp_packthread_tile_init<true, scale>();
        sdpa_custom_mm_block_init_pack_short();
        pack_block_contiguous_init(cb_out);
        cb_wait_front(cb_q, qk_tiles);
        cb_reserve_back(cb_out, v_tiles);
        cb_reserve_back(cb_stats, 1);
        tile_regs_acquire();
        for (std::uint32_t chunk = 0; chunk < chunks; ++chunk) {
            // The original shared/separate layouts keep the API defaults.
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
                    qk_tiles,
                    corr_fidelity>(
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
                    chunk == chunks - 1,
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
                    chunk == chunks - 1,
                    false);
            }
        }
        // Emit the partial O and max/sum exactly as a distributed SDPA worker.
        // Reciprocal normalization has separate coverage; keeping it out of this
        // layout test avoids BF16 cancellation in the reciprocal-minus-one path.
        ckernel::test_helpers::pack_sdpa_chunk_partials<v_tiles>(
            max_offset / packed_tile_size, output_offset / packed_tile_size, cb_stats, cb_out);
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
        sdpa_custom_mm_block_uninit();
        ckernel::test_helpers::wait_for_sdpa_chunk_pack();
        cb_pop_front(cb_q, qk_tiles);
    }
}
