// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/experimental/sdpa.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr std::uint32_t rounds = get_compile_time_arg_val(0);
    constexpr std::uint32_t block_size = get_compile_time_arg_val(1);
    constexpr std::uint32_t num_blocks = get_compile_time_arg_val(2);
    constexpr bool normalize = get_compile_time_arg_val(3);
    constexpr bool untilize = get_compile_time_arg_val(4);
    constexpr bool dense = untilize;
    constexpr std::uint32_t scale = 0x3f000000;  // 0.5f
    constexpr auto worker_ms = tt::CBIndex::c_0;
    constexpr auto previous_ms = tt::CBIndex::c_1;
    constexpr auto worker_l = tt::CBIndex::c_2;
    constexpr auto previous_l = tt::CBIndex::c_3;
    constexpr auto output_l = tt::CBIndex::c_16;
    constexpr auto output_ms = tt::CBIndex::c_17;

    compute_kernel_hw_startup(worker_ms, output_l);
    // The fused Blaze producer enables the Blackhole source-register remap.
    // Untilize must preserve it while changing only the PACK configuration.
    MATH((llk_math_reconfig_remap(true)));
    for (std::uint32_t round = 0; round < rounds; ++round) {
        reconfig_full_operand_srca(worker_ms);
        pack_reconfig_data_format(output_l);
        exp_tile_init<false, scale>();
        sdpa_tail<false, normalize, block_size, num_blocks, scale, VectorMode::C, dense, untilize, true>(
            worker_ms, previous_ms, output_ms, worker_l, previous_l, output_l);
    }
}
