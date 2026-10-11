// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/experimental/sdpa.h"

// Compiles compute_sdpa_chunks_pipelined's semaphore and layout contracts; never executed on a device.
void kernel_main() {
    constexpr std::uint32_t chunk_size = get_compile_time_arg_val(0);
    constexpr std::uint32_t qk_granularity = get_compile_time_arg_val(1);
    constexpr std::uint32_t exp_granularity = get_compile_time_arg_val(2);
    constexpr std::uint32_t row_tiles = get_compile_time_arg_val(3);
    constexpr std::uint32_t v_tiles = get_compile_time_arg_val(4);
    constexpr bool separate_v = get_compile_time_arg_val(5);
    compute_sdpa_chunks_pipelined<
        chunk_size,
        row_tiles,
        v_tiles,
        0x3f800000,
        false,
        false,
        16,
        false,
        qk_granularity,
        exp_granularity,
        1,
        true,
        separate_v>(
        tt::CBIndex::c_0,
        tt::CBIndex::c_1,
        tt::CBIndex::c_2,
        tt::CBIndex::c_3,
        tt::CBIndex::c_16,
        0,
        128,
        256,
        288,
        304,
        320,
        2,
        true,
        true);
}
