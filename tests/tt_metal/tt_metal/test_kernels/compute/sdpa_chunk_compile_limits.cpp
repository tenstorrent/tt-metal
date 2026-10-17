// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/experimental/sdpa.h"

// Compiles semaphore, layout and argument-list contracts; never executed on a device.
// SDPA_DIRECT_CALL selects what is compiled: 0 (default) the compute_sdpa_chunk entry point;
// 1 the current direct sdpa_custom_mm_reuse_dest_srcb argument lists, which must compile;
// 2-4 legacy direct argument lists that carried a transpose argument, which must not
// compile instead of re-binding transpose as kt_dim.
#ifndef SDPA_DIRECT_CALL
#define SDPA_DIRECT_CALL 0
#endif
void kernel_main() {
    constexpr std::uint32_t chunk_size = get_compile_time_arg_val(0);
    constexpr std::uint32_t qk_granularity = get_compile_time_arg_val(1);
    constexpr std::uint32_t exp_granularity = get_compile_time_arg_val(2);
    constexpr std::uint32_t row_tiles = get_compile_time_arg_val(3);
    constexpr std::uint32_t v_tiles = get_compile_time_arg_val(4);
    constexpr std::uint32_t v_offset = get_compile_time_arg_val(5);
    constexpr std::uint32_t qk_tiles = get_compile_time_arg_val(6);
    constexpr bool separate_v = get_compile_time_arg_val(7);
    constexpr bool transpose_v = get_compile_time_arg_val(8);
#if SDPA_DIRECT_CALL == 0
    compute_sdpa_chunk<
        chunk_size,
        row_tiles,
        v_tiles,
        0x3f800000,
        false,
        transpose_v,
        16,
        false,
        qk_granularity,
        exp_granularity,
        1,
        true,
        separate_v,
        true,
        v_offset,
        qk_tiles>(
        tt::CBIndex::c_0,
        tt::CBIndex::c_1,
        tt::CBIndex::c_2,
        tt::CBIndex::c_3,
        tt::CBIndex::c_16,
        0,
        256,
        288,
        304,
        320,
        true,
        true,
        true);
#else
    constexpr std::uint32_t kt_dim = qk_tiles;
    constexpr std::uint32_t nt_dim = v_tiles;
    constexpr std::uint32_t in1_k_stride = 1;
#if SDPA_DIRECT_CALL == 1
    sdpa_custom_mm_reuse_dest_srcb_block_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16, kt_dim, nt_dim);
    sdpa_custom_mm_reuse_dest_srcb_block_init_short(
        tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16, kt_dim, nt_dim);
    sdpa_custom_mm_reuse_dest_srcb_block<1, 1>(
        tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, 0, kt_dim, nt_dim, in1_k_stride, false);
#elif SDPA_DIRECT_CALL == 2
    // Legacy init: (in0, in1, out, transpose, kt_dim).
    sdpa_custom_mm_reuse_dest_srcb_block_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16, 0, kt_dim);
#elif SDPA_DIRECT_CALL == 3
    // Legacy block ending in (transpose, kt_dim, nt_dim).
    sdpa_custom_mm_reuse_dest_srcb_block<1, 1>(tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, 0, false, kt_dim, nt_dim);
#elif SDPA_DIRECT_CALL == 4
    // Legacy block with a leading transpose and an explicit stride.
    sdpa_custom_mm_reuse_dest_srcb_block<1, 1>(
        tt::CBIndex::c_0, tt::CBIndex::c_1, 0, 0, 0, 0, true, kt_dim, nt_dim, in1_k_stride);
#endif
#endif
}
