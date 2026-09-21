// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reuse the frozen numerical implementation; enable its native partial-K mask.
// Mask integration revision 2: explicitly restore BF16 unpack for the palette.
#define kernel_main unmasked_compute_main
#include "experiments/sdpa-l2/bfp4-lofi-v2/fullchip/compute.cpp"
#undef kernel_main

void kernel_main() {
    constexpr uint32_t k_chunks = get_compile_time_arg_val(0);
    constexpr uint32_t scale = get_compile_time_arg_val(1);
    constexpr uint32_t q_tiles = get_compile_time_arg_val(2);
    const uint32_t jobs = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_init(0, 1);
    cb_wait_front(0, q_tiles * 4);
    cb_wait_front(3, 1);
    cb_wait_front(4, 1);
    cb_wait_front(15, 2);
    LightweightMaskContext mask{};
    mask.global_n_partial_col = SDPA_K_PARTIAL_COL;
    mask.global_n_partial_tile_idx = 1;
    sdpa_standard_v2<
        q_tiles, SDPA_K_CHUNK_TILES, SDPA_K_CHUNK_TILES * k_chunks, 4, 4, scale,
#ifdef SDPA_FP32_STREAMING
        1, 4, 1, 4,
#else
        2, 4, 2, 4,
#endif
        true, 0, 1, 2, 6, 3, 14, 4, 5, 16, 15>(jobs, k_chunks, 8, 9, 10, 11, 12, 13, 0, 0, mask);
}
