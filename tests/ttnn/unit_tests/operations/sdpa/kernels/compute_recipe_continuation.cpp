// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#if defined(WATCHER_ENABLED)
#pragma GCC optimize("Os")
#elif defined(SDPA_RECIPE_ACCURATE)
#pragma GCC optimize("O2")
#endif

#include "api/compute/compute_kernel_hw_startup.h"
#ifdef SDPA_RECIPE_LOFI
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/streaming/lofi_scaling.hpp"
#endif
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/streaming/recipe_sfpu.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/streaming/recipe_streaming.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/streaming/recipe_checkpoint.hpp"

void kernel_main() {
    constexpr uint32_t chunks = get_compile_time_arg_val(0);
    constexpr uint32_t scale = get_compile_time_arg_val(1);
    constexpr uint32_t split = get_compile_time_arg_val(3);
    constexpr bool reload_q = get_compile_time_arg_val(4);
    constexpr bool stage = get_compile_time_arg_val(5);
    static_assert(split > 0 && split < chunks);
    compute_kernel_hw_startup<SrcOrder::Reverse>(0, 1, 16);
    matmul_init(0, 1);
    cb_wait_front(0, 32);
    cb_wait_front(3, 1);
    cb_wait_front(4, 1);
    init_sdpa_streaming_semaphores();
    RecipeAccumulatorState state{{12, 10, 8}, {13, 11, 9}};
    auto segment = [&](RecipeAccumulatorState& accumulator, uint32_t count, bool final, bool release) {
        sdpa_segment_v2<
            8,
            16,
            4,
            4,
            scale,
#ifdef SDPA_RECIPE_FP32
            1,
            4,
            1,
            4,
#else
            2,
            4,
            2,
            4,
#endif
            0,
            1,
            2,
            6,
            3,
            14,
            4,
            5,
            16,
            true>(accumulator, count, final, release);
    };
    if constexpr (stage) {
        segment(state, split, false, true);
        recipe_checkpoint<17, 18>(state, 0, false);
        RecipeAccumulatorState other{{12, 10, 8}, {13, 11, 9}};
        segment(other, chunks, true, true);
        state = {{12, 10, 8}, {13, 11, 9}};
        recipe_checkpoint<17, 18>(state, 0, true);
        segment(state, chunks - split, true, true);
    } else {
        segment(state, split, false, reload_q);
        segment(state, chunks - split, true, true);
    }
}
