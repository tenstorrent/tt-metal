// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The fused row matmul alone on one core (TRISC only) (fused/fidelity/lab.py): [ROWS, 32] row tiles times [32, 32]
// weight tiles, both resident in this core's L1 as the tensors that back the input CBs, FP32 accumulation, the
// FP32 sums packed as they are into the tensor that backs the output CB. No data movement kernel and no wait on
// another core: the program cannot stall on anything but itself.
//
// The K range is cut into blocks of kb K tiles (one hifi_mm_block call each, as a streamed unit does); with a
// SplitAcc layout only the last call merges the partials. `reps` repeats the whole matmul (timing; the output is
// that of one matmul, DEST is cleared by every pack).
//
// compile-time args: [cb_x, cb_w, cb_out, kt, ct, kb, passes, phase_step, reps]
// defines: LAB_LAYOUT (0 Default, 1 DefaultSplitAcc, 2 Dense, 3 DenseSplitAcc, 4 Compact)
//          LAB_COLUMN_MAJOR  the weight tiles are column-major (read_transposed); needs kb == kt
//          LAB_REF           the tree's custom_mm instead of hifi_mm (passes must be 1)
//          LAB_ROWS16        [16, 32] row tiles through hifi_mm16 (layout Dense or Default)
//          LAB_MERGE         SplitAcc layouts: hifi_mm_merge (exact, SFPU) instead of the last block's finalize
//          LAB_ROUND         with LAB_MERGE: the merged sums rounded to BF16 (hifi_mm_merge<.., true>)

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/experimental/custom_mm.h"
#include "api/compute/pack.h"
#include "hifi_mm.h"

#ifndef LAB_LAYOUT
#define LAB_LAYOUT 3
#endif

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t ct = get_compile_time_arg_val(4);
    constexpr uint32_t kb = get_compile_time_arg_val(5);
    constexpr uint32_t passes = get_compile_time_arg_val(6);
    constexpr uint32_t phase_step = get_compile_time_arg_val(7);
    constexpr uint32_t reps = get_compile_time_arg_val(8);

    constexpr bool transpose = false;
#if LAB_LAYOUT == 0
    constexpr auto layout = CustomMmDestLayout::Default;
#elif LAB_LAYOUT == 1
    constexpr auto layout = CustomMmDestLayout::DefaultSplitAcc;
#elif LAB_LAYOUT == 2
    constexpr auto layout = CustomMmDestLayout::Dense;
#elif LAB_LAYOUT == 3
    constexpr auto layout = CustomMmDestLayout::DenseSplitAcc;
#else
    constexpr auto layout = CustomMmDestLayout::Compact;
#endif
    constexpr bool split = custom_mm_split_acc(layout);
#ifdef LAB_COLUMN_MAJOR
    constexpr bool read_transposed = true;
    static_assert(kb == kt, "column-major weights are read as one block");
#else
    constexpr bool read_transposed = false;
#endif
    static_assert(kt % kb == 0, "blocks of kb K tiles");
    constexpr uint32_t n_blocks = kt / kb;

#ifdef LAB_ROWS16
    hifi_mm16_block_init<passes, phase_step, layout>(cb_x, cb_w, cb_out, ct);
#elif defined(LAB_REF)
    static_assert(passes == 1, "custom_mm is LoFi");
    custom_mm_block_init<transpose, layout>(cb_x, cb_w, cb_out, ct);
#else
    hifi_mm_block_init<passes, phase_step, transpose, layout>(cb_x, cb_w, cb_out, ct);
#endif
#ifdef LAB_MERGE
    static_assert(split, "LAB_MERGE needs a SplitAcc layout");
    hifi_mm_merge_init();
#endif

    for (uint32_t rep = 0; rep < reps; ++rep) {
        tile_regs_acquire();
        for (uint32_t b = 0; b < n_blocks; ++b) {
            const uint32_t w0 = read_transposed ? 0 : b * kb * ct;
#ifdef LAB_ROWS16
            hifi_mm16_block<layout, read_transposed>(cb_x, cb_w, b * kb, w0, 0, kb, ct);
#elif defined(LAB_REF)
            if (split && b + 1 == n_blocks) {
                custom_mm_block<layout, split, read_transposed>(cb_x, cb_w, b * kb, w0, 0, kb, ct);
            } else {
                custom_mm_block<layout, false, read_transposed>(cb_x, cb_w, b * kb, w0, 0, kb, ct);
            }
#else
#ifdef LAB_MERGE
            hifi_mm_block<passes, layout, false, read_transposed>(cb_x, cb_w, b * kb, w0, 0, kb, ct);
#else
            if (split && b + 1 == n_blocks) {
                hifi_mm_block<passes, layout, split, read_transposed>(cb_x, cb_w, b * kb, w0, 0, kb, ct);
            } else {
                hifi_mm_block<passes, layout, false, read_transposed>(cb_x, cb_w, b * kb, w0, 0, kb, ct);
            }
#endif
#endif
        }
#ifdef LAB_MERGE
#ifdef LAB_ROUND
        hifi_mm_merge<ct, layout, true>();
#else
        hifi_mm_merge<ct, layout>();
#endif
#endif
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t c = 0; c < ct; ++c) {
            pack_tile<true>(c, cb_out, c);
        }
        tile_regs_release();
    }
#ifdef LAB_REF
    custom_mm_block_uninit<layout>(cb_out);
#else
    hifi_mm_block_uninit<layout>(cb_out);
#endif
}
