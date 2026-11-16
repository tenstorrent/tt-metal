// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/experimental/rope_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr std::uint32_t count = get_compile_time_arg_val(0);
    constexpr std::uint32_t block = 4;
    CircularBuffer input(tt::CBIndex::c_0);
    CircularBuffer output(tt::CBIndex::c_16);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);

    for (std::uint32_t tile = 0; tile < count; tile += block) {
        input.wait_front(block);
        output.reserve_back(block);
        copy_init(tt::CBIndex::c_0);
        tile_regs_acquire();
        for (std::uint32_t i = 0; i < block; ++i) {
            copy_tile(tt::CBIndex::c_0, i, i);
        }
        rope_sfpu_fused_init();
        if (tile == 0) {
            // Two x tiles followed by the phase tile and an untouched guard.
            // The public API must infer FP32 DEST even though the input CB is BF16.
            rope_sfpu_inplace_fused<2, 1, 8, true>();
        } else {
            // Move the phase before x, rotate both row halves, and fold in a scale.
            rope_sfpu_inplace_fused_rows<2, 1, 64, 64, 0, 64, true, 32, true>(0xc0000000);  // -2.0f
        }
        tile_regs_commit();
        tile_regs_wait();
        for (std::uint32_t i = 0; i < block; ++i) {
            pack_tile(i, tt::CBIndex::c_16);
        }
        tile_regs_release();
        input.pop_front(block);
        output.push_back(block);
    }
}
