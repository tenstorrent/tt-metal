// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/experimental/eltwise_mul_scalar.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr std::uint32_t iterations = get_compile_time_arg_val(0);
    constexpr auto reuse = get_compile_time_arg_val(1) != 0 ? EltwiseBinaryReuseDestType::DEST_TO_SRCB
                                                            : EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    constexpr std::uint32_t dest_tiles = 3;
    constexpr std::uint32_t target = 1;
    CircularBuffer input(tt::CBIndex::c_0);
    CircularBuffer output(tt::CBIndex::c_16);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        input.wait_front(1);
        output.reserve_back(dest_tiles);
        tile_regs_acquire();
        fill_tile_init();
        for (std::uint32_t tile = 0; tile < dest_tiles; ++tile) {
            fill_tile(tile, static_cast<float>(1u << (iteration + tile)));
        }
        // Init must follow the input CB's tiny face height, even after a full-
        // tile SFPU operation and when reusing the other half of DEST.
        deepseek_binary_dest_reuse_tiles_init<reuse>(tt::CBIndex::c_0);
        deepseek_binary_dest_reuse_tiles<DST_ACCUM_MODE, reuse>(tt::CBIndex::c_0, 0, target);
        tile_regs_commit();
        tile_regs_wait();
        // Full-tile output also exposes writes outside the target DEST slot.
        for (std::uint32_t tile = 0; tile < dest_tiles; ++tile) {
            pack_tile(tile, tt::CBIndex::c_16);
        }
        tile_regs_release();
        input.pop_front(1);
        output.push_back(dest_tiles);
    }
}
