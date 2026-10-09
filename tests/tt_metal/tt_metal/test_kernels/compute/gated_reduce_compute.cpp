// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/experimental/gated_reduce.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "cb_operand_helpers.h"

void kernel_main() {
    constexpr std::uint32_t count = get_compile_time_arg_val(0);
    constexpr std::uint32_t block_tiles = 4;
    constexpr auto shape = experimental::to_llk_mem_descriptor(experimental::Cb<tt::CBIndex::c_0>{}).shape;
    constexpr auto rows = shape.total_row_dim();
    constexpr int iterations = rows <= 4 ? 2 : rows <= 8 ? 4 : 8;
    constexpr auto mode = rows <= 16 ? VectorMode::R : VectorMode::RC;
    using Gate = sfpu::GatedReduceGate;
    using Up = sfpu::GatedReduceUp;

    CircularBuffer input(tt::CBIndex::c_0);
    CircularBuffer output(tt::CBIndex::c_16);
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    for (std::uint32_t base = 0; base < count; base += block_tiles) {
        input.wait_front(block_tiles);
        output.reserve_back(block_tiles);
        copy_init(tt::CBIndex::c_0);
        tile_regs_acquire();
        for (std::uint32_t tile = 0; tile < block_tiles; ++tile) {
            copy_tile(tt::CBIndex::c_0, tile, tile);
        }
        // A different SFPU initializer precedes every batch; gated_reduce must
        // establish its own sigmoid constants and shared unary state.
        exp_tile_init();
        gated_reduce_tile_init();
        constexpr std::uint32_t scale = 0x3f340000;      // 0.703125
        constexpr std::uint32_t out_scale = 0xbfa80000;  // -1.3125
        constexpr std::uint32_t limit = 0x3fa00000;      // 1.25
        constexpr std::uint32_t alpha = 0x3fd9db23;      // 1.702
        switch (base / block_tiles) {
            case 0:
                // Two experts per acquire, output scale and up scale disabled.
                for (std::uint32_t gate = 0; gate < block_tiles; gate += 2) {
                    gated_reduce_tile<Gate::Silu, Up::Identity, true, false, false, iterations>(
                        gate, scale, out_scale, limit, alpha, mode);
                }
                break;
            case 1:
                // Nonzero gate slot with untouched guard tiles on both sides.
                gated_reduce_tile<Gate::ClampedSilu, Up::Clamp, true, true, true, iterations>(
                    1, scale, out_scale, limit, alpha, mode);
                break;
            case 2:
                // Last valid pair, partial expert batch, swapped scalar values.
                gated_reduce_tile<Gate::Silu, Up::Clamp, false, true, true, iterations>(
                    2, out_scale, scale, limit, alpha, mode);
                break;
            case 3:
                gated_reduce_tile<Gate::ClampedSilu, Up::Identity, false, false, true, iterations>(
                    0, scale, out_scale, limit, alpha, mode);
                break;
        }
        tile_regs_commit();
        tile_regs_wait();
        for (std::uint32_t tile = 0; tile < block_tiles; ++tile) {
            pack_tile(tile, tt::CBIndex::c_16);
        }
        tile_regs_release();
        input.pop_front(block_tiles);
        output.push_back(block_tiles);
    }
}
