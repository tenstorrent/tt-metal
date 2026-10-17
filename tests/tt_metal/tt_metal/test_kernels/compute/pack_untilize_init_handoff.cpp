// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

// Blackhole: a pack-untilize init left armed without its uninit, handed off to pack_untilize_dest_init on an
// output whose dest-register (pack source) datum size differs. Runs with fp32 dest accumulation: the first init
// targets a Float16_b scratch output (2-byte pack source, c_17, never written), the second the real Float32
// output (4-byte pack source). The second init must program its own Y/W strides for the 4-byte source; an init
// that only programs Z and leaves Y/W to a data-format reconfig keeps the 2-byte strides (the reconfig leaves the
// strides alone while a pack-untilize init is armed) and untilizes the wrong dest rows. Exact golden.
void kernel_main() {
    constexpr uint32_t rows = get_compile_time_arg_val(0);
    constexpr uint32_t width = get_compile_time_arg_val(1);

    constexpr uint32_t in_cb = tt::CBIndex::c_0;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    constexpr uint32_t scratch_out_cb = tt::CBIndex::c_17;

    CircularBuffer cb_in(in_cb);
    CircularBuffer cb_out(out_cb);

    compute_kernel_hw_startup(in_cb, scratch_out_cb);
    // First untilize init (unpack/math copy + pack untilize for the scratch output), never uninit-ed.
    pack_untilize_init<width, width>(in_cb, scratch_out_cb);
    // Hand-off to the real output: pack side only.
    pack_untilize_dest_init<width, width>(out_cb);

    for (uint32_t row = 0; row < rows; ++row) {
        cb_in.wait_front(width);
        cb_out.reserve_back(width);
        tile_regs_acquire();
        for (uint32_t tile = 0; tile < width; ++tile) {
            copy_tile(in_cb, tile, tile);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_untilize_dest<width, width>(out_cb, 1 /*block_rt_dim*/, 0 /*block_c_index*/);
        tile_regs_release();
        cb_in.pop_front(width);
        cb_out.push_back(width);
    }
    pack_untilize_uninit(out_cb);
}
