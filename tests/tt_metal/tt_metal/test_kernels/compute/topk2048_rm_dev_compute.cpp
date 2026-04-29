// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Placeholder TopK-2048 compute: unpack Float32 input tiles to dest, SFPU sort TBD, pack
// the first k_output_tiles to CB16 (passthrough for bring-up). Extra input tiles are
// consumed (unpacked) so the input CB drains; extend here with full top-K.
//

#include <cstdint>

#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/compute_kernel_api.h"
#include "experimental/dataflow_buffer.h"
#include "experimental/circular_buffer.h"
#include "api/debug/device_print.h"

void kernel_main() {
    const uint32_t num_in_tiles = get_compile_time_arg_val(0);
    const uint32_t k_output_tiles = get_compile_time_arg_val(1);

    unary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_16);

    experimental::CircularBuffer cb0(tt::CBIndex::c_0);
    experimental::CircularBuffer cb16(tt::CBIndex::c_16);

    acquire_dst();
    topk_xl_init();
    const bool ascending = true;
    const bool descending = false;

    uint64_t topk_local_sort_wall_clock_cycles = 0;
    uint64_t topk_merge_wall_clock_cycles = 0;
    uint64_t topk_rebuild_wall_clock_cycles = 0;

    // first input tile pair
    cb0.wait_front(2);
    copy_tile(tt::CBIndex::c_0, 0, 0);
    copy_tile(tt::CBIndex::c_0, 1, 1);
    topk_xl_local_sort(0, descending);
    cb0.pop_front(2);

    // subsequent input tile pairs
    for (uint32_t t = 2; t < num_in_tiles; t += 2) {
        cb0.wait_front(2);

        copy_tile(tt::CBIndex::c_0, 0, 2);
        copy_tile(tt::CBIndex::c_0, 1, 3);

        const uint64_t wall_t0 = ckernel::read_wall_clock();
        topk_xl_local_sort(2, ascending);
        const uint64_t wall_t1 = ckernel::read_wall_clock();
        topk_xl_merge(0);
        const uint64_t wall_t2 = ckernel::read_wall_clock();
        topk_xl_rebuild(0, descending);
        const uint64_t wall_t3 = ckernel::read_wall_clock();

        topk_local_sort_wall_clock_cycles += (wall_t1 - wall_t0);
        topk_merge_wall_clock_cycles += (wall_t2 - wall_t1);
        topk_rebuild_wall_clock_cycles += (wall_t3 - wall_t2);

        cb0.pop_front(2);
    }

    // #ifdef TRISC_MATH
    //     // use risc to read DEST directly
    //     cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_fmt_RMW, 0b000 /* RISC_DEST_FMT_FP32 */);
    //     cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW, 0);
    //     cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW, 0);
    //     tensix_sync();

    //     volatile float* dst32 = reinterpret_cast<volatile float*>(0xFFBD8000U);
    //     DEVICE_PRINT("DEST[row][col] FP32:\n");
    //     for (int row = 0; row < 128; row++) {
    //         DEVICE_PRINT("row {:03d}: ", row);
    //         for (int col = 0; col < 16; col++) {
    //             float val = dst32[row * 16 + col];
    //             DEVICE_PRINT("{:11.6f} ", val);
    //         }
    //         DEVICE_PRINT("\n");
    //     }
    // #endif

    cb16.reserve_back(k_output_tiles);
    for (uint32_t t = 0; t < k_output_tiles; t++) {
        pack_tile(t, tt::CBIndex::c_16);
    }
    cb16.push_back(k_output_tiles);

    DEVICE_PRINT(
        "topk_xl_local_sort wall_clock cycles: {} topk_xl_merge wall_clock cycles: {} topk_xl_rebuild wall_clock "
        "cycles: {}\n",
        topk_local_sort_wall_clock_cycles,
        topk_merge_wall_clock_cycles,
        topk_rebuild_wall_clock_cycles);

    release_dst();
}
