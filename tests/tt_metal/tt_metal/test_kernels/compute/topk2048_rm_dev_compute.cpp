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
    const bool ascending = false;

    // one input takes 2 tiles
    for (uint32_t t = 0; t < num_in_tiles; t += 2) {
        cb0.wait_front(2);

        copy_tile(tt::CBIndex::c_0, 0, 0);
        copy_tile(tt::CBIndex::c_0, 1, 1);

        topk_xl_local_sort(0, ascending);
        // topk_xl_merge(0);
        // topk_xl_rebuild(0, ascending);

        cb0.pop_front(2);
    }

#ifdef TRISC_MATH
    // use risc to read DEST directly
    cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_fmt_RMW, 0b000 /* RISC_DEST_FMT_FP32 */);
    cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_no_swizzle_RMW, 0);
    cfg_rmw(RISC_DEST_ACCESS_CTRL_SEC2_unsigned_int_RMW, 0);
    tensix_sync();

    volatile float* dst32 = reinterpret_cast<volatile float*>(0xFFBD8000U);
    DEVICE_PRINT("DEST[row][col] FP32:\n");
    for (int row = 0; row < 128; row++) {
        DEVICE_PRINT("row {:03d}: ", row);
        for (int col = 0; col < 16; col++) {
            float val = dst32[row * 16 + col];
            DEVICE_PRINT("{:8.3f} ", val);
        }
        DEVICE_PRINT("\n");
    }
#endif

    cb16.reserve_back(k_output_tiles);
    for (uint32_t t = 0; t < k_output_tiles; t++) {
        pack_tile(t, tt::CBIndex::c_16);
    }
    cb16.push_back(k_output_tiles);

    release_dst();
}
