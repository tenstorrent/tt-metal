// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Compute kernel for minimal_binary_op.
// Supports both FPU (bfloat16) and SFPU (float32) paths via IS_FP32 define.
//
// Compile-time args:
//   [0] block_size      — tiles processed per CB wait/push cycle
//   [1] sub_block_size  — tiles per tile_regs_acquire/release cycle (SFPU path only;
//                         must divide block_size; FPU path always processes 1 tile per cycle)
//
// Runtime args:
//   [0] Wt  — total tiles to process for this core
//
// Defines (set by host):
//   IS_FP32          — 1 → SFPU path (float32), 0 → FPU path (bfloat16)
//   PROCESS_OP_INIT  — one-time init expression, e.g.:
//                      FPU:  binary_tiles_init<true, ELWADD>(c_0, c_1)
//                      SFPU: add_binary_tile_init()
//   PROCESS_OP       — per-tile op macro:
//                      FPU  (5 args): add_tiles  / mul_tiles
//                      SFPU (3 args): add_binary_tile / mul_binary_tile

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/fill.h"

void kernel_main() {
    constexpr uint32_t block_size = get_compile_time_arg_val(0);
    constexpr uint32_t sub_block_size = get_compile_time_arg_val(1);

    const uint32_t Wt = get_arg_val<uint32_t>(0);

    constexpr auto c_a = tt::CBIndex::c_0;
    constexpr auto c_b = tt::CBIndex::c_1;
    constexpr auto c_out = tt::CBIndex::c_2;

    // One-time initialization
#if IS_FP32
    unary_op_init_common(c_a, c_out);
#else
    binary_op_init_common(c_a, c_b, c_out);
#endif
    PROCESS_OP_INIT;

    UNPACK(DPRINT << "block size = " << block_size << ENDL(););
    uint32_t tiles_done = 0;
    while (tiles_done < Wt) {
        UNPACK(DPRINT << "tiles dones - " << tiles_done << ENDL(););
#if IS_FP32

        UNPACK(DPRINT << "FP32 path" << ENDL(););
        // SFPU path: batch block_size tiles; process sub_block_size per DST acquire/release.
        // copy_tile with explicit tile indices works correctly here.
        cb_wait_front(c_a, block_size);
        cb_wait_front(c_b, block_size);
        cb_reserve_back(c_out, block_size);
        for (uint32_t sb = 0; sb < block_size; sb += sub_block_size) {
            UNPACK(DPRINT << "sb = " << sb << ENDL(););
            tile_regs_acquire();
            copy_tile_to_dst_init_short(c_a);
            for (uint32_t k = 0; k < sub_block_size; ++k) {
                UNPACK(DPRINT << "copy a from " << sb + k << " to " << k * 2 << ENDL(););
                copy_tile(c_a, sb + k, k * 2);  // A tile → DST[k*2]
            }
            copy_tile_to_dst_init_short(c_b);
            for (uint32_t k = 0; k < sub_block_size; ++k) {
                UNPACK(DPRINT << "copy b from " << sb + k << " to " << k * 2 + 1 << ENDL(););
                copy_tile(c_b, sb + k, k * 2 + 1);  // B tile → DST[k*2+1]
                fill_tile_init();
                // fill_tile(k * 2, 3.f);
                fill_tile(k * 2 + 1, 4.f);
                PROCESS_OP(k * 2, k * 2 + 1, k * 2);  // op(DST[k*2], DST[k*2+1]) → DST[k*2]
                                                      // fill_tile_init();
                // fill_tile(k * 2, 12.f);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t k = 0; k < sub_block_size; ++k) {
                pack_tile(k * 2, c_out);
            }
            tile_regs_release();
        }
        cb_push_back(c_out, block_size);
        cb_pop_front(c_a, block_size);
        cb_pop_front(c_b, block_size);
#else
        // FPU path: add_tiles() with tile index > 0 is not reliable on this hardware.
        // The packer's sequential counter resets on tile_regs_release(), so packing
        // across multiple acquire/release cycles within one reserved region is incorrect.
        // Use binary_ng style: one complete cycle (wait/reserve/acquire/op/pack/release/push/pop)
        // per tile.  sub_block_size is intentionally unused in this path.
        cb_wait_front(c_a, block_size);
        cb_wait_front(c_b, block_size);
        cb_reserve_back(c_out, block_size);
        for (uint32_t k = 0; k < block_size; ++k) {
            tile_regs_acquire();
            PROCESS_OP(c_a, c_b, k, k, 0);  // tile 0 → DST[0]
            tile_regs_commit();

            tile_regs_wait();
            pack_tile(0, c_out);
            tile_regs_release();
        }
        cb_pop_front(c_a, block_size);
        cb_pop_front(c_b, block_size);
        cb_push_back(c_out, block_size);
#endif
        tiles_done += block_size;
    }
}
