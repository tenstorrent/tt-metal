// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Minimal repro for a Wormhole-B0-only LLK bug seen with:
//   - fp32_dest_acc_en = true
//   - DstSync::SyncHalf (block_size = 4 with fp32 dest acc)
//   - An UnpackToDest fp32 copy_tile (from any CB in UnpackToDestFp32 mode)
//     followed by sub_tiles_bcast_cols + binary_dest_reuse_tiles<ELWMUL,
//     DEST_TO_SRCB> on a CB whose primary buffer index is in Default (Tf32)
//     unpack mode.
//
// The kernel intentionally does the same compute sequence in both Case A
// (cb_primary and cb_alias have SEPARATE L1 allocations) and Case B
// (cb_primary and cb_alias share ONE L1 allocation via two CBFormatDescriptors).
// Both cases fail identically on WH and pass identically on BH, which
// proves multi-buffer-index CB aliasing is NOT a required trigger condition.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"

using namespace ckernel;

void kernel_main() {
    // Compile-time args:
    //   0: n_blocks (each block is block_size tiles, DstSync::SyncHalf alternates parity)
    //   1: block_size (number of tiles per dst-half block; 4 for fp32 dest acc on WH)
    constexpr uint32_t n_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t block_size = get_compile_time_arg_val(1);

    // CB layout: cb_primary (Tf32, Default unpack mode) holds x. cb_alias is the
    // second buffer-index view of the SAME (Case B) or a SEPARATE (Case A) L1
    // allocation, in UnpackToDestFp32 mode. The host writes the same x bytes
    // into both views. cb_bcast_a and cb_bcast_b are column-broadcast tiles
    // (only the first column is meaningful) used by the sub and mul ops.
    constexpr uint32_t cb_primary = tt::CBIndex::c_0;  // x, Default Tf32 unpack mode
    constexpr uint32_t cb_alias = tt::CBIndex::c_29;   // alias of cb_primary, UnpackToDestFp32
    constexpr uint32_t cb_bcast_a = tt::CBIndex::c_2;  // column-broadcast 'a' (subtrahend)
    constexpr uint32_t cb_bcast_b = tt::CBIndex::c_3;  // column-broadcast 'b' (multiplier)
    constexpr uint32_t cb_out = tt::CBIndex::c_16;     // output

    compute_kernel_hw_startup(cb_primary, cb_bcast_a, cb_out);

    // Wait once for the bcast operand tiles; reader pushes them up front and they
    // are read repeatedly. (One tile each suffices because we use them as broadcasts.)
    cb_wait_front(cb_bcast_a, 1);
    cb_wait_front(cb_bcast_b, 1);

    for (uint32_t block = 0; block < n_blocks; ++block) {
        // Wait for one block's worth of x tiles in cb_primary (and equivalently
        // cb_alias, which aliases the same allocation or has a separate semaphore).
        cb_wait_front(cb_primary, block_size);
        cb_wait_front(cb_alias, block_size);

        tile_regs_acquire();

        // Step 1: copy_tile from the ALIAS index. With unpack_to_dest_mode=
        // UnpackToDestFp32 on the alias index and Float32 data, the LLK takes
        // the UnpackToDest fp32 path. We discard the result by overwriting the
        // same dst slot below; the side effects on unpacker state are what
        // matters.
        reconfig_data_format_srca(cb_primary, cb_alias);
        copy_tile_init(cb_alias);
        copy_tile(cb_alias, 0, 0);

        // Step 2: sub_tiles_bcast_cols on the PRIMARY index. Produces (x - a) in
        // dst[i] for each i in [0, block_size).
        reconfig_data_format_srca(cb_alias, cb_primary);
        sub_bcast_cols_init_short(cb_primary, cb_bcast_a);
        for (uint32_t i = 0; i < block_size; ++i) {
            sub_tiles_bcast_cols(cb_primary, cb_bcast_a, i, 0, i);
        }

        // Step 3: dest-reuse ELWMUL with cb_bcast_b. Reads dst[i] back via SrcB,
        // multiplies by cb_bcast_b[0], writes (x - a) * b into dst[i].
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_bcast_b);
        for (uint32_t i = 0; i < block_size; ++i) {
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                cb_bcast_b, 0, i);
        }

        tile_regs_commit();

        // Pack out the block.
        cb_reserve_back(cb_out, block_size);
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        for (uint32_t i = 0; i < block_size; ++i) {
            pack_tile(i, cb_out);
        }
        tile_regs_release();
        cb_push_back(cb_out, block_size);

        cb_pop_front(cb_primary, block_size);
        cb_pop_front(cb_alias, block_size);
    }

    cb_pop_front(cb_bcast_a, 1);
    cb_pop_front(cb_bcast_b, 1);
}
