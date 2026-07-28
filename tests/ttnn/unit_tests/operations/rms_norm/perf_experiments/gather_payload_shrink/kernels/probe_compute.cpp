// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// gather_payload_shrink MECHANISM PROBE — compute.
//
// QUESTION. The cross-core gather ships one FULL 4 KB fp32 tile per tile-row
// whose only content is 32 row-sums (128 B). To column-PACK `ht` tile-rows'
// row-sums into ONE tile (an `ht`x payload cut) we need a REDUCE_ROW whose
// result lands in a chosen COLUMN instead of always column 0.
//
// The BH LLK comment says REDUCE_ROW SUM is `MVMUL` with the scaler in SrcA and
// the data in SrcB, i.e. dest = data x scalerᵀ. If that is literally true then
// the scaler's ROW index picks the output COLUMN, and the whole scheme is one
// `reduce_tile` per tile-row accumulating into ONE dest tile.
//
// Two things can break it and both are measured here:
//   (a) the actual (scaler position) -> (dest position) map, and
//   (b) `reduce_init`'s PACKER EDGE MASK, which is documented to write every
//       datum outside column 0 as ZERO — that would erase a column-packed dest
//       on the way to L1 unless `reduce_uninit()` (mask clear) is issued first.
//
// RAW LLK / raw compute-API use is deliberate: this probe's whole subject is a
// non-canonical scaler layout that no kernel_lib reduce helper can express, and
// the packer-mask interaction is invisible at the helper level.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reduce.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"

namespace {
constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_sc = 2;
constexpr uint32_t cb_out = 16;
}  // namespace

void kernel_main() {
    constexpr uint32_t NUM_IN_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t NUM_SC_TILES = get_compile_time_arg_val(1);
    constexpr uint32_t NUM_OUT_TILES = get_compile_time_arg_val(2);

    // REDUCE_ROW SUM maps scaler -> SrcA and data -> SrcB (reduce.h:70-76), so
    // startup must program the source registers in that (reversed) order.
    compute_kernel_hw_startup<ckernel::SrcOrder::Reverse>(cb_in, cb_sc, cb_out);

    cb_wait_front(cb_in, NUM_IN_TILES);
    cb_wait_front(cb_sc, NUM_SC_TILES);
    cb_reserve_back(cb_out, NUM_OUT_TILES);

    // ---- experiments 0..7: one scaler layout each, mask CLEARED before the
    // pack so the raw dest content is visible.
    for (uint32_t s = 0; s < 8; ++s) {
        reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(cb_in, cb_sc, cb_out);
        tile_regs_acquire();
        reduce_tile<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(cb_in, cb_sc, /*itile=*/0, s, /*idst=*/0);
        tile_regs_commit();
        reduce_uninit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }

    // ---- experiment 8: accumulate scalers 0,1,2 into ONE dest, pack WITH the
    // reduce packer mask still in effect (expect: only column 0 survives).
    reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(cb_in, cb_sc, cb_out);
    tile_regs_acquire();
    for (uint32_t s = 0; s < 3; ++s) {
        reduce_tile<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(cb_in, cb_sc, 0, s, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    // ---- experiment 9: same accumulate, mask CLEARED before the pack.
    reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(cb_in, cb_sc, cb_out);
    tile_regs_acquire();
    for (uint32_t s = 0; s < 3; ++s) {
        reduce_tile<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(cb_in, cb_sc, 0, s, 0);
    }
    tile_regs_commit();
    reduce_uninit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    // ---- experiment 10: canonical scaler over FOUR input tiles accumulating
    // into one dest (the op's own within-tile fold pattern) -> 4 x 32 = 128.
    reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(cb_in, cb_sc, cb_out);
    tile_regs_acquire();
    for (uint32_t t = 0; t < 4; ++t) {
        reduce_tile<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(cb_in, cb_sc, t, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    // ---- experiment 11: mul_tiles_bcast<COL>(cb_sc[7], cb_in[0]) — does a
    // column-broadcast multiply read column 0 of the SECOND operand? This is
    // the fallback column-pack mechanism (broadcast a folded row-sum across the
    // tile, mask it down to one column, accumulate).
    reduce_uninit();
    mul_bcast_cols_init_short(cb_sc, cb_in);
    tile_regs_acquire();
    mul_tiles_bcast_cols(cb_sc, cb_in, 7, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out);
    tile_regs_release();

    for (uint32_t t = 12; t < NUM_OUT_TILES; ++t) {
        reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(cb_in, cb_sc, cb_out);
        tile_regs_acquire();
        reduce_tile<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(cb_in, cb_sc, 0, 3, 0);
        tile_regs_commit();
        reduce_uninit();
        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();
    }

    cb_push_back(cb_out, NUM_OUT_TILES);
}
