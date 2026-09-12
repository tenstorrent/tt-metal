// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused KV-cache decode-time select, single kernel pass, no DRAM intermediates:
//   cache_new = cache + onehot * (new_val - cache)
//
// Per tile:
//   1. broadcast onehot's valid column (col 0) across all 32 columns  -> onehot_b
//   2. broadcast new_val's valid row (row 0) across all 32 rows       -> new_b
//   3. diff   = new_b - cache
//   4. scaled = onehot_b * diff
//   5. out    = cache + scaled
// All of steps 1-5 stay in L1/dst registers; only `cache`/`onehot`/`new_val` are
// read from DRAM and only `out` is written back to DRAM (by the writer kernel).

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"

namespace ckernel {
// `add_tiles`/`sub_tiles` (api/compute/eltwise_binary.h) hardcode MathFidelity::LoFi for the
// MATH() call (only `mul_tiles` honors the configured MATH_FIDELITY). For this op the inputs are
// float32 and the whole point is an EXACT select (cache_new = cache + onehot*(new-cache) must
// reproduce cache exactly wherever onehot==0) — LoFi's reduced-mantissa datapath was measured to
// introduce ~0.4% relative error on every element (touched AND untouched), which is unacceptable
// precision loss for a "copy-through" cache. These local variants are byte-identical to
// add_tiles/sub_tiles except they pass MATH_FIDELITY (the configured math_fidelity, HiFi4 here)
// instead of the hardcoded LoFi. Their *_init counterparts (add_tiles_init/sub_tiles_init) already
// use MATH_FIDELITY, so no separate init is needed.
ALWI void add_tiles_hifi(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWADD,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
}

ALWI void sub_tiles_hifi(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWSUB,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          MATH_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, true /* clear_fp32_dst_acc */)));
}
}  // namespace ckernel

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_cache = tt::CBIndex::c_0;
    constexpr auto cb_onehot = tt::CBIndex::c_1;
    constexpr auto cb_new = tt::CBIndex::c_2;
    constexpr auto cb_onehot_b = tt::CBIndex::c_3;
    constexpr auto cb_new_b = tt::CBIndex::c_4;
    constexpr auto cb_diff = tt::CBIndex::c_5;
    constexpr auto cb_scaled = tt::CBIndex::c_6;
    constexpr auto cb_out = tt::CBIndex::c_7;

    constexpr uint32_t dst0 = 0;

    using namespace ckernel;

    binary_op_init_common(cb_cache, cb_new, cb_out);

    for (uint32_t i = 0; i < n_tiles; ++i) {
        // --- (1) onehot_b[h,w] = onehot[h,0]  (broadcast the valid column across columns) ---
        cb_wait_front(cb_onehot, 1);
        cb_reserve_back(cb_onehot_b, 1);
        unary_bcast_init<BroadcastType::COL>(cb_onehot, cb_onehot_b);
        tile_regs_acquire();
        unary_bcast<BroadcastType::COL>(cb_onehot, 0, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_onehot_b);
        tile_regs_release();
        cb_push_back(cb_onehot_b, 1);
        cb_pop_front(cb_onehot, 1);

        // --- (2) new_b[h,w] = new_val[0,w]  (broadcast the valid row across rows) ---
        cb_wait_front(cb_new, 1);
        cb_reserve_back(cb_new_b, 1);
        unary_bcast_init<BroadcastType::ROW>(cb_new, cb_new_b);
        tile_regs_acquire();
        unary_bcast<BroadcastType::ROW>(cb_new, 0, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_new_b);
        tile_regs_release();
        cb_push_back(cb_new_b, 1);
        cb_pop_front(cb_new, 1);

        // --- (3) diff = new_b - cache ---
        cb_wait_front(cb_new_b, 1);
        cb_wait_front(cb_cache, 1);
        cb_reserve_back(cb_diff, 1);
        sub_tiles_init(cb_new_b, cb_cache);
        tile_regs_acquire();
        sub_tiles_hifi(cb_new_b, cb_cache, 0, 0, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_diff);
        tile_regs_release();
        cb_push_back(cb_diff, 1);
        cb_pop_front(cb_new_b, 1);

        // --- (4) scaled = onehot_b * diff ---
        cb_wait_front(cb_onehot_b, 1);
        cb_wait_front(cb_diff, 1);
        cb_reserve_back(cb_scaled, 1);
        mul_tiles_init(cb_onehot_b, cb_diff);
        tile_regs_acquire();
        mul_tiles(cb_onehot_b, cb_diff, 0, 0, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_scaled);
        tile_regs_release();
        cb_push_back(cb_scaled, 1);
        cb_pop_front(cb_onehot_b, 1);
        cb_pop_front(cb_diff, 1);

        // --- (5) out = cache + scaled ---
        cb_wait_front(cb_scaled, 1);
        cb_reserve_back(cb_out, 1);
        add_tiles_init(cb_cache, cb_scaled);
        tile_regs_acquire();
        add_tiles_hifi(cb_cache, cb_scaled, 0, 0, dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_out);
        tile_regs_release();
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_cache, 1);
        cb_pop_front(cb_scaled, 1);
    }
}
