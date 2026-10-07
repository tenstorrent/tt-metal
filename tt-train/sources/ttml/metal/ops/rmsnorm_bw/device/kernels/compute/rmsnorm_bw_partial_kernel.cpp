// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// rmsnorm_bw phase A: per (tile-row, slice) work item
//   partial = sum_{tiles j in slice} a_j * gamma_j * dL_dout_j     (elementwise; the last tile masked to C)
// One fp32 tile per item; phase B sums the slices and row-reduces.

#include "rmsnorm_bw_compute_common.hpp"

void kernel_main() {
    compute_kernel_hw_startup(cb::dy, cb::gamma, cb::partial_out);
    cb_wait_front(cb::zero, 1);
#ifdef DO_MASK_W
    cb_wait_front(cb::mask, 1);
#endif

    for (uint32_t item = 0; item < work_count; ++item) {
        const uint32_t work_start = get_arg_val<uint32_t>(0);
        const SliceGeometry sl = slice_for(work_start + item);

        tile_regs_acquire();
        load_tile(cb::zero, 0, 0);  // accumulator
        for (uint32_t c = 0; c < sl.ncols; c += block) {
            const uint32_t n = (c + block <= sl.ncols) ? block : (sl.ncols - c);
            cb_wait_front(cb::a, n);
            cb_wait_front(cb::gamma, n);
            cb_wait_front(cb::dy, n);
            for (uint32_t i = 0; i < n; ++i) {
                gamma_times_dy(i, 1);
                load_tile(cb::a, i, 2);
                mul_binary_tile_init();
                mul_binary_tile(1, 2, 1);
#ifdef DO_MASK_W
                if (sl.col0 + c + i + 1 == Wt) {
                    load_tile(cb::mask, 0, 2);
                    mul_binary_tile_init();
                    mul_binary_tile(1, 2, 1);
                }
#endif
                add_binary_tile_init();
                add_binary_tile(0, 1, 0);
            }
            cb_pop_front(cb::a, n);
            cb_pop_front(cb::gamma, n);
            cb_pop_front(cb::dy, n);
        }
        tile_regs_commit();
        pack_and_push(0, cb::partial_out);
    }
}
