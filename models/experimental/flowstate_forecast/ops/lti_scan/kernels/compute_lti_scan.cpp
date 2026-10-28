// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Time-outer diagonal complex LTI scan, fp32 throughout. Per step and unit (one 32 x 32 tile of (B, P)):
// Bu (2 tiles) in, the previous state (2 tiles, FIFO through STATE_CB) and the unit's Lbar tiles (resident),
// one SFPU pass, x out (2 tiles) and back into STATE_CB for the next step.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#include "lti_sfpu.h"

void kernel_main() {
    const uint32_t num_units = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(1);

    constexpr uint32_t in_cb = tt::CBIndex::c_0;
    constexpr uint32_t lbar_cb = tt::CBIndex::c_1;
    constexpr uint32_t state_cb = tt::CBIndex::c_2;
    constexpr uint32_t out_cb = tt::CBIndex::c_16;
    CircularBuffer cb_in(in_cb), cb_lbar(lbar_cb), cb_state(state_cb), cb_out(out_cb);

    compute_kernel_hw_startup(in_cb, out_cb);
    cb_lbar.wait_front(2 * num_units);

    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t u = 0; u < num_units; ++u) {
            cb_in.wait_front(2);
            tile_regs_acquire();
            copy_init(in_cb);
            copy_tile(in_cb, 0, 2);
            copy_tile(in_cb, 1, 3);
            if (t > 0) {
                cb_state.wait_front(2);
                copy_init(state_cb);
                copy_tile(state_cb, 0, 0);
                copy_tile(state_cb, 1, 1);
                copy_init(lbar_cb);
                copy_tile(lbar_cb, 2 * u, 4);
                copy_tile(lbar_cb, 2 * u + 1, 5);
                MATH((llk_math_lti_step_init()));
                MATH((llk_math_lti_step<false>()));
            } else {
                MATH((llk_math_lti_step_init()));
                MATH((llk_math_lti_step<true>()));
            }
            tile_regs_commit();
            cb_in.pop_front(2);
            if (t > 0) {
                cb_state.pop_front(2);
            }

            tile_regs_wait();
            cb_out.reserve_back(2);
            pack_tile(0, out_cb);
            pack_tile(1, out_cb);
            cb_out.push_back(2);
            if (t + 1 < T) {
                cb_state.reserve_back(2);
                pack_tile(0, state_cb);
                pack_tile(1, state_cb);
                cb_state.push_back(2);
            }
            tile_regs_release();
        }
    }
}
