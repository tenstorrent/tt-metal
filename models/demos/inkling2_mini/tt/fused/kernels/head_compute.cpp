// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Logits tail, compute: per dense page y = (tanh(x * (1/cap))) * cap, fp32 SFPU, the same LLK calls as
// ttnn.multiply (binary_ng SFPU scalar path) and ttnn.tanh, so the values are bit-equal to that op chain.
#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace {
FORCE_INLINE void load(uint32_t cb, uint32_t tile, uint32_t dst) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, tile, dst);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t cb_in = 0, cb_s = 1, cb_out = 16;

    compute_kernel_hw_startup(cb_in, cb_out);
    cb_wait_front(cb_s, 2);
    cb_wait_front(cb_in, T);
    cb_reserve_back(cb_out, T);
    for (uint32_t t = 0; t < T; ++t) {
        tile_regs_acquire();
        load(cb_in, t, 0);
        load(cb_s, 0, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tanh_tile_init();
        tanh_tile(0);
        load(cb_s, 1, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_data_format(cb_out);
        pack_tile(0, cb_out);
        tile_regs_release();
    }
    cb_push_back(cb_out, T);
    cb_pop_front(cb_in, T);
    cb_pop_front(cb_s, 2);
}
