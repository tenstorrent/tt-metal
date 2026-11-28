// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Local-only combine (Laguna), compute: untilize each unit's tile row (Ht tiles, BLK per pack call) into 32 bf16
// rows of cb_out.

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack_untilize.h"

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t BLK = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in = 0, cb_n = 1, cb_out = 16;
    compute_kernel_hw_startup(cb_in, cb_out);
    pack_untilize_init<BLK, Ht>(cb_in, cb_out);
    cb_wait_front(cb_n, 1);
    const uint32_t n = read_tile_value(cb_n, 0, 0);
    cb_pop_front(cb_n, 1);
    for (uint32_t u = 0; u < n; ++u) {
        cb_reserve_back(cb_out, 32);
        for (uint32_t b = 0; b < Ht / BLK; ++b) {
            cb_wait_front(cb_in, BLK);
            pack_untilize_block<BLK, Ht>(cb_in, 1, cb_out, b);
            cb_pop_front(cb_in, BLK);
        }
        cb_push_back(cb_out, 32);
    }
    pack_untilize_uninit(cb_out);
}
