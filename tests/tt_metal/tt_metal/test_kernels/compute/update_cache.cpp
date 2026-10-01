// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"

// Largest pack_untilize block width (<= DEST tile capacity) dividing full_ct_dim.
constexpr uint32_t untilize_pack_block_ct(uint32_t full_ct_dim) {
    const uint32_t max_bct = DST_ACCUM_MODE ? 4 : 8;
    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (full_ct_dim % bct == 0) {
            return bct;
        }
    }
    return 1;
}

// Untilize `full_ct_dim` tiles from icb to ocb using pack_untilize (replaces the removed
// unpack-based untilize op). Handles the full cb hand-off (wait/reserve/pop/push).
template <uint32_t full_ct_dim>
ALWI void untilize_to_cb(uint32_t icb, uint32_t ocb) {
    constexpr uint32_t block_ct = untilize_pack_block_ct(full_ct_dim);
    constexpr uint32_t num_blocks = full_ct_dim / block_ct;
    pack_untilize_init<block_ct, full_ct_dim>(icb, ocb);
    cb_wait_front(icb, full_ct_dim);
    cb_reserve_back(ocb, full_ct_dim);
    for (uint32_t b = 0; b < num_blocks; ++b) {
        pack_untilize_block<block_ct, full_ct_dim>(icb, 1, ocb, b);
        cb_pop_front(icb, block_ct);
    }
    cb_push_back(ocb, full_ct_dim);
    pack_untilize_uninit(ocb);
}

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t cache_cb = get_compile_time_arg_val(0);
    constexpr uint32_t in_cb = get_compile_time_arg_val(1);
    constexpr uint32_t untilized_cache_cb = get_compile_time_arg_val(2);
    constexpr uint32_t untilized_cache2_cb = get_compile_time_arg_val(3);
    constexpr uint32_t untilized_in_cb = get_compile_time_arg_val(4);
    constexpr uint32_t out_cb = get_compile_time_arg_val(5);
    constexpr uint32_t B = get_compile_time_arg_val(6);
    constexpr uint32_t Wt = get_compile_time_arg_val(7);

    compute_kernel_hw_startup(in_cb, untilized_in_cb);

    for (uint32_t b = 0; b < B / 32; b++) {
        untilize_to_cb<Wt>(in_cb, untilized_in_cb);

        for (uint32_t u = 0; u < 32; u++) {
            untilize_to_cb<Wt>(cache_cb, untilized_cache_cb);

            tilize_init(untilized_cache2_cb, Wt, out_cb);
            cb_wait_front(untilized_cache2_cb, Wt);
            cb_reserve_back(out_cb, Wt);
            tilize_block(untilized_cache2_cb, Wt, out_cb);
            cb_push_back(out_cb, Wt);
            // Untilized cache CBs share same address space
            // Compute pops both
            cb_pop_front(untilized_cache2_cb, Wt);
            cb_pop_front(untilized_cache_cb, Wt);
            tilize_uninit(untilized_cache2_cb, out_cb);
        }
    }
}
