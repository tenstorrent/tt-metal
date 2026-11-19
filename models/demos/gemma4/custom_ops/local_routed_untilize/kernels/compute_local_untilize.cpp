// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// LocalRoutedUntilize compute: per unit (one tile-row of an expert region) pack-untilizes the H_T tiles the reader
// streams into cb_in, BLOCK_CT at a time, into 32 row pages of cb_out (bf8 -> bf16 in the unpacker / packer). The
// unit count comes from the reader through cb_n (read_tile_value: UNPACK reads, MATH / PACK get it by mailbox).
//
// compile-time args: [cb_in, cb_out, cb_n, H_T, BLOCK_CT]
#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack_untilize.h"

void kernel_main() {
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_n = get_compile_time_arg_val(2);
    constexpr uint32_t H_T = get_compile_time_arg_val(3);
    constexpr uint32_t BLOCK_CT = get_compile_time_arg_val(4);
    constexpr uint32_t NB = H_T / BLOCK_CT;

    compute_kernel_hw_startup(cb_in, cb_out);
    pack_untilize_init<BLOCK_CT, H_T>(cb_in, cb_out);
    cb_wait_front(cb_n, 1);
    const uint32_t n_units = read_tile_value(cb_n, 0, 0);
    for (uint32_t unit = 0; unit < n_units; ++unit) {
        cb_reserve_back(cb_out, 32);
        for (uint32_t b = 0; b < NB; ++b) {
            cb_wait_front(cb_in, BLOCK_CT);
            pack_untilize_block<BLOCK_CT, H_T>(cb_in, 1, cb_out, b);
            cb_pop_front(cb_in, BLOCK_CT);
        }
        cb_push_back(cb_out, 32);
    }
    pack_untilize_uninit(cb_out);
    cb_pop_front(cb_n, 1);
}
