// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Counts the markers compute_bit_identity.cpp leaves in each flag tile (nonzero elements, each the BF16-exact float
// (0x3F81 + offset) << 16 of a differing element) and writes this core's result page: [differences, tiles checked,
// then up to 30 differing inputs as tile * 1024 + offset].

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t first_tile = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t page = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_flags = 16;
    constexpr uint32_t cb_result = 17;
    constexpr uint32_t fp32_flags = get_compile_time_arg_val(0);
    constexpr uint32_t result_words = 32;
    constexpr auto out_args = TensorAccessorArgs<1>();
    const auto out = TensorAccessor(out_args, out_addr, result_words * 4);

    cb_reserve_back(cb_result, 1);
    volatile tt_l1_ptr uint32_t* result = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_result));
    uint32_t differences = 0;
    for (uint32_t t = first_tile; t < first_tile + num_tiles; t++) {
        cb_wait_front(cb_flags, 1);
        volatile tt_l1_ptr uint32_t* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_flags));
        // FP32 flags: one marker per word; BF16 flags: two per word.
        constexpr uint32_t words = fp32_flags ? 1024 : 512;
        for (uint32_t w = 0; w < words; w++) {
            const uint32_t v = flags[w];
            if (v == 0) {
                continue;
            }
            const uint32_t halves[2] = {fp32_flags ? (v >> 16) : (v & 0xFFFF), fp32_flags ? 0 : (v >> 16)};
            for (uint32_t h = 0; h < 2; h++) {
                if (halves[h] == 0) {
                    continue;
                }
                if (differences < result_words - 2) {
                    result[2 + differences] = t * 1024 + (halves[h] - 0x3F81);
                }
                differences++;
            }
        }
        cb_pop_front(cb_flags, 1);
    }
    result[0] = differences;
    result[1] = num_tiles;
    for (uint32_t i = 2 + (differences < result_words - 2 ? differences : result_words - 2); i < result_words; i++) {
        result[i] = 0xFFFFFFFF;
    }
    noc_async_write(get_write_ptr(cb_result), out.get_noc_addr(page), result_words * 4);
    noc_async_write_barrier();
}
