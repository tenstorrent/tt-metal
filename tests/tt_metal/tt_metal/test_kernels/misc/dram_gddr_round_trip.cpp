// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compile_time_args.h"

// A CCE reaches GDDR through its remapper at a flat 64-bit address, not over the NOC, so the two
// GDDR addresses arrive as low/high pairs of 32-bit compile-time args.
constexpr uint64_t join_addr(uint32_t low, uint32_t high) { return (static_cast<uint64_t>(high) << 32) | low; }

void kernel_main() {
    constexpr uint64_t src_gddr_addr = join_addr(get_compile_time_arg_val(0), get_compile_time_arg_val(1));
    constexpr uint64_t dst_gddr_addr = join_addr(get_compile_time_arg_val(2), get_compile_time_arg_val(3));
    constexpr uint32_t staging_addr = get_compile_time_arg_val(4);
    constexpr uint32_t num_words = get_compile_time_arg_val(5);

    volatile uint32_t* src = reinterpret_cast<volatile uint32_t*>(src_gddr_addr);
    volatile uint32_t* dst = reinterpret_cast<volatile uint32_t*>(dst_gddr_addr);
    volatile tt_l1_ptr uint32_t* staging = reinterpret_cast<tt_l1_ptr uint32_t*>(staging_addr);

    for (uint32_t i = 0; i < num_words; i++) {
        staging[i] = src[i];
    }
    for (uint32_t i = 0; i < num_words; i++) {
        dst[i] = staging[i];
    }
}
