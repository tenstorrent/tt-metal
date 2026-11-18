// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>
#include "api/compile_time_args.h"

// A kernel that names a 1x1 subblock as compile-time args (minimal_matmul: subblock_h, subblock_w): every call is one
// tile
constexpr bool llk_unpack_AB_matmul_named_1x1() {
#ifdef KERNEL_COMPILE_TIME_ARG_MAP
    std::uint32_t h = 0, w = 0;
    for (const auto& [arg_name, arg_value] : named_args_map) {
        if (arg_name == std::string_view("subblock_h")) {
            h = arg_value;
        } else if (arg_name == std::string_view("subblock_w")) {
            w = arg_value;
        }
    }
    return h == 1 && w == 1;
#else
    return false;
#endif
}

// The k steps per output block a kernel names (minimal_matmul: K_block_tiles x K_num_blocks), else 0
constexpr std::uint32_t llk_unpack_AB_matmul_named_k_steps() {
#ifdef KERNEL_COMPILE_TIME_ARG_MAP
    std::uint32_t kb = 0, nb = 0;
    for (const auto& [arg_name, arg_value] : named_args_map) {
        if (arg_name == std::string_view("K_block_tiles")) {
            kb = arg_value;
        } else if (arg_name == std::string_view("K_num_blocks")) {
            nb = arg_value;
        }
    }
    return kb * nb;
#else
    return 0;
#endif
}

// A named 1x1 kernel with 32 or more k steps per output block: its one-tile unpack binds the kernel (tt_dit 32x2048x32
// LoFi -2.6 %); with fewer the reader binds and the faster unpack only shifts its timing (flux2 1024x128x768 LoFi +0.4
// %)
constexpr bool llk_unpack_AB_matmul_named_1x1_long_k() {
    return llk_unpack_AB_matmul_named_1x1() && llk_unpack_AB_matmul_named_k_steps() >= 32;
}
