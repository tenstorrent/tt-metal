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
