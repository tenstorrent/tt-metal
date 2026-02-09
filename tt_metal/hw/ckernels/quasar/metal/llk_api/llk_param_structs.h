// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

//***
//  Unpack LLK param structs
//***

constexpr std::uint32_t default_tile_dims[2] = {32, 32};

//***
//  Math LLK param structs
//***

struct llk_math_eltwise_unary_params_t {
    std::int32_t sfpu_params[6];  // TODO: Fix how we assign this from hlkc
    std::int32_t unused;
};

//***
//  Pack LLK param structs
//***

struct llk_pack_params_t {
    std::uint32_t pack_output;
};
