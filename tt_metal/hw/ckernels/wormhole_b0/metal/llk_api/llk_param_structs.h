// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

//***
//  Unpack LLK param structs
//***

constexpr std::uint32_t default_tile_dims[2] = {32, 32};

struct llk_unpack_A_params_t {
    std::uint32_t unpA_operand;
};

struct llk_unpack_AB_matmul_params_t {
    std::uint32_t unpA_operand;
    std::uint32_t unpB_operand;
    std::uint32_t transpose_xy_srca;
};

//***
//  Pack LLK param structs
//***

struct llk_relu_config_t {
    std::uint32_t
        ApplyRelu : 16;  // 0 ? no relu, 1 ? val<0=>val=0, 2 ? val<threshold=>val=0, 3 - val>threshold=>val=threshold
    std::uint32_t Threshold : 16;  // fp16
};

union llk_relu_config_u {
    llk_relu_config_t f;
    std::uint32_t val;
};

struct llk_pack_params_t {
    std::uint32_t pack_output;
    llk_relu_config_u relu_config;
    bool srnd_fpu_en;
};
