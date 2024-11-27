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

struct llk_unpack_AB_params_t {
    std::uint32_t unpA_operand;
    std::uint32_t unpB_operand;
};

struct llk_unpack_reduce_params_t {
    std::uint32_t unpA_operand;
    // std::uint32_t unpB_operand;  // TODO: Should be removed when llk hw args are cleaned up
};

struct llk_unpack_tilize_params_t {
    std::uint32_t unpA_operand;
    std::uint32_t unpA_block_c_dim;
};

struct llk_unpack_untilize_params_t {
    std::uint32_t unpA_operand;
};

//***
//  Math LLK param structs
//***

struct llk_math_eltwise_binary_params_t {
    std::int32_t unused;
};

struct llk_math_eltwise_unary_params_t {
    std::int32_t sfpu_params[6];  // TODO: Fix how we assign this from hlkc
    std::int32_t unused;
};

struct llk_math_matmul_params_t {
    std::int32_t unused;
};

struct llk_math_reduce_params_t {
    std::int32_t unused;
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
