// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tensor_shape.h"

/**
 * @brief Populates TensorShape from explicit dimensions.
 */
inline ckernel::TensorShape tensor_shape_from_dimensions(
    std::uint32_t face_r_dim, std::uint32_t face_c_dim, std::uint32_t num_faces_r_dim, std::uint32_t num_faces_c_dim)
{
    return {
        static_cast<std::uint8_t>(face_r_dim),
        static_cast<std::uint8_t>(face_c_dim),
        static_cast<std::uint8_t>(num_faces_r_dim),
        static_cast<std::uint8_t>(num_faces_c_dim),
    };
}

#ifdef SPEED_OF_LIGHT
#define TENSOR_SHAPE_FROM_PARAMS(params) tensor_shape_from_dimensions(TEST_FACE_R_DIM, TEST_FACE_C_DIM, num_faces_r_dim_A, num_faces_c_dim_A)
#else
/**
 * @brief Populates TensorShape struct args from runtime test parameters.
 *
 * @param params: Runtime parameters passed through pytest.
 */
template <typename Params>
inline ckernel::TensorShape tensor_shape_from_params(const Params& params)
{
    return tensor_shape_from_dimensions(params.TEST_FACE_R_DIM, params.TEST_FACE_C_DIM, params.num_faces_r_dim_A, params.num_faces_c_dim_A);
}

#define TENSOR_SHAPE_FROM_PARAMS(params) tensor_shape_from_params(params)
#endif
