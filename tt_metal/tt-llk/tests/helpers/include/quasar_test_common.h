// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tensor_shape.h"

inline void clear_dest_dvalid_wait_mask(std::uint32_t wait_mask_addr)
{
    volatile std::uint32_t* cfg = reinterpret_cast<volatile std::uint32_t*>(TENSIX_CFG_BASE);
    cfg[wait_mask_addr]         = 0;
}

/**
 * @brief Disables the DEST DVALID handshake for UNPACK.
 */
inline void set_up_zero_dest_dvalid_handshake_for_unpack()
{
    clear_dest_dvalid_wait_mask(UNPACK_TO_DEST_DVALID_CTRL_wait_mask_ADDR32);
}

/**
 * @brief Disables the DEST DVALID handshake for MATH.
 */
inline void set_up_zero_dest_dvalid_handshake_for_math()
{
    clear_dest_dvalid_wait_mask(MATH_DEST_DVALID_CTRL_wait_mask_ADDR32);
}

/**
 * @brief Disables the DEST DVALID handshake for SFPU.
 */
inline void set_up_zero_dest_dvalid_handshake_for_sfpu()
{
    clear_dest_dvalid_wait_mask(SFPU_DEST_DVALID_CTRL_wait_mask_ADDR32);
}

/**
 * @brief Disables the DEST DVALID handshake for PACK.
 */
inline void set_up_zero_dest_dvalid_handshake_for_pack()
{
    clear_dest_dvalid_wait_mask(PACK_DEST_DVALID_CTRL_wait_mask_ADDR32);
}

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
