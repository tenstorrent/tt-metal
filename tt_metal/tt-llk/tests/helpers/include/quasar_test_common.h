// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "tensor_shape.h"

/**
 * @brief Build a buffer descriptor from the tensor shape / L1 base / format and program it into the
 * BD table entry `buf_desc_id`. Test-side analog of the migrated op inits: it does what the product
 * `bfd_alloc_and_program` does, but with a caller-chosen id (tests pick fixed ids) and without
 * allocating. The register data format is passed separately to the `_llk_*_configure_` call.
 *
 * @tparam MODE: L1 access mode; Strided (PACR/UNPACR_STRIDE tiny-tiles) forces y_dim = z_dim = 1.
 */
template <ckernel::trisc::L1AccessMode MODE = ckernel::trisc::L1AccessMode::Continuous>
inline void program_buf_desc(std::uint32_t buf_desc_id, const ckernel::TensorShape& shape, unsigned base_l1_16B, unsigned data_format)
{
    ckernel::trisc::_configure_buf_desc_table_(buf_desc_id, ckernel::trisc::construct_buf_desc<MODE>(shape, base_l1_16B, data_format));
}

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
