// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "llk_math_common.h"
#include "perf.h"
#include "tensor_shape.h"

inline void perf_unpack_set_srcb_once_then_srca_per_face(std::uint32_t srcb_once_count, std::uint32_t srca_per_face_count)
{
    _perf_unpack_loop_set_valid<false /*set_a*/, true /*set_b*/>(srcb_once_count);
    _perf_unpack_loop_set_valid<true /*set_a*/, false /*set_b*/>(srca_per_face_count);
}

inline void perf_math_clear_srca_per_face_then_srcb_once(std::uint32_t srca_per_face_count, std::uint32_t srcb_once_count)
{
    _perf_math_loop_clear_valid<true /*clear_a*/, false /*clear_b*/>(srca_per_face_count);
    _perf_math_loop_clear_valid<false /*clear_a*/, true /*clear_b*/>(srcb_once_count);
}

template <bool implied_math_format, bool fp32_dest_acc_en>
inline void configure_math_hardware_for_float32_int32_or_default(DataFormat math_format, DataFormat dest_format)
{
    if constexpr (fp32_dest_acc_en)
    {
        if (dest_format == DataFormat::Float32)
        {
            _llk_math_srcAB_hw_configure_<implied_math_format, true /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
        }
        else if (dest_format == DataFormat::Int32)
        {
            _llk_math_srcAB_hw_configure_<implied_math_format, false /*fp32_dest*/, true /*int32_dest*/>(math_format, math_format);
        }
        else
        {
            _llk_math_srcAB_hw_configure_<implied_math_format, false /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
        }
    }
    else
    {
        _llk_math_srcAB_hw_configure_<implied_math_format, false /*fp32_dest*/, false /*int32_dest*/>(math_format, math_format);
    }
}

template <ckernel::dest_dvalid_client thread>
inline void set_up_fpu_to_pack_dest_dvalid_chain()
{
    ckernel::set_up_dest_dvalid_per_thread<thread>({ckernel::dest_dvalid_client::FPU, ckernel::dest_dvalid_client::PACK});
}

template <ckernel::dest_dvalid_client thread>
inline void set_up_unpack_to_pack_dest_dvalid_chain()
{
    ckernel::set_up_dest_dvalid_per_thread<thread>({ckernel::dest_dvalid_client::UNPACK, ckernel::dest_dvalid_client::PACK});
}

template <ckernel::dest_dvalid_client thread>
inline void set_up_unpack_to_fpu_to_pack_dest_dvalid_chain()
{
    ckernel::set_up_dest_dvalid_per_thread<thread>({ckernel::dest_dvalid_client::UNPACK, ckernel::dest_dvalid_client::FPU, ckernel::dest_dvalid_client::PACK});
}

template <ckernel::dest_dvalid_client thread>
inline void set_up_fpu_to_sfpu_to_pack_dest_dvalid_chain()
{
    ckernel::set_up_dest_dvalid_per_thread<thread>({ckernel::dest_dvalid_client::FPU, ckernel::dest_dvalid_client::SFPU, ckernel::dest_dvalid_client::PACK});
}

template <ckernel::dest_dvalid_client thread>
inline void set_up_unpack_to_sfpu_to_pack_dest_dvalid_chain()
{
    ckernel::set_up_dest_dvalid_per_thread<thread>({ckernel::dest_dvalid_client::UNPACK, ckernel::dest_dvalid_client::SFPU, ckernel::dest_dvalid_client::PACK});
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
