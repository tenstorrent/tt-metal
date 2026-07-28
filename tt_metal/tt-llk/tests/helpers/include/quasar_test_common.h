// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tensor_shape.h"

/**
 * @brief Disables the DEST DVALID handshake for PACK.
 */
inline void set_up_zero_dest_dvalid_handshake_for_pack()
{
    auto cfg                                    = (std::uint32_t volatile*)TENSIX_CFG_BASE;
    cfg[PACK_DEST_DVALID_CTRL_wait_mask_ADDR32] = 0;
}

/**
 * @brief Populates TensorShape struct args from runtime test parameters.
 *
 * @param params: Runtime parameters passed through pytest.
 */
template <typename Params>
inline ckernel::TensorShape tensor_shape_from_params(const Params& params)
{
    return {
        static_cast<std::uint8_t>(params.TEST_FACE_R_DIM),
        static_cast<std::uint8_t>(params.TEST_FACE_C_DIM),
        static_cast<std::uint8_t>(params.num_faces_r_dim_A),
        static_cast<std::uint8_t>(params.num_faces_c_dim_A),
    };
}
