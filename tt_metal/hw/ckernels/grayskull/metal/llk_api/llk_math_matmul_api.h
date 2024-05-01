// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_matmul.h"

/*************************************************************************
 * LLK MATMUL
 *************************************************************************/

template <int NUM_FIDELITY_PHASES>
inline void llk_math_matmul_init(
    const std::uint32_t operandA /*not used*/,
    const std::uint32_t operandB /*not used*/,
    const std::uint32_t transpose = 0,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {

    _llk_math_matmul_init_<NUM_FIDELITY_PHASES, DstTileFaceLayout::RowMajor>(
        transpose,
        ct_dim,
        rt_dim,
        kt_dim);
}


template <int NUM_FIDELITY_PHASES, uint32_t num_faces = 4>
inline void llk_math_matmul(
    const uint dst_index,
    const bool transpose = false,
    const std::uint32_t ct_dim = 1,
    const std::uint32_t rt_dim = 1,
    const std::uint32_t kt_dim = 1) {
    for (std::uint32_t rt=0; rt<rt_dim; rt++) {
        for (std::uint32_t ct=0; ct<ct_dim; ct++) {
            _llk_math_matmul_<NUM_FIDELITY_PHASES, DstTileFaceLayout::RowMajor, num_faces>(dst_index+rt*ct_dim+ct, transpose);
        }
    }
}
