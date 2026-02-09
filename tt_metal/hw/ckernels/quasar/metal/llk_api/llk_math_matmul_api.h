// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_matmul.h"

/*************************************************************************
 * LLK MATMUL
 *************************************************************************/

template <int NUM_FIDELITY_PHASES, bool EN_DI = false, bool EN_X2 = false>
inline void llk_math_matmul_init(const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
    _llk_math_matmul_init_<static_cast<ckernel::MathFidelity>(NUM_FIDELITY_PHASES), EN_DI, EN_X2>(ct_dim, rt_dim);
}

inline void llk_math_matmul(
    const std::uint32_t dst_index, const std::uint32_t ct_dim = 1, const std::uint32_t rt_dim = 1) {
    _llk_math_matmul_tile_(dst_index);
}
