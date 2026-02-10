// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_reshuffle_rows.h"
#include "llk_defs.h"

namespace ckernel {
namespace sfpu {

template <ApproximationMode APPROX_MODE>
inline void calculate_reshuffle_rows(uint idx_addr) {
    _calculate_reshuffle_rows_(idx_addr);
}

}  // namespace sfpu
}  // namespace ckernel
