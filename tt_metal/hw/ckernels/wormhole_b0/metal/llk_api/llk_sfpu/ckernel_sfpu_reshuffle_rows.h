// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_sfpu_reshuffle_rows.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void calculate_reshuffle_rows(uint idx_addr) {
    _calculate_reshuffle_rows_(idx_addr);
}

}  // namespace sfpu
}  // namespace ckernel
