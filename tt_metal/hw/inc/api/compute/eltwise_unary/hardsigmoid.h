// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"

namespace ckernel {

ALWI void hardsigmoid_tile_init() { SFPU_INIT_KERNEL_CALL(hardsigmoid, ckernel::sfpu::hardsigmoid_init, APPROX); }

ALWI void hardsigmoid_tile(uint32_t idst) { SFPU_UNARY_NO_PARAM_KERNEL_FN(calculate_hardsigmoid, RC, APPROX, idst); }

}  // namespace ckernel
