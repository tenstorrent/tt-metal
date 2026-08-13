// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sfpi.h"

namespace ckernel::sfpu {

sfpi_inline sfpi::vBool _sfpu_is_fp16_zero_(const sfpi::vFloat& value) { return value == 0.0f; }

}  // namespace ckernel::sfpu
