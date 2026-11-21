// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "sfpi.h"
// WH cbrt runs stock's kernel, defined in ckernel_sfpu_cbrt.h, which includes this file.
namespace ckernel::sfpu {
template <bool, bool, int>
inline void calculate_cube_root();
}
