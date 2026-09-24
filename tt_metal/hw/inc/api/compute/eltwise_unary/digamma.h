// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "api/compute/common_globals.h"
#ifdef TRISC_MATH
#include "ckernel_sfpu_digamma.h"
#endif

namespace ckernel {

ALWI void digamma_tile(std::uint32_t idst) { MATH((sfpu::Digamma<APPROX>::run(idst))); }

ALWI void digamma_tile_init() { MATH((sfpu::Digamma<APPROX>::init())); }

}  // namespace ckernel
