// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "sfpi.h"

namespace ckernel::sfpu::ttpoly {
using namespace ::sfpi;
using ::sfpi::DataLayout;
inline vFloat sfpu_mad(vFloat a, vFloat b, vFloat c) {
    return __builtin_rvtt_sfpmad(a.get(), b.get(), c.get(), SFPMAD_MOD1_OFFSET_NONE);
}

}  // namespace ckernel::sfpu::ttpoly
