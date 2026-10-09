// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel_defs.h"
#include "llk_assert.h"
#include "llk_defs.h"

namespace ckernel {
namespace sfpu {

// sub_int (Blackhole / Wormhole tt-llk _sub_int_) is not ported to Quasar. The Compute API keeps
// sub_int_tile on every arch, so its kernel entry point rejects the call here.
template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _sub_int_(
    [[maybe_unused]] const std::uint32_t dst_index_in0,
    [[maybe_unused]] const std::uint32_t dst_index_in1,
    [[maybe_unused]] const std::uint32_t dst_index_out) {
    LLK_ASSERT(false, "sub_int_tile is not supported on Quasar");
}

}  // namespace sfpu
}  // namespace ckernel
