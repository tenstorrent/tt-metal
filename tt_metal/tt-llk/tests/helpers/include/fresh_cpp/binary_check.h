// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel
{
// Quasar still needs the accumulator format for its destination tile bound;
// current WH/BH derive that bound from the configured destination state.
template <DstSync DST_SYNC, bool DST_ACCUM>
inline void campaign_sfpu_binary_check_(std::uint32_t in0, std::uint32_t in1, std::uint32_t out, VectorMode vector_mode)
{
#if defined(ARCH_QUASAR)
    _sfpu_binary_check_<DST_SYNC, DST_ACCUM>(in0, in1, out, vector_mode);
#else
    _sfpu_binary_check_<DST_SYNC>(in0, in1, out, vector_mode);
#endif
}
} // namespace ckernel
