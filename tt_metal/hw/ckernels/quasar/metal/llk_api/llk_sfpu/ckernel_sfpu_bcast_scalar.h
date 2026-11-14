// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The SFPU scalar broadcast from DEST (sfpu_bcast_scalar_* / sfpu_mul_bcast_scalar) is not ported to Quasar. The
// Compute API keeps its entry points on every arch; a kernel that calls one fails to build.

#include <cstdint>

#include "ckernel_defs.h"

namespace ckernel {
namespace sfpu {

[[gnu::error("sfpu_bcast_scalar_* / sfpu_mul_bcast_scalar are not supported on Quasar")]] void
sfpu_bcast_scalar_is_not_supported_on_quasar();

inline void _sfpu_bcast_scalar_init_() { sfpu_bcast_scalar_is_not_supported_on_quasar(); }

inline void _sfpu_bcast_scalar_load_(
    [[maybe_unused]] std::uint32_t dst_index_scalar,
    [[maybe_unused]] std::uint32_t unused_in1,
    [[maybe_unused]] std::uint32_t unused_out) {
    sfpu_bcast_scalar_is_not_supported_on_quasar();
}

template <int ITERATIONS>
inline void _calculate_mul_bcast_scalar_(
    [[maybe_unused]] std::uint32_t dst_index,
    [[maybe_unused]] std::uint32_t unused_in1,
    [[maybe_unused]] std::uint32_t unused_out,
    [[maybe_unused]] VectorMode vector_mode) {
    sfpu_bcast_scalar_is_not_supported_on_quasar();
}

}  // namespace sfpu
}  // namespace ckernel
