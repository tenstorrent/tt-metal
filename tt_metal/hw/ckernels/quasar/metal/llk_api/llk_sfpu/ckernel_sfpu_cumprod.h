// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ckernel {
namespace sfpu {

// cumprod_tile is not ported to Quasar. cumprod.h still compiles there; a kernel that calls
// cumprod_tile fails to build.
[[gnu::error("cumprod_tile is not supported on Quasar")]] void cumprod_tile_is_not_supported_on_quasar();

template <bool APPROXIMATION_MODE /*unused*/>
inline void calculate_cumprod([[maybe_unused]] const bool first) {
    cumprod_tile_is_not_supported_on_quasar();
}

template <bool APPROXIMATION_MODE /*unused*/>
inline void cumprod_init() {
    cumprod_tile_is_not_supported_on_quasar();
}

}  // namespace sfpu
}  // namespace ckernel
