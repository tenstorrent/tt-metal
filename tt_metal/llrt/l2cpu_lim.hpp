// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ll_api {

// LIM aperture of a Blackhole L2CPU tile. H2D/D2H sockets map it with a window anchored at the LIM
// base rather than at 0, because LIM does not start at 0, so callers writing through that window
// convert absolute LIM addresses to window-relative offsets. Anything addressed through it must lie
// entirely within [kL2cpuLimBase, kL2cpuLimTlbEnd).
inline constexpr uint64_t kL2cpuLimBase = 0x08000000ULL;
inline constexpr uint64_t kL2cpuLimTlbSize = 2ULL * 1024 * 1024;
inline constexpr uint64_t kL2cpuLimTlbEnd = kL2cpuLimBase + kL2cpuLimTlbSize;

}  // namespace ll_api
