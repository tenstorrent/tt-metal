// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/router_counters_types.h"

// Wormhole has no router counters in its register map, so the table is empty and the profiler option is refused.
// Include through perf_counters/router_counters_inventory.h.

namespace llk::perf::router
{

inline constexpr std::uint32_t ROUTER_REGS_BASE  = 0xFFB20000;
inline constexpr std::uint32_t ROUTER_NOC_STRIDE = 0x10000;
inline constexpr std::uint32_t ROUTER_NOCS       = 2;
inline constexpr std::uint32_t ROUTER_VC_STRIDE  = 0;
inline constexpr std::uint32_t COUNTER_BITS      = 0;
inline constexpr CountKind COUNT_KIND            = CountKind::NONE;

// clang-format off
inline constexpr std::array<Bank, 0> router_banks = {};
// clang-format on

} // namespace llk::perf::router
