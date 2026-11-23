// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/router_counters_types.h"

// Quasar: 32-bit counters on four output and four input ports. A count is a flit (head, body or tail), not a packet;
// unproven on silicon.

namespace llk::perf::router
{

inline constexpr std::uint32_t ROUTER_REGS_BASE  = 0x02000000; // NoC register space base
inline constexpr std::uint32_t ROUTER_NOC_STRIDE = 0x10000;
inline constexpr std::uint32_t ROUTER_NOCS       = 1;
inline constexpr std::uint32_t ROUTER_VC_STRIDE  = 4;
inline constexpr std::uint32_t COUNTER_BITS      = 32;
inline constexpr CountKind COUNT_KIND            = CountKind::ALL_FLIT;

// clang-format off
inline constexpr std::array<Bank, 8> router_banks = {
    {
    {RouterPort::OUT_PORT1, 0x600},
    {RouterPort::OUT_PORT2, 0x640},
    {RouterPort::OUT_PORT3, 0x680},
    {RouterPort::OUT_PORT4, 0x6C0},
    {RouterPort::IN_PORT1, 0x700},
    {RouterPort::IN_PORT2, 0x740},
    {RouterPort::IN_PORT3, 0x780},
    {RouterPort::IN_PORT4, 0x7C0}}};
// clang-format on

} // namespace llk::perf::router
