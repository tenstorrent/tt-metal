// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/router_counters_types.h"

// Blackhole Tensix router counters. Include through perf_counters/router_counters_inventory.h.
// The counters are 64-bit and count head flits, so one count is one packet. Only two output ports have them; a packet
// that ejects into the tile's NoC interface is not counted and there is no input side. A capture reads the low word of
// each, which gives a delta modulo 2^32.

namespace llk::perf::router
{

inline constexpr std::uint32_t ROUTER_REGS_BASE  = 0xFFB20000; // NoC register space base
inline constexpr std::uint32_t ROUTER_NOC_STRIDE = 0x10000;
inline constexpr std::uint32_t ROUTER_NOCS       = 2;
inline constexpr std::uint32_t ROUTER_VC_STRIDE  = 8; // two words per virtual channel, low word first
inline constexpr std::uint32_t COUNTER_BITS      = 64;
inline constexpr CountKind COUNT_KIND            = CountKind::HEAD_FLIT;

// clang-format off
inline constexpr std::array<Bank, 2> router_banks = {
    {
    {RouterPort::OUT_PORT1, 0x500},
    {RouterPort::OUT_PORT2, 0x580}}};
// clang-format on

} // namespace llk::perf::router
