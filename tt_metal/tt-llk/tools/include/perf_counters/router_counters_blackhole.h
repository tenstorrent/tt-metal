// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/router_counters_types.h"

// Blackhole: 64-bit head flit counters (one count per packet) on two output ports, none on input or ejection.
// A capture reads the low word.

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
