// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "perf_counters/noc_counters_types.h"

// Blackhole Tensix NIU counters. Include through perf_counters/noc_counters_inventory.h.

namespace llk::perf::noc
{

inline constexpr std::uint32_t NIU_STATUS_BASE = 0xFFB20200; // NoC register space base plus the status file offset 0x200
inline constexpr std::uint32_t NIU_NOC_STRIDE  = 0x10000;
inline constexpr std::uint32_t NIU_NOCS        = 2;
// Bytes in one data word of the *_DATA_WORD_* counters: a 64-byte flit (tt-isa-documentation BlackholeA0/NoC/Counters.md).
inline constexpr std::uint32_t NIU_DATA_WORD_BYTES = 64;

} // namespace llk::perf::noc
