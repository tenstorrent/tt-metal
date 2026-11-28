// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "perf_counters/noc_counters_types.h"

// Quasar NIU counters (one NoC). Include through perf_counters/noc_counters_inventory.h.

namespace llk::perf::noc
{

inline constexpr std::uint32_t NIU_STATUS_BASE = 0x02000200; // NoC register space base plus the status file offset 0x200
inline constexpr std::uint32_t NIU_NOC_STRIDE  = 0x10000;
inline constexpr std::uint32_t NIU_NOCS        = 1;
// Bytes in one data word of the *_DATA_WORD_* counters: a 2048-bit flit, checked on the Quasar emulator.
inline constexpr std::uint32_t NIU_DATA_WORD_BYTES = 256;

} // namespace llk::perf::noc
