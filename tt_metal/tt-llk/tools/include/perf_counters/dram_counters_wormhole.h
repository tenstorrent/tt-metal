// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "perf_counters/dram_counters_types.h"

// Wormhole DRAM channel tiles: their NIU registers sit in NoC address space, instance (noc0 y + 2) modulo 3, NoC 1 0x8000
// above NoC 0. Include through perf_counters/dram_counters_inventory.h.

namespace llk::perf::dram::wormhole
{

inline constexpr Layout layout = {0x100080000ull, 0x8000u, 0x10000u, 3u, 2u, 32u, 2u};

} // namespace llk::perf::dram::wormhole
