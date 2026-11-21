// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "perf_counters/dram_counters_types.h"

// Quasar: the NIU status registers have the same offsets as on the other architectures inside a NoC register space
// that starts at 0x02000000 on a tile with a Tensix-style NoC overlay. How a host reaches a DRAM tile, how many NoCs
// it has and the width of a data word are not known, so the table is empty and the layout has zeros.
// Include through perf_counters/dram_counters_inventory.h.

namespace llk::perf::dram::quasar
{

inline constexpr Layout layout = {0u, 0u, 0u, 1u, 0u, 0u, 0u};

inline constexpr std::array<Entry, 0> niu_counters = {};

} // namespace llk::perf::dram::quasar
