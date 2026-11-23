// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "perf_counters/dram_counters_types.h"

// Quasar: how a host reaches a DRAM tile, its NoC count and the data word width are unknown, so the table is empty.
// Include through perf_counters/dram_counters_inventory.h.

namespace llk::perf::dram::quasar
{

inline constexpr Layout layout = {0u, 0u, 0u, 1u, 0u, 0u, 0u};

inline constexpr std::array<Entry, 0> niu_counters = {};

} // namespace llk::perf::dram::quasar
