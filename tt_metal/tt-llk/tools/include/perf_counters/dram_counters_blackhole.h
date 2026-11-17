// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "perf_counters/dram_counters_types.h"

// Blackhole DRAM channel tile counters. Include through perf_counters/dram_counters_inventory.h.
// The NIU slave counters are the status registers 0x33 to 0x3B of the tile NIU: offset = 0x200 + 4 * index.

namespace llk::perf::dram::blackhole
{

inline constexpr Layout layout = {0xFFB20000u, 0x10000u, 64u, 2u};

inline constexpr std::array<Entry, 6> niu_counters = {{
    {DramCounter::NIU_SLV_RD_REQ_RECEIVED, 0x2D4},
    {DramCounter::NIU_SLV_RD_DATA_WORD_SENT, 0x2CC},
    {DramCounter::NIU_SLV_NONPOSTED_WR_REQ_RECEIVED, 0x2E8},
    {DramCounter::NIU_SLV_POSTED_WR_REQ_RECEIVED, 0x2EC},
    {DramCounter::NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED, 0x2E0},
    {DramCounter::NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED, 0x2E4},
}};

} // namespace llk::perf::dram::blackhole
