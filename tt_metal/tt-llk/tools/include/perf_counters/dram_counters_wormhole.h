// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include "perf_counters/dram_counters_types.h"

// Wormhole DRAM channel tile counters; include through perf_counters/dram_counters_inventory.h. The NIU registers sit
// in NoC address space, instance noc0 y modulo 3, NoC 1 0x8000 above NoC 0, worker status offsets, 32 B data words.

namespace llk::perf::dram::wormhole
{

inline constexpr Layout layout = {0x100080000ull, 0x8000u, 0x10000u, 3u, 2u, 32u, 2u};

inline constexpr std::array<Entry, 6> niu_counters = {{
    {DramCounter::NIU_SLV_RD_REQ_RECEIVED, 0x2D4},
    {DramCounter::NIU_SLV_RD_DATA_WORD_SENT, 0x2CC},
    {DramCounter::NIU_SLV_NONPOSTED_WR_REQ_RECEIVED, 0x2E8},
    {DramCounter::NIU_SLV_POSTED_WR_REQ_RECEIVED, 0x2EC},
    {DramCounter::NIU_SLV_NONPOSTED_WR_DATA_WORD_RECEIVED, 0x2E0},
    {DramCounter::NIU_SLV_POSTED_WR_DATA_WORD_RECEIVED, 0x2E4},
}};

} // namespace llk::perf::dram::wormhole
