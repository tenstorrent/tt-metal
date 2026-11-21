// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "perf_counters/noc_counters_inventory.h"

namespace llk::perf::noc
{

// A plain load of the tile's own NIU register; any core of the tile can read it.
inline std::uint32_t read(std::uint32_t noc, std::uint32_t reg)
{
    return *reinterpret_cast<volatile std::uint32_t*>(NIU_STATUS_BASE + noc * NIU_NOC_STRIDE + reg * 4);
}

// Values a capture of the groups yields: the selected counters of every NoC.
template <std::uint32_t GroupMask>
constexpr std::size_t value_count()
{
    return selected_offsets<GroupMask>().size();
}

// Delta false stores the readings of the selected registers, NoC 0 first, in table order; true turns the stored ones into
// deltas modulo 2^32, as nothing resets the counters. values may be volatile.
template <std::uint32_t GroupMask, typename Values>
inline void capture(Values values, bool delta)
{
    static constexpr auto offsets = selected_offsets<GroupMask>();
    for (std::size_t i = 0; i < offsets.size(); i++)
    {
        const std::uint32_t now = *reinterpret_cast<volatile std::uint32_t*>(NIU_STATUS_BASE + offsets[i] * 4);
        values[i]               = delta ? now - values[i] : now;
    }
}

} // namespace llk::perf::noc
