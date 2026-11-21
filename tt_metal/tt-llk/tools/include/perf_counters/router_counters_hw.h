// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "perf_counters/router_counters_inventory.h"

namespace llk::perf::router
{

// A plain load of the tile's own router register; any core of the tile can read it. On Blackhole this is the low word
// of the 64-bit counter.
inline std::uint32_t read(std::uint32_t noc, std::uint32_t bank_offset, std::uint32_t vc)
{
    return *reinterpret_cast<volatile std::uint32_t*>(ROUTER_REGS_BASE + noc * ROUTER_NOC_STRIDE + bank_offset + vc * ROUTER_VC_STRIDE);
}

// Delta false stores the readings of the selected banks and channels, NoC 0 first, banks in table order, a bank's channels
// added up unless the option keeps them apart; true turns the stored ones into deltas modulo 2^32. values may be volatile.
template <std::uint32_t Option, typename Values>
inline void capture(Values values, bool delta)
{
    static constexpr auto banks = selected_bank_offsets<Option>();
    constexpr std::uint32_t vcs = Option & VC_MASK_ALL;
    constexpr bool per_vc       = (Option & PER_VC) != 0;
    std::uint32_t i             = 0;
    for (std::uint32_t noc = 0; noc < ROUTER_NOCS; noc++)
    {
        for (std::uint32_t b = 0; b < banks.size(); b++)
        {
            std::uint32_t sum = 0;
            for (std::uint32_t vc = 0; vc < NUM_VCS; vc++)
            {
                if (((vcs >> vc) & 1u) == 0)
                {
                    continue;
                }
                const std::uint32_t now = read(noc, banks[b], vc);
                if (per_vc)
                {
                    values[i] = delta ? now - values[i] : now;
                    i++;
                }
                else
                {
                    sum += now;
                }
            }
            if (!per_vc)
            {
                values[i] = delta ? sum - values[i] : sum;
                i++;
            }
        }
    }
}

} // namespace llk::perf::router
