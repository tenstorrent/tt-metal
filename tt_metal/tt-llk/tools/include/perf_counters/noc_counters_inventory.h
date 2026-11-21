// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/noc_counters_types.h"

#if defined(ARCH_BLACKHOLE)
#include "perf_counters/noc_counters_blackhole.h"
#elif defined(ARCH_WORMHOLE) || defined(ARCH_WORMHOLE_B0)
#include "perf_counters/noc_counters_wormhole.h"
#elif defined(ARCH_QUASAR)
#include "perf_counters/noc_counters_quasar.h"
#else
#error "perf_counters/noc_counters_inventory.h needs ARCH_BLACKHOLE, ARCH_WORMHOLE, ARCH_WORMHOLE_B0 or ARCH_QUASAR"
#endif

namespace llk::perf::noc
{

namespace detail
{

constexpr std::size_t count_for(std::uint32_t group_mask)
{
    std::size_t n = 0;
    for (const Entry& e : noc_counters)
    {
        n += (group_mask >> static_cast<unsigned>(e.group)) & 1u;
    }
    return n;
}

} // namespace detail

// The status register indices of the selected groups, in table order. A capture reads this list once per NoC.
template <std::uint32_t GroupMask>
constexpr std::array<std::uint8_t, detail::count_for(GroupMask)> selected_regs()
{
    std::array<std::uint8_t, detail::count_for(GroupMask)> out {};
    std::size_t i = 0;
    for (const Entry& e : noc_counters)
    {
        if ((GroupMask >> static_cast<unsigned>(e.group)) & 1u)
        {
            out[i++] = e.reg;
        }
    }
    return out;
}

// Word offsets from the status file of NoC 0 of the selected counters of every NoC, NoC 0 first, in table order.
template <std::uint32_t GroupMask>
constexpr std::array<std::uint16_t, NIU_NOCS * detail::count_for(GroupMask)> selected_offsets()
{
    constexpr auto regs = selected_regs<GroupMask>();
    std::array<std::uint16_t, NIU_NOCS * regs.size()> out {};
    for (std::size_t i = 0; i < out.size(); i++)
    {
        out[i] = static_cast<std::uint16_t>((i / regs.size()) * (NIU_NOC_STRIDE / 4) + regs[i % regs.size()]);
    }
    return out;
}

} // namespace llk::perf::noc
