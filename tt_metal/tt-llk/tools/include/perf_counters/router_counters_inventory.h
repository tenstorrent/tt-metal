// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/router_counters_types.h"

#if defined(ARCH_BLACKHOLE)
#include "perf_counters/router_counters_blackhole.h"
#elif defined(ARCH_WORMHOLE) || defined(ARCH_WORMHOLE_B0)
#include "perf_counters/router_counters_wormhole.h"
#elif defined(ARCH_QUASAR)
#include "perf_counters/router_counters_quasar.h"
#else
#error "perf_counters/router_counters_inventory.h needs ARCH_BLACKHOLE, ARCH_WORMHOLE, ARCH_WORMHOLE_B0 or ARCH_QUASAR"
#endif

namespace llk::perf::router
{

namespace detail
{

constexpr std::size_t popcount(std::uint32_t v)
{
    std::size_t n = 0;
    for (; v != 0; v &= v - 1)
    {
        n++;
    }
    return n;
}

constexpr std::uint32_t table_bank_mask()
{
    return (1u << router_banks.size()) - 1u;
}

} // namespace detail

// The banks the option selects, as table indices of the architecture; bank bits past the table are dropped.
constexpr std::uint32_t selected_bank_mask(std::uint32_t option)
{
    return (option >> BANK_SHIFT) & detail::table_bank_mask();
}

// Byte offsets of the selected banks from the NoC register base, in table order.
template <std::uint32_t Option>
constexpr std::array<std::uint16_t, detail::popcount(selected_bank_mask(Option))> selected_bank_offsets()
{
    std::array<std::uint16_t, detail::popcount(selected_bank_mask(Option))> out {};
    std::size_t n = 0;
    for (std::size_t b = 0; b < router_banks.size(); b++)
    {
        if ((selected_bank_mask(Option) >> b) & 1u)
        {
            out[n++] = router_banks[b].offset;
        }
    }
    return out;
}

// The selected virtual channels, ascending.
template <std::uint32_t Option>
constexpr std::array<std::uint8_t, detail::popcount(Option& VC_MASK_ALL)> selected_vcs()
{
    std::array<std::uint8_t, detail::popcount(Option & VC_MASK_ALL)> out {};
    std::size_t n = 0;
    for (std::uint32_t vc = 0; vc < NUM_VCS; vc++)
    {
        if ((Option >> vc) & 1u)
        {
            out[n++] = static_cast<std::uint8_t>(vc);
        }
    }
    return out;
}

// Values a capture of the option yields: NoCs x banks x (selected virtual channels when kept apart, else 1).
constexpr std::size_t value_count(std::uint32_t option)
{
    return ROUTER_NOCS * detail::popcount(selected_bank_mask(option)) * ((option & PER_VC) ? detail::popcount(option & VC_MASK_ALL) : 1u);
}

} // namespace llk::perf::router
