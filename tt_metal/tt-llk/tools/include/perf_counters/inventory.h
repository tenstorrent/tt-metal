// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

#include "perf_counters/types.h"

#if defined(ARCH_BLACKHOLE)
#include "perf_counters/blackhole.h"
#elif defined(ARCH_WORMHOLE) || defined(ARCH_WORMHOLE_B0)
#include "perf_counters/wormhole.h"
#elif defined(ARCH_QUASAR)
#error "Quasar counter tables are not in tt-llk yet"
#else
#error "perf_counters/inventory.h needs ARCH_BLACKHOLE, ARCH_WORMHOLE or ARCH_WORMHOLE_B0"
#endif

namespace llk::perf
{

struct Table
{
    const Entry* data;
    std::size_t size;
};

namespace detail
{

template <std::size_t N>
constexpr Table as_table(const std::array<Entry, N>& entries)
{
    return Table {entries.data(), entries.size()};
}

} // namespace detail

// The select table of one bank. The L1 bank has one table per mux position; positions the hardware does
// not decode return an empty table.
constexpr Table table_for(Bank bank, std::uint8_t l1_mux = 0)
{
    switch (bank)
    {
        case Bank::INSTRN_THREAD:
            return detail::as_table(instrn_counters);
        case Bank::FPU:
            return detail::as_table(fpu_counters);
        case Bank::TDMA_UNPACK:
            return detail::as_table(unpack_counters);
        case Bank::TDMA_PACK:
            return detail::as_table(pack_counters);
        case Bank::L1:
            switch (l1_mux)
            {
                case 0:
                    return detail::as_table(l1_0_counters);
                case 1:
                    return detail::as_table(l1_1_counters);
                case 2:
                    return detail::as_table(l1_2_counters);
                case 3:
                    return detail::as_table(l1_3_counters);
                case 4:
                    return detail::as_table(l1_4_counters);
                case 5:
                    return detail::as_table(l1_5_counters);
                default:
                    return Table {nullptr, 0};
            }
    }
    return Table {nullptr, 0};
}

} // namespace llk::perf
