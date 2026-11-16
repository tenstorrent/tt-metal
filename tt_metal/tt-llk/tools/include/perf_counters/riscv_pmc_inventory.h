// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/riscv_pmc_types.h"

#if defined(ARCH_BLACKHOLE)
#include "perf_counters/riscv_pmc_blackhole.h"
#elif defined(ARCH_WORMHOLE) || defined(ARCH_WORMHOLE_B0)
#include "perf_counters/riscv_pmc_wormhole.h"
#elif defined(ARCH_QUASAR)
#include "perf_counters/riscv_pmc_quasar.h"
#else
#error "perf_counters/riscv_pmc_inventory.h needs ARCH_BLACKHOLE, ARCH_WORMHOLE, ARCH_WORMHOLE_B0 or ARCH_QUASAR"
#endif

namespace llk::perf::riscv
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

// Empty for a core kind the architecture does not have.
constexpr Table events_for(CoreKind kind)
{
#if defined(ARCH_QUASAR)
    switch (kind)
    {
        case CoreKind::TRISC:
            return detail::as_table(trisc_events);
        case CoreKind::DM:
            return detail::as_table(dm_events);
        default:
            return Table {nullptr, 0};
    }
#else
    return kind == CoreKind::BABY ? detail::as_table(baby_events) : Table {nullptr, 0};
#endif
}

} // namespace llk::perf::riscv
