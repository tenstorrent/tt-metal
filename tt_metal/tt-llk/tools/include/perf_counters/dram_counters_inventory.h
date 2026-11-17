// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "perf_counters/dram_counters_blackhole.h"
#include "perf_counters/dram_counters_quasar.h"
#include "perf_counters/dram_counters_types.h"
#include "perf_counters/dram_counters_wormhole.h"

namespace llk::perf::dram
{

namespace detail
{

template <std::size_t N>
constexpr Table as_table(const std::array<Entry, N>& entries)
{
    return Table {entries.data(), entries.size()};
}

} // namespace detail

// The host picks the architecture at run time, so there is no ARCH_* switch here.
constexpr Table niu_table(Arch arch)
{
    switch (arch)
    {
        case Arch::BLACKHOLE:
            return detail::as_table(blackhole::niu_counters);
        case Arch::WORMHOLE:
            return detail::as_table(wormhole::niu_counters);
        case Arch::QUASAR:
            return detail::as_table(quasar::niu_counters);
    }
    return Table {nullptr, 0};
}

constexpr Layout layout_for(Arch arch)
{
    switch (arch)
    {
        case Arch::BLACKHOLE:
            return blackhole::layout;
        case Arch::WORMHOLE:
            return wormhole::layout;
        case Arch::QUASAR:
            return quasar::layout;
    }
    return Layout {0, 0, 0, 0};
}

// Register address of a counter on the tile: the offset moves with the NoC instance.
constexpr std::uint32_t niu_address(Arch arch, std::uint32_t noc, std::uint32_t offset)
{
    const Layout l = layout_for(arch);
    return l.niu_base + noc * l.noc_instance_stride + offset;
}

} // namespace llk::perf::dram
