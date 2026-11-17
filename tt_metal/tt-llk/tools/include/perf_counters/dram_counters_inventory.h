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
    return Layout {0, 0, 0, 1, 0, 0, 0};
}

// Register address of a counter on a tile with the given noc0 y: the offset moves with the NoC instance and the
// register instance of the tile.
constexpr std::uint64_t niu_address(Arch arch, std::uint32_t noc, std::uint32_t tile_noc0_y, std::uint32_t offset)
{
    const Layout l = layout_for(arch);
    const std::uint32_t instance = (tile_noc0_y + l.tile_instance_bias) % l.tile_instance_count;
    return l.niu_base + std::uint64_t(noc) * l.noc_instance_stride + std::uint64_t(instance) * l.tile_instance_stride + offset;
}

} // namespace llk::perf::dram
