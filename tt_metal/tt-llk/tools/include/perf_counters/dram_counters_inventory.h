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

constexpr ControllerLayout controller_layout_for(Arch arch)
{
    return arch == Arch::BLACKHOLE ? blackhole::controller_layout : ControllerLayout {0, 0, 0, 0, 0, 0};
}

constexpr ControllerTable controller_table(Arch arch)
{
    if (arch == Arch::BLACKHOLE)
    {
        return ControllerTable {blackhole::controller_counters.data(), blackhole::controller_counters.size()};
    }
    return ControllerTable {nullptr, 0};
}

// Address of NIU status register reg (noc_counters_types.h) of a tile with the given noc0 y: it moves with the NoC
// instance and the register instance of the tile.
constexpr std::uint64_t niu_address(Arch arch, std::uint32_t noc, std::uint32_t tile_noc0_y, std::uint32_t reg)
{
    const Layout l = layout_for(arch);
    const std::uint32_t instance = (tile_noc0_y + l.tile_instance_bias) % l.tile_instance_count;
    return l.niu_base + std::uint64_t(noc) * l.noc_instance_stride + std::uint64_t(instance) * l.tile_instance_stride + NIU_STATUS_OFFSET + 4 * reg;
}

} // namespace llk::perf::dram
