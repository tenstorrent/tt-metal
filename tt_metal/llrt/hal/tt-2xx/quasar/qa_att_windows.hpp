// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file
 * @brief The host's list of the Quasar ATT maps: one row per map with its
 * TT_METAL_NOC_ATT name, the define that selects it in device builds, and the
 * map data itself. Host code looks maps up here instead of hard-coding names
 * or map facts. The device-side selector header (att_config.h) is not included:
 * it picks one map by macro, while the host needs all of them.
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "internal/tt-2xx/quasar/noc/att/att_address.h"
#include "internal/tt-2xx/quasar/noc/att/configs/grendel_qsr1_att_config.h"
#include "internal/tt-2xx/quasar/noc/att/configs/horizon_2x3_att_config.h"
#include "internal/tt-2xx/quasar/noc/att/configs/quasar_aether_2x3_att_config.h"
#include "internal/tt-2xx/quasar/noc/att/configs/quasar_aether_2x3_uncapped_att_config.h"

namespace tt::tt_metal::quasar_att {

struct MapInfo {
    std::string_view name;           // TT_METAL_NOC_ATT value
    std::string_view config_define;  // NOC_ATT_CONFIG_* selecting the map in att_config.h
    const noc_att::MapData& map;
};

inline constexpr MapInfo MAPS[] = {
    {"grendel_qsr1", "NOC_ATT_CONFIG_GRENDEL_QSR1", grendel_qsr1_att_config::MAP},
    {"quasar_aether_2x3", "NOC_ATT_CONFIG_QUASAR_AETHER_2X3", quasar_aether_2x3_att_config::MAP},
    {"quasar_aether_2x3_uncapped",
     "NOC_ATT_CONFIG_QUASAR_AETHER_2X3_UNCAPPED",
     quasar_aether_2x3_uncapped_att_config::MAP},
    {"horizon_2x3", "NOC_ATT_CONFIG_HORIZON_2X3", horizon_2x3_att_config::MAP},
};

/// Names accepted by TT_METAL_NOC_ATT, for diagnostics (keep in step with MAPS).
inline constexpr std::string_view KNOWN_MAP_NAMES =
    "grendel_qsr1, quasar_aether_2x3, quasar_aether_2x3_uncapped or horizon_2x3";

/// @brief The registry row for @p name, or nullptr for an unknown map.
constexpr const MapInfo* find_map(std::string_view name) {
    for (const MapInfo& info : MAPS) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

/// @brief The DRAM a bank may address under this map: the size of the DRAM
/// window's local-address field. The allocator takes the smaller of this and the
/// descriptor's DRAM size.
///
/// - quasar_aether_2x3: 64 MiB, the DRAM a 2x3 emulator channel backs
///   (qa_emulator_models.hpp), which the descriptor is already clamped to.
/// - quasar_aether_2x3_uncapped: 1 GiB, the DRAM the soc descriptor declares
///   (TT_METAL_SIMULATOR_DRAM_UNCAPPED=1).
/// - horizon_2x3: 64 MiB.
/// - grendel_qsr1: 8 GiB, larger than the descriptor's DRAM view.
constexpr std::uint64_t dram_window_local_address_limit(const MapInfo& info) {
    return noc_att::map_window(info.map, noc_att::WindowClass::Dram).local_address_limit();
}

/// @brief Whether the interleaved L1 banks form one ascending stride-1 selector run in @p map: bank b's selector is
/// bank 0's + b, all in the worker window. The Quasar address generator's interleaved recipe relies on it: it programs
/// bank 0's selector as the base and adds the bank number (addrgen_sequencer.h, plan_interleaved). The JIT checks it
/// per device; where it fails, interleaved L1 TensorAccessor transfers use software (TT_TA_ADDRGEN_INTERLEAVED_L1_SW).
///
/// It fails when the compute grid is not the whole ATT worker grid, e.g. a column taken for Tensix dispatch: L1 banks
/// skip that column, the worker window's row-major selectors do not.
///
/// @p worker_cores: the L1 banks' cores in bank order, in the kernel-visible frame (anything with .x and .y).
template <typename Cores>
constexpr bool worker_banks_form_selector_run(const noc_att::MapData& map, const Cores& worker_cores) {
    std::uint32_t bank = 0;
    std::uint32_t first = 0;
    for (const auto& core : worker_cores) {
        const noc_att::ResolvedTile tile =
            noc_att::resolve(map, noc_att::Address::worker(core.x, core.y, /*offset=*/0));
        if (!tile.valid || tile.window != noc_att::WindowClass::Worker) {
            return false;
        }
        if (bank == 0) {
            first = tile.selector;
        } else if (tile.selector != first + bank) {
            return false;
        }
        ++bank;
    }
    return true;
}

/// @brief As worker_banks_form_selector_run, for DRAM banks 0..@p num_dram_banks - 1. DRAM bank b is the map's logical
/// DRAM channel b whatever the compute grid, so this depends on the map alone: it is checked for every map below.
constexpr bool dram_banks_form_selector_run(const noc_att::MapData& map, std::uint32_t num_dram_banks) {
    std::uint32_t first = 0;
    for (std::uint32_t bank = 0; bank < num_dram_banks; ++bank) {
        const noc_att::ResolvedTile tile = noc_att::resolve(map, noc_att::Address::dram(bank, /*offset=*/0));
        if (!tile.valid) {
            return false;
        }
        if (bank == 0) {
            first = tile.selector;
        } else if (tile.selector != first + bank) {
            return false;
        }
    }
    return true;
}

namespace detail {
struct WorkerXy {
    std::uint32_t x;
    std::uint32_t y;
};
// L1 banks over the workers x in [x0, x1], y in [y0, y1], in the allocator's bank order (row-major).
template <std::size_t N>
constexpr std::array<WorkerXy, N> row_major_workers(std::uint32_t x0, std::uint32_t x1, std::uint32_t y0) {
    std::array<WorkerXy, N> cores{};
    const std::uint32_t width = x1 - x0 + 1;
    for (std::uint32_t i = 0; i < N; ++i) {
        cores[i] = {x0 + (i % width), y0 + (i / width)};
    }
    return cores;
}
}  // namespace detail

// grendel_qsr1 numbers the 8x4 workers (x 2..9, y 2..5) row-major. All 32 as compute cores: a run. With x 9 taken for
// Tensix dispatch (quasar_simulation_8x4_arch_tensix_dispatch.yaml), bank 7 is (2, 3), selector 8: not a run.
static_assert(worker_banks_form_selector_run(
    find_map("grendel_qsr1")->map, detail::row_major_workers<32>(/*x0=*/2, /*x1=*/9, /*y0=*/2)));
static_assert(!worker_banks_form_selector_run(
    find_map("grendel_qsr1")->map, detail::row_major_workers<28>(/*x0=*/2, /*x1=*/8, /*y0=*/2)));

constexpr bool every_map_dram_banks_form_selector_run() {
    for (const MapInfo& info : MAPS) {
        if (!dram_banks_form_selector_run(info.map, static_cast<std::uint32_t>(info.map.dram_selectors.size()))) {
            return false;
        }
    }
    return true;
}
static_assert(
    every_map_dram_banks_form_selector_run(),
    "the address generator's interleaved DRAM recipe needs each map's DRAM selectors in one ascending run");

static_assert(
    dram_window_local_address_limit(*find_map("quasar_aether_2x3")) == (std::uint64_t{64} << 20),
    "quasar_aether_2x3 DRAM window must carry 64 MiB of local address, the size of the emulator's DRAM tile");
static_assert(
    dram_window_local_address_limit(*find_map("quasar_aether_2x3_uncapped")) == (std::uint64_t{1} << 30),
    "quasar_aether_2x3_uncapped DRAM window must carry the 1 GiB of local address the soc descriptor declares");
static_assert(
    dram_window_local_address_limit(*find_map("horizon_2x3")) == (std::uint64_t{64} << 20),
    "horizon_2x3 DRAM window must carry 64 MiB of local address, the size of the emulator's DRAM tile");
static_assert(
    dram_window_local_address_limit(*find_map("grendel_qsr1")) == (std::uint64_t{8} << 30),
    "grendel_qsr1 DRAM window expected to carry 8 GiB of local address");

}  // namespace tt::tt_metal::quasar_att
