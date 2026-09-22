// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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

#include <cstdint>
#include <string_view>

#include "internal/tt-2xx/quasar/noc/att/att_address.h"
#include "internal/tt-2xx/quasar/noc/att/configs/grendel_qsr1_att_config.h"
#include "internal/tt-2xx/quasar/noc/att/configs/quasar_aether_2x3_att_config.h"

namespace tt::tt_metal::quasar_att {

struct MapInfo {
    std::string_view name;           // TT_METAL_NOC_ATT value
    std::string_view config_define;  // NOC_ATT_CONFIG_* selecting the map in att_config.h
    const noc_att::MapData& map;
};

inline constexpr MapInfo MAPS[] = {
    {"grendel_qsr1", "NOC_ATT_CONFIG_GRENDEL_QSR1", grendel_qsr1_att_config::MAP},
    {"quasar_aether_2x3", "NOC_ATT_CONFIG_QUASAR_AETHER_2X3", quasar_aether_2x3_att_config::MAP},
};

/// Names accepted by TT_METAL_NOC_ATT, for diagnostics (keep in step with MAPS).
inline constexpr std::string_view KNOWN_MAP_NAMES = "grendel_qsr1 or quasar_aether_2x3";

/// @brief The registry row for @p name, or nullptr for an unknown map.
constexpr const MapInfo* find_map(std::string_view name) {
    for (const MapInfo& info : MAPS) {
        if (info.name == name) {
            return &info;
        }
    }
    return nullptr;
}

/// @brief How many bytes of DRAM a bank can address under this map: the size
/// of the DRAM window's offset field. The allocator caps the DRAM bank size at
/// this so top-down allocations (kernel binaries) stay reachable.
///
/// - quasar_aether_2x3: 64 MiB, smaller than the DRAM the descriptor
///   advertises, so the cap matters there.
/// - grendel_qsr1: 8 GiB, larger than the descriptor's DRAM view, so the cap
///   changes nothing there.
constexpr std::uint64_t dram_window_local_address_limit(const MapInfo& info) {
    return noc_att::map_window(info.map, noc_att::WindowClass::Dram).local_address_limit();
}

static_assert(
    dram_window_local_address_limit(*find_map("quasar_aether_2x3")) == (std::uint64_t{64} << 20),
    "quasar_aether_2x3 DRAM window expected to carry 64 MiB of local address");
static_assert(
    dram_window_local_address_limit(*find_map("grendel_qsr1")) == (std::uint64_t{8} << 30),
    "grendel_qsr1 DRAM window expected to carry 8 GiB of local address");

}  // namespace tt::tt_metal::quasar_att
