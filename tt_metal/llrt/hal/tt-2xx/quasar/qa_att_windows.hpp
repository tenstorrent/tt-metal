// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file
 * @brief Host-side registry of the Quasar ATT maps tt-metal can build against.
 *
 * One row per map: the TT_METAL_NOC_ATT name, the NOC_ATT_CONFIG_* define the
 * JIT build selects it with, and the transcribed map data itself, so the host
 * can answer questions about a map's windows (today: how much DRAM local
 * address the map can express) without re-deriving anything from the device
 * headers. Only the constexpr configuration headers are included - never
 * att_config.h, which selects a map by preprocessor macro and is for device
 * builds. The registry reads MAP.windows only, so it is unaffected by fields
 * the maps grow for other consumers.
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

/// @brief One past the largest DRAM local address a map's DRAM window can
/// carry: the ceiling on any per-bank DRAM address device code can compose
/// under that map. The allocator clamps the DRAM bank size to it.
///
/// - quasar_aether_2x3: 64 MiB (a single 2^33 remote window with the endpoint
///   selector at bit 26), smaller than the descriptor's DRAM view, so the
///   clamp is what keeps top-down allocations addressable.
/// - grendel_qsr1: 8 GiB (selector at bit 33). Bit 32 of that local field is
///   the D2D link select onto the same GDDR, so the descriptor's 1 GiB view,
///   not the window, is the real bound there - min() leaves it unclamped.
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
