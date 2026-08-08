// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/debug/assert.h"
#include "internal/risc_attribs.h"
#include "noc/att/att.h"

// Runtime selects one data configuration at compile time. All address
// construction below is shared; the NOC transport consumes the result as an
// opaque 64-bit address.
#if defined(NOC_ATT_CONFIG_QUASAR_AETHER_2X3) && defined(NOC_ATT_CONFIG_GRENDEL_QSR1)
#error "Select exactly one NOC ATT configuration"
#elif defined(NOC_ATT_CONFIG_QUASAR_AETHER_2X3)
#include "noc/att/configs/quasar_aether_2x3_att_config.h"
namespace selected_att_config = quasar_aether_2x3_att_config;
#elif defined(NOC_ATT_CONFIG_GRENDEL_QSR1)
#include "noc/att/configs/grendel_qsr1_att_config.h"
namespace selected_att_config = grendel_qsr1_att_config;
#else
#error "NOC_ATT_ENABLED requires an explicit NOC ATT configuration"
#endif

namespace active_att {

constexpr uint64_t MAP_SIGNATURE = selected_att_config::MAP_SIGNATURE;
constexpr noc_att::Window LOCAL_WINDOW = selected_att_config::LOCAL_WINDOW;
constexpr noc_att::Window WORKER_WINDOW = selected_att_config::WORKER_WINDOW;
constexpr noc_att::Window DRAM_WINDOW = selected_att_config::DRAM_WINDOW;
constexpr noc_att::Window TILE_WINDOW = selected_att_config::TILE_WINDOW;

constexpr uint32_t ATT_MAX_WORKERS =
    sizeof(selected_att_config::ATT_WORKER_SELECTORS) / sizeof(selected_att_config::ATT_WORKER_SELECTORS[0]);
constexpr uint32_t ATT_MAX_DRAM_BANKS =
    sizeof(selected_att_config::ATT_DRAM_SELECTORS) / sizeof(selected_att_config::ATT_DRAM_SELECTORS[0]);
constexpr uint32_t ATT_MAX_TILES =
    sizeof(selected_att_config::ATT_TILE_SELECTORS) / sizeof(selected_att_config::ATT_TILE_SELECTORS[0]);

constexpr bool local_address_supported(uint64_t local_address, uint64_t size = 1) {
    const uint64_t limit = noc_att::local_address_limit(LOCAL_WINDOW);
    return size > 0 && local_address < limit && size <= limit - local_address;
}

FORCE_INLINE uint64_t embed_local_address(uint64_t local_l1_address) {
    ASSERT(local_address_supported(local_l1_address));
    return noc_att::make_local_address(LOCAL_WINDOW, local_l1_address);
}

FORCE_INLINE uint64_t loopback_scratch_address(uint32_t offset = 0) {
    const uint64_t local_l1_address = selected_att_config::ATT_LOCAL_SCRATCH_OFFSET + offset;
    ASSERT(local_address_supported(local_l1_address, sizeof(uint32_t)));
    return embed_local_address(local_l1_address);
}

constexpr uint32_t worker_index(uint32_t logical_x, uint32_t logical_y) {
    return logical_y * selected_att_config::ATT_WORKER_GRID_X + logical_x;
}

FORCE_INLINE uint32_t worker_selector(uint32_t logical_x, uint32_t logical_y) {
    ASSERT(logical_x < selected_att_config::ATT_WORKER_GRID_X && logical_y < selected_att_config::ATT_WORKER_GRID_Y);
    return selected_att_config::ATT_WORKER_SELECTORS[worker_index(logical_x, logical_y)];
}

FORCE_INLINE uint32_t worker_logical_x_from_api(uint32_t api_x) {
    ASSERT(api_x >= selected_att_config::ATT_WORKER_API_ORIGIN_X);
    const uint32_t logical_x = api_x - selected_att_config::ATT_WORKER_API_ORIGIN_X;
    ASSERT(logical_x < selected_att_config::ATT_WORKER_GRID_X);
    return logical_x;
}

FORCE_INLINE uint32_t worker_logical_y_from_api(uint32_t api_y) {
    ASSERT(api_y >= selected_att_config::ATT_WORKER_API_ORIGIN_Y);
    const uint32_t logical_y = api_y - selected_att_config::ATT_WORKER_API_ORIGIN_Y;
    ASSERT(logical_y < selected_att_config::ATT_WORKER_GRID_Y);
    return logical_y;
}

FORCE_INLINE uint64_t worker_address_logical_xy(uint32_t logical_x, uint32_t logical_y, uint64_t offset) {
    ASSERT(offset < noc_att::local_address_limit(WORKER_WINDOW));
    return noc_att::make_address(WORKER_WINDOW, worker_selector(logical_x, logical_y), offset);
}

FORCE_INLINE uint64_t worker_address_api_xy(uint32_t api_x, uint32_t api_y, uint64_t offset) {
    return worker_address_logical_xy(worker_logical_x_from_api(api_x), worker_logical_y_from_api(api_y), offset);
}

FORCE_INLINE uint64_t dram_address(uint32_t bank, uint64_t offset) {
    ASSERT(bank < ATT_MAX_DRAM_BANKS);
    ASSERT(offset < noc_att::local_address_limit(DRAM_WINDOW));
    return noc_att::make_address(DRAM_WINDOW, selected_att_config::ATT_DRAM_SELECTORS[bank], offset);
}

FORCE_INLINE uint32_t tile_physical_index(uint32_t physical_x, uint32_t physical_y) {
    ASSERT(physical_x < selected_att_config::ATT_TILE_GRID_X && physical_y < selected_att_config::ATT_TILE_GRID_Y);
    return physical_y * selected_att_config::ATT_TILE_GRID_X + physical_x;
}

FORCE_INLINE uint32_t tile_selector_from_physical_xy(uint32_t physical_x, uint32_t physical_y) {
    return selected_att_config::ATT_TILE_SELECTORS[tile_physical_index(physical_x, physical_y)];
}

FORCE_INLINE uint64_t tile_address_physical_xy(uint32_t physical_x, uint32_t physical_y, uint64_t offset) {
    ASSERT(offset < noc_att::local_address_limit(TILE_WINDOW));
    return noc_att::make_address(TILE_WINDOW, tile_selector_from_physical_xy(physical_x, physical_y), offset);
}

FORCE_INLINE uint64_t replace_local_address(uint64_t global_address, uint64_t new_local_address) {
    for (const auto& window : selected_att_config::WINDOWS) {
        if (!noc_att::matches(window, global_address)) {
            continue;
        }
        if (window.compare == LOCAL_WINDOW.compare) {
            ASSERT(local_address_supported(new_local_address));
            return embed_local_address(new_local_address);
        }
        ASSERT(new_local_address < noc_att::local_address_limit(window));
        return noc_att::replace_local_address(window, global_address, new_local_address);
    }
    ASSERT(false);
    return global_address;
}

static_assert(noc_att::valid(LOCAL_WINDOW));
static_assert(noc_att::valid(WORKER_WINDOW));
static_assert(noc_att::valid(DRAM_WINDOW));
static_assert(noc_att::valid(TILE_WINDOW));
static_assert(ATT_MAX_WORKERS == selected_att_config::ATT_WORKER_GRID_X * selected_att_config::ATT_WORKER_GRID_Y);
static_assert(ATT_MAX_TILES == selected_att_config::ATT_TILE_GRID_X * selected_att_config::ATT_TILE_GRID_Y);

}  // namespace active_att
