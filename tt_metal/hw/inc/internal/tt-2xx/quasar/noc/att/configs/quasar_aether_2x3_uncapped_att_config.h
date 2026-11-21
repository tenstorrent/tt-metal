// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/tt-2xx/quasar/noc/att/att.h"
#include "internal/tt-2xx/quasar/noc/att/att_address.h"
#include "internal/tt-2xx/quasar/noc/att/configs/quasar_aether_2x3_att_config.h"

/**
 * @file
 * @brief quasar_aether_2x3 with a DRAM window that addresses the full 1 GiB the
 * soc descriptor declares per channel. Selected by TT_METAL_SIMULATOR_DRAM_UNCAPPED=1
 * (or TT_METAL_NOC_ATT=quasar_aether_2x3_uncapped) for runs that need the
 * whole address space. The emulator's DRAM tile backs 64 MiB, so offsets at or
 * above it wrap onto the low 64 MiB.
 *
 * The local, worker and full-tile windows and every table but the DRAM ones are
 * the base map's. DRAM tiles resolve through the DRAM window, by bank or by host
 * coordinate; the base map's full-tile rows for (0,0) and (1,0) stay programmed in
 * the hardware table but nothing resolves to them.
 *
 * Data only. Address resolution over this data lives in noc/att/att_address.h.
 */
namespace quasar_aether_2x3_uncapped_att_config {

namespace base = quasar_aether_2x3_att_config;

using base::LOCAL_WINDOW;
using base::LOCAL_WINDOW_BASE;
using base::REMOTE_WINDOW;

// One selector bit at bit 30 picks the bank; the 30 bits below it carry the
// 1 GiB offset. translate=1 with BAR 0 delivers the bare offset to the DRAM
// tile, the same address XY addressing delivers.
constexpr noc_att::Window DRAM_WINDOW{
    .compare = 0x1400000000ull,
    .mask_bits = 31,
    .endpoint_shift = 30,
    .endpoint_size = 1,
    .endpoint_table_offset = 7,
    .translate_address = true,
};  // mask-table slot 2, BAR/rebase 0

static_assert(DRAM_WINDOW.local_address_limit() == (std::uint64_t{1} << 30));

// Rows 7 and 8 sit past the local row and the six remote rows, so a DRAM operand never
// shares an endpoint row with a worker and is_self_address cannot take it for self.
static_assert(
    DRAM_WINDOW.endpoint_table_offset >=
    REMOTE_WINDOW.endpoint_table_offset + sizeof(base::ATT_FULL_TILE_ENDPOINT_WORDS) / sizeof(std::uint16_t));

// Bank 0 is tile (0,0) at selector 0, bank 1 is tile (1,0) at selector 1.
constexpr std::uint16_t ATT_DRAM_ENDPOINT_WORDS[] = {0x00, 0x01};
constexpr std::uint8_t ATT_LOGICAL_DRAM_SELECTORS[] = {0, 1};

// The base map with the Dram role moved to its own window.
inline constexpr noc_att::MapData MAP{
    .windows = {{noc_att::NO_WINDOW, REMOTE_WINDOW, DRAM_WINDOW, REMOTE_WINDOW, LOCAL_WINDOW}},
    .local_window_class = noc_att::WindowClass::Local,
    .worker_origin_x = base::ATT_WORKER_API_ORIGIN_X,
    .worker_origin_y = base::ATT_WORKER_API_ORIGIN_Y,
    .worker_grid_x = base::ATT_WORKER_GRID_X,
    .worker_grid_y = base::ATT_WORKER_GRID_Y,
    .worker_selectors = {base::ATT_WORKER_SELECTORS},
    .node_id_offset_x = 0,
    .node_id_offset_y = 0,
    .worker_endpoint_words = {base::ATT_WORKER_ENDPOINT_WORDS},
    .full_tile_endpoint_words = {base::ATT_FULL_TILE_ENDPOINT_WORDS},
    .dram_endpoint_words = {ATT_DRAM_ENDPOINT_WORDS},
    .dram_selectors = {ATT_LOGICAL_DRAM_SELECTORS},
    .dispatch_entries = {base::DISPATCH_ENTRIES},
};

}  // namespace quasar_aether_2x3_uncapped_att_config
