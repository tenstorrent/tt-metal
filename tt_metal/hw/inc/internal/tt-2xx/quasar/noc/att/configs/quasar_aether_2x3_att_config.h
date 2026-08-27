// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "internal/tt-2xx/quasar/noc/att/att.h"

/**
 * @file
 * @brief Quasar's small Aether GRID_2x3/GRID_2x3_DISPATCH bring-up map, based
 * on firmware/datamover/perf_testing_lib/include/aether_utils.h with selectors
 * for the top-row NOC2AXI/dispatch tiles - and with the LOCAL window upgraded
 * to QSR1 slot-14 geometry (decision 2026-08-27). This map only exists because
 * our bring-up replay programs it, so it is ours to shape: giving the local
 * window the product geometry (nonzero compare, translate=1 with BAR 0,
 * selector field, patched self endpoint) means every emulator run exercises
 * write-source / read-return / atomic-return roles through a TRANSLATING
 * window - the exact role/window combination QSR1's local operands use, which
 * a pass-through local window can never produce. It also removes the
 * compare=0 window entirely, so a leaked RAW local address matches no window
 * and faults instead of being silently absorbed.
 *
 * Data only. Address resolution over this data lives in noc/att/att_address.h.
 */
namespace quasar_aether_2x3_att_config {

// tt-metal's checked-in Aether descriptor exposes workers at (0,1),(1,1).
constexpr std::uint32_t ATT_WORKER_API_ORIGIN_X = 0;
constexpr std::uint32_t ATT_WORKER_API_ORIGIN_Y = 1;
constexpr std::uint32_t ATT_WORKER_GRID_X = 2;
constexpr std::uint32_t ATT_WORKER_GRID_Y = 1;

constexpr std::uint32_t ATT_TILE_GRID_X = 2;
constexpr std::uint32_t ATT_TILE_GRID_Y = 3;

constexpr std::uint8_t ATT_WORKER_SELECTORS[] = {0, 1};

// Endpoint words ((y << 6) | x) by selector: the inverse map software needs to
// resolve a core's own selector from its NOC_NODE_ID (see the QSR1 config for
// the full rationale). On this map they are additionally the SOURCE the
// bring-up replay programs the hardware endpoint table from - there is no boot
// image on the emulator - and the image test checks that consistency.
constexpr std::uint16_t ATT_WORKER_ENDPOINT_WORDS[] = {0x40, 0x41};
constexpr std::uint16_t ATT_FULL_TILE_ENDPOINT_WORDS[] = {0x40, 0x41, 0x00, 0x01, 0x80, 0x81};

// Array index uses tt-metal/UMD-visible y*2+x. Selectors 0..3 match
// aether_utils.h. The UMD descriptor exposes dispatch at (1,2), while the
// 2x3_DISPATCH RTL places it at (0,2), so the top-row selector order provides
// that alias explicitly.
constexpr std::uint8_t ATT_TILE_SELECTORS[] = {
    2,
    3,  // y=0: DRAM
    0,
    1,  // y=1: Tensix
    5,
    4,  // y=2 UMD view: NOC2AXI, dispatch
};

// Matches Aether::configure_aether_dram(GRID_2x3): bank 0 targets (0,0)
// through selector 2 and bank 1 targets (1,0) through selector 3.
constexpr std::uint8_t ATT_LOGICAL_DRAM_SELECTORS[] = {2, 3};

// QSR1 slot-14 geometry scaled to this endpoint table: selector bits [31:26]
// of the base fold to 0, so base | local selects entry 0 - the per-initiator
// patched self endpoint. translate=1 with BAR 0 rebases the outgoing address
// to the bare local bits.
constexpr noc_att::Window LOCAL_WINDOW{
    .compare = 0x1800000000ull,
    .mask_bits = 33,
    .endpoint_shift = 26,
    .endpoint_size = 6,
    .endpoint_table_offset = 0,
    .translate_address = true,
};  // mask-table slot 0

// Narrowed from aether_utils' 10-bit selector / 2^36 span: six tiles need six
// selectors, and the narrower window keeps it disjoint from the local window
// at 0x18_0000_0000 (windows must not overlap - hardware would resolve the
// overlap by first-match priority, which the software model deliberately does
// not reproduce).
constexpr noc_att::Window REMOTE_WINDOW{
    .compare = 0x1000000000ull,
    .mask_bits = 32,
    .endpoint_shift = 26,
    .endpoint_size = 6,
    .endpoint_table_offset = 1,
    .translate_address = true,
};  // mask-table slot 1, BAR/rebase 0

// Every operand for this initiator's own L1 is LOCAL_WINDOW_BASE | local_address.
// The ATT enablement surfaces this value as the NOC_ATT_LOCAL_WINDOW_BASE macro
// that noc_nonblocking_api_v3.h requires, with a static_assert against this
// constant so the two cannot drift apart.
constexpr std::uint64_t LOCAL_WINDOW_BASE = LOCAL_WINDOW.make_address(/*selector*/ 0, /*local_address*/ 0);
static_assert(LOCAL_WINDOW_BASE == 0x1800000000ull);

}  // namespace quasar_aether_2x3_att_config
