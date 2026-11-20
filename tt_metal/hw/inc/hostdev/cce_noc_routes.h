// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

// Keraunos CCE NOC routes. A Keraunos CCE (a Quasar Ethernet core) has no NOC endpoint, so a NOC address it uses
// resolves through this table, which the host writes at MEM_IERISC_NOC_ROUTES_BASE of every such core before
// launch. Coordinates are this chip's: each Keraunos core of the chip's SoC descriptor at its virtual coordinate and at
// its Ethernet channel's (eth_chan_to_noc_xy). sram_base is the address of the core's SRAM offset 0 for an iDMA copy:
// the Keraunos local address on this CCE, the system address on another HSIO tile, 0 for this tile's other CCE, which
// a CCE does not address. The device writes fault and fault_detail when it stops on a NOC call it cannot route.
inline constexpr uint32_t CCE_NOC_ROUTES_MAX = 16;
inline constexpr uint32_t CCE_NOC_ROUTES_VALID = 0x54554F52;  // "ROUT"

enum class CceNocRouteKind : uint8_t {
    THIS_CCE = 1,
    THIS_TILE_OTHER_CCE = 2,
    OTHER_TILE = 3,
};

struct CceNocRoute {
    uint64_t sram_base;
    uint8_t x;
    uint8_t y;
    CceNocRouteKind kind;
    uint8_t reserved[5];
};

struct CceNocRoutes {
    uint32_t valid;  // CCE_NOC_ROUTES_VALID once written
    uint32_t num_routes;
    uint8_t self_x;  // this core's virtual coordinate
    uint8_t self_y;
    uint8_t reserved[2];
    uint32_t fault;
    uint64_t fault_detail;
    CceNocRoute routes[CCE_NOC_ROUTES_MAX];
};

static_assert(sizeof(CceNocRoute) == 16);
static_assert(sizeof(CceNocRoutes) == 280);
static_assert(offsetof(CceNocRoutes, routes) == 24);

// System address of HSIO tile 0's window (Keraunos-E100 Table 14); tiles are keraunos.tile_stride apart.
inline constexpr uint64_t KERAUNOS_HSIO0_SYSTEM_BASE = 0x1204000000ULL;

// System address of the POSTED register of LLK tile counter `counter` (0-15) of CCE `cce` on HSIO tile `hsio`, tiles
// `tile_stride` bytes apart. In a tile's window CCE0's counter 0 is at 0x3000 and CCE1's at 0x100_3000, counters 0x40
// apart, POSTED at +0x8 (chippy registers/ker/smc_view/hsio_tile.h and registers/ker/cce/cce_reg.h).
constexpr uint64_t cce_tile_counter_posted_system_addr(
    uint32_t hsio, uint64_t tile_stride, uint32_t cce, uint32_t counter) {
    return KERAUNOS_HSIO0_SYSTEM_BASE + hsio * tile_stride + cce * 0x1000000ULL + 0x3000 + counter * 0x40 + 0x8;
}
