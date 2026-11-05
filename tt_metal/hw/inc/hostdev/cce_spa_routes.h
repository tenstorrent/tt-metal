// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

// Per-CCE SPA routes. The host writes one after the bank-to-NOC staging table during DRAM firmware
// load. core_info_msg_t cannot carry it: that struct is shared with Blackhole, whose DRAM mailbox
// is sized to the struct exactly.
//
// GDDR: gddr_d2d_base[0] is the D2D0 window and gddr_d2d_base[1] is the D2D1 window, 16 instance
// strides above D2D0. gddr_d2d[instance] is the port this core uses to reach that SPA instance.
// A local instance uses port 0, which is the window the CCE remapper claims. SRAM is one package
// window, 4 MiB per CCE, indexed by CCE.
inline constexpr uint32_t CCE_SPA_ROUTES_MAX_INSTANCES = 16;
inline constexpr uint64_t CCE_GDDR_D2D_LANE_INSTANCES = 16;

struct CceSpaRoutes {
    uint64_t gddr_d2d_base[2];
    uint64_t gddr_instance_stride;
    uint64_t sram_spa_base;
    uint64_t sram_tile_stride;
    uint8_t gddr_d2d[CCE_SPA_ROUTES_MAX_INSTANCES];
};

static_assert(sizeof(CceSpaRoutes) == 56);
static_assert(offsetof(CceSpaRoutes, gddr_d2d) == 40);

// Keraunos CCE NOC routes. A Keraunos CCE (a Quasar Ethernet core) has no NOC endpoint, so a NOC address it uses
// resolves through this table, which the host writes after the bank-to-NOC staging table of every such core before
// launch. Coordinates are this chip's: each Keraunos core of the chip's SoC descriptor at its virtual coordinate and at
// its Ethernet channel's (eth_chan_to_noc_xy). sram_base is the address of the core's SRAM offset 0 for an iDMA copy:
// the Keraunos local address on this CCE, the system address on another HSIO tile, 0 for this tile's other CCE, which
// a CCE does not address. The device writes fault and fault_detail when it stops on a NOC call it cannot route.
inline constexpr uint32_t CCE_NOC_ROUTES_MAX = 16;
inline constexpr uint32_t CCE_NOC_ROUTES_VALID = 0x54554F52;  // "ROUT"

enum class CceNocRouteKind : uint8_t {
    UNMAPPED = 0,
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
