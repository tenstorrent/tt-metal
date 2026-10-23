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
