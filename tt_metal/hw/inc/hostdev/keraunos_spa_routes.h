// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

// SPA routes for one Keraunos ethernet core (one HSIO tile, both CCEs). The host writes it at
// MEM_IERISC_SPA_ROUTES_BASE during idle-ethernet firmware load, from the descriptor's keraunos and
// mimir_channels blocks, so device code holds no package constants.
//
// HSIO SRAM:      hsio_sram_base + hsio * hsio_tile_stride + eth_cce * eth_cce_sram_size
// Mimir CCE SRAM: mimir_cce_sram_base + global_mimir_cce * mimir_cce_sram_stride   (Keraunos D2D BAR1)
// Mimir GDDR:     mimir_gddr_base[mimir]                                     (through this package's D2D)
inline constexpr uint32_t KERAUNOS_SPA_ROUTES_MAX_MIMIRS = 4;

struct KeraunosSpaRoutes {
    uint64_t hsio_sram_base;
    uint64_t hsio_tile_stride;
    uint64_t eth_cce_sram_size;
    uint64_t mimir_cce_sram_base;
    uint64_t mimir_cce_sram_stride;
    uint64_t mimir_gddr_base[KERAUNOS_SPA_ROUTES_MAX_MIMIRS];
    uint32_t hsio;          // this core's HSIO tile
    uint32_t num_hsio;      // HSIO tiles with a programmed window
    uint32_t num_mimirs;    // valid mimir_gddr_base entries
    uint32_t reserved;
};

static_assert(sizeof(KeraunosSpaRoutes) == 88);
