// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

// System address of HSIO tile 0's window; tiles are keraunos.tile_stride apart.
inline constexpr uint64_t KERAUNOS_HSIO0_SYSTEM_BASE = 0x1204000000ULL;

// System address of the POSTED register of LLK tile counter `counter` (0-15) of CCE `cce` on HSIO tile `hsio`, tiles
// `tile_stride` bytes apart. In a tile's window CCE0's counter 0 is at 0x3000 and CCE1's at 0x100_3000, counters 0x40
// apart, POSTED at +0x8.
constexpr uint64_t cce_tile_counter_posted_system_addr(
    uint32_t hsio, uint64_t tile_stride, uint32_t cce, uint32_t counter) {
    return KERAUNOS_HSIO0_SYSTEM_BASE + hsio * tile_stride + cce * 0x1000000ULL + 0x3000 + counter * 0x40 + 0x8;
}
