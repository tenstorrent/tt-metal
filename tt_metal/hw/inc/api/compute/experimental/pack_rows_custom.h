// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/common.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/pack.h"

namespace ckernel {

#ifdef ARCH_BLACKHOLE
constexpr uint32_t MIN_BLOCKED_PACK_TILES = 2;
#else
constexpr uint32_t MIN_BLOCKED_PACK_TILES = 8;
#endif
ALWI bool should_use_blocked_pack_width(uint32_t pack_width) { return pack_width >= MIN_BLOCKED_PACK_TILES; }

#ifdef TRISC_PACK
inline __attribute__((always_inline)) static void sdpa_pack_tile_ooo(uint32_t dst, uint32_t cb, uint32_t idx) {
    llk_pack<DST_ACCUM_MODE, true, PackMode::Default>(dst, cb, idx);
}
#else
ALWI void sdpa_pack_tile_ooo(uint32_t, uint32_t, uint32_t) {}
#endif

ALWI void pack_contiguous_rows(
    uint32_t out_cb,
    uint32_t row_base,
    uint32_t row_count,
    uint32_t row_stride,
    uint32_t col_base,
    uint32_t pack_width,
    bool accumulate_l1 = false,
    ReluConfig relu = ReluConfig::none(),
    uint32_t dst_base = 0) {
    pack_reconfig_data_format(out_cb);
    const bool use_blocked_pack_width = should_use_blocked_pack_width(pack_width);
    PACK((llk_pack_init<PackMode::Default, false, true, true>(out_cb, use_blocked_pack_width ? pack_width : 1)));
    PACK((llk_pack_reconfig_l1_acc(accumulate_l1 ? 1 : 0)));
    PACK((llk_pack_relu_config(relu)));
    uint32_t dst_index = dst_base;
    for (uint32_t row = 0; row < row_count; ++row) {
        uint32_t out_tile_index = (row_base + row) * row_stride + col_base;
        if (use_blocked_pack_width) {
            sdpa_pack_tile_ooo(dst_index, out_cb, out_tile_index);
            dst_index += pack_width;
        } else {
            for (uint32_t col = 0; col < pack_width; ++col) {
                sdpa_pack_tile_ooo(dst_index++, out_cb, out_tile_index + col);
            }
        }
    }
    if (accumulate_l1) {
        PACK((llk_pack_reconfig_l1_acc(0)));
    }
}

}  // namespace ckernel
