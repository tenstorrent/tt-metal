// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dataflow-kernel helpers shared by the forward and backward chunk_gated_delta_rule readers.  Kernel
// only (it needs the dataflow API), unlike chunk_gated_delta_rule_common.hpp, which the host includes too.

#pragma once

#include <cstdint>

#include "tt-train/sources/ttml/metal/common/dataflow_utils.hpp"

namespace chunk_gated_delta_rule_common {

constexpr uint32_t FP32_NEG_ONE_BITS = 0xBF800000;  // -1.0f

// A square [nt*32, nt*32] fp32 constant, written as nt * nt tiles in row-major tile order into `cb_id`
// starting at tile `tile_offset` of the reserved region.  Element (r, c) of the whole matrix is
// `one_bits` where `one_at(r, c)` and zero elsewhere.  The tiles are zero-filled over the NoC, then only
// the non-zero lanes are stored.  Waits for the zero fill, so the caller needs no barrier of its own.
template <typename Pred>
inline void build_cc(
    uint32_t cb_id, uint32_t tile_offset, uint32_t nt, Pred one_at, uint32_t one_bits = FP32_ONE_BITS) {
    const uint32_t tile_bytes = get_tile_size(cb_id);
    const uint32_t base = get_write_ptr(cb_id) + tile_offset * tile_bytes;
    fill_tile_zeros(base, nt * nt * tile_bytes);
    noc_async_read_barrier();
    for (uint32_t ti = 0; ti < nt; ++ti) {
        for (uint32_t si = 0; si < nt; ++si) {
            volatile tt_l1_ptr uint32_t* tile =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + (ti * nt + si) * tile_bytes);
            for (uint32_t r = 0; r < 32U; ++r) {
                for (uint32_t c = 0; c < 32U; ++c) {
                    if (one_at(ti * 32U + r, si * 32U + c)) {
                        tile[get_tilized_idx(r, c)] = one_bits;
                    }
                }
            }
        }
    }
}

}  // namespace chunk_gated_delta_rule_common
