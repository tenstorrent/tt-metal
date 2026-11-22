// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Resident DFlash2 drafter, ingest mode (M6b C1; M6A_DESIGN_ADDENDUM_1 §2, §4.5): the committed rows' target taps ->
// the drafter's context K / V ring rows, every drafter layer, one launch of up to 32 rows. Streamer schedule:
//   in   the slices' tiles of the taps (chip c's 1344 columns of each of the 5 taps: Kin = 210 tiles, input-sharded)
//   fc   partial = in @ W_fc (this core's hidden columns) -> its slice and the hub (all-reduce round 0)
//   ctx  hidden_norm(fc) from the slices; per drafter layer k | v = ctx @ W_kv (this core's 2 column tiles of one
//        head block: bank b holds k head b (b < 4) or v head b - 4) -> the head block's slice (slice b)
// Weight ring, block placement and the per-bank streaming are streamer_common.hpp's (the target's), unchanged.
#pragma once

#include <stdint.h>

namespace ding {

constexpr uint32_t kEntries = 2;
constexpr uint32_t kFc = 0, kKv = 1;

// streamer CBs
constexpr uint32_t cb_w0 = 0;        // weight ring, entry e -> cb_w0 + e
constexpr uint32_t cb_consumed = 7;  // tiles_acked carries compute's consumed ring units
constexpr uint32_t cb_in = 8;        // the taps' tiles (Kin), then (same buffer) ctx (Ht)
constexpr uint32_t cb_h = 9;
constexpr uint32_t cb_pout = 11;     // the fc partial (nd_max tiles)
constexpr uint32_t cb_kv = 12;       // this core's k | v tiles of one layer (kKvTiles)

constexpr uint32_t kUnit = 64;
constexpr uint32_t kKvTiles = 2;      // k | v column tiles per streamer (64 per chip over 32 streamers)
constexpr uint32_t kHeadTiles = 8;    // one head block (256 columns)

// Compile-time args (the three RISCs alike): 0 layers, 1 num_streamers, 2 Ht, 3 Kin, 4 chip, 5 ring_bytes, 6 nd_max,
// 7 K (slices), 8 sem_in (this core's: the slices' tap and ctx multicasts), 9 sem_kv (the slices'), 10 M (rows),
// then per entry e at 11 + 5 e: Kt, sb, pages, page_size, block_bytes
constexpr uint32_t layers = get_compile_time_arg_val(0);
constexpr uint32_t num_streamers = get_compile_time_arg_val(1);
constexpr uint32_t Ht = get_compile_time_arg_val(2);
constexpr uint32_t Kin = get_compile_time_arg_val(3);
constexpr uint32_t chip = get_compile_time_arg_val(4);
constexpr uint32_t ring_bytes = get_compile_time_arg_val(5);
constexpr uint32_t nd_max = get_compile_time_arg_val(6);
constexpr uint32_t K = get_compile_time_arg_val(7);
constexpr uint32_t sem_in = get_compile_time_arg_val(8);
constexpr uint32_t sem_kv = get_compile_time_arg_val(9);
constexpr uint32_t M = get_compile_time_arg_val(10);
constexpr uint32_t kTileBytes = 2048;  // 32 x 32 bf16

template <uint32_t E>
struct Entry {
    static constexpr uint32_t Kt = get_compile_time_arg_val(11 + 5 * E + 0);
    static constexpr uint32_t sb = get_compile_time_arg_val(11 + 5 * E + 1);
    static constexpr uint32_t pages = get_compile_time_arg_val(11 + 5 * E + 2);
    static constexpr uint32_t page_size = get_compile_time_arg_val(11 + 5 * E + 3);
    static constexpr uint32_t block_bytes = get_compile_time_arg_val(11 + 5 * E + 4);
    static constexpr uint32_t nkb = Kt / sb;
    static constexpr uint32_t cb = cb_w0 + E;
};

// streamer_common.hpp's ring cursor (the reader and compute agree on every block)
struct RingCursor {
    uint32_t pos = 0;
    uint32_t units = 0;

    template <uint32_t E>
    uint32_t place() {
        constexpr uint32_t block = Entry<E>::block_bytes;
        if (pos + block > ring_bytes) {
            units += (ring_bytes - pos) / kUnit;
            pos = 0;
        }
        const uint32_t at = pos;
        pos += block;
        units += block / kUnit;
        return at;
    }
};

}  // namespace ding
