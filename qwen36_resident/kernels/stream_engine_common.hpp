// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Weight-streaming engine (resident decode prototype): shared compile-time schedule.
//
// One resident program walks `iters` x `layers` x kEntries M=1 matmuls (the per-layer projection
// pattern, e.g. qkvzab / out / gate_up / down). Each streaming core owns a column slice of every
// weight inside its DRAM bank's shard (K-contiguous sticks) and computes that slice of the output.
//
// Compile-time args (identical for all three RISCs):
//   0 layers, 1 iters, 2 weight_sets (distinct weight copies cycled over layers)
//   then per entry e (kEntries = 4), 5 values each at 3 + 5e:
//     Kt, sb (K tiles per streamed block), pages_per_block, page_size (bytes), block_bytes
//   then 23 ring_bytes: size of the weight ring shared by all entries
//
// Weight ring: CBs cb_w0 .. cb_w0 + kEntries - 1 alias one L1 allocation. Blocks are placed in schedule
// order (a block that does not fit before the end starts over at the ring base), so the reader and
// compute derive every block's position independently. Free space is tracked with a 16-bit count of
// consumed 64-byte units that compute publishes to cb_consumed's tiles_acked stream register after the
// unpacker is done (the same store cb_pop_front uses).
#pragma once

#include <stdint.h>

namespace stream_engine {

constexpr uint32_t kEntries = 4;
constexpr uint32_t cb_in0 = 0;
constexpr uint32_t cb_w0 = 1;  // weights of entry e live in CB cb_w0 + e
constexpr uint32_t cb_out = 5;
constexpr uint32_t cb_gate = 6;
constexpr uint32_t cb_consumed = 7;
constexpr uint32_t kUnit = 64;  // ring accounting granularity (bytes); all blocks are multiples of it

constexpr uint32_t layers = get_compile_time_arg_val(0);
constexpr uint32_t iters = get_compile_time_arg_val(1);
constexpr uint32_t weight_sets = get_compile_time_arg_val(2);
constexpr uint32_t ring_bytes = get_compile_time_arg_val(23);

template <uint32_t E>
struct Entry {
    static constexpr uint32_t Kt = get_compile_time_arg_val(3 + 5 * E + 0);
    static constexpr uint32_t sb = get_compile_time_arg_val(3 + 5 * E + 1);
    static constexpr uint32_t pages = get_compile_time_arg_val(3 + 5 * E + 2);
    static constexpr uint32_t page_size = get_compile_time_arg_val(3 + 5 * E + 3);
    static constexpr uint32_t block_bytes = get_compile_time_arg_val(3 + 5 * E + 4);
    static constexpr uint32_t nkb = Kt / sb;
    static constexpr uint32_t cb = cb_w0 + E;
};

// Deterministic block placement in the weight ring.
struct RingCursor {
    uint32_t pos = 0;    // byte offset of the next block inside the ring
    uint32_t units = 0;  // 64-byte units accounted so far (including skipped ring tails), mod 2^16

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

}  // namespace stream_engine
