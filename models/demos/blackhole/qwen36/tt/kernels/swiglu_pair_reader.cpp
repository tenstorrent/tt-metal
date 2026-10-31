// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MLP_GU_FUSED split-SwiGLU (tt/mlp_swiglu.py), reader: the gate and up tiles of this core's output tiles.
//
// Input : the fused [gate|up] decode matmul output, [.., Mt*32, 2*NP*32] bf16 TILE, interleaved or sharded (any layout a
//         TensorAccessor resolves; the page id is the row-major tile index).
//         PAIR = 1: tile-pair interleaved per tile row, [g0 u0 g1 u1 ...] (the prefill AGMM weight's column order).
//         PAIR = 0: halves per tile row, [g0 .. g(NP-1) | u0 .. u(NP-1)].
// Output tile o (row-major over Mt x NP, o = row * NP + p) needs gate tile g(row, p) and up tile u(row, p); this core
// owns output tiles [start, start + count) (runtime args) and pushes them BATCH pairs at a time, gate first, into CB_IN.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
    constexpr uint32_t PAIR = get_compile_time_arg_val(1);
    constexpr uint32_t NP = get_compile_time_arg_val(2);  // output tiles (gate/up pairs) per tile row
    constexpr uint32_t BATCH = get_compile_time_arg_val(3);  // pairs per CB push (the CB holds 2 * BATCH tiles)
    constexpr auto in_args = TensorAccessorArgs<4>();
    static_assert(BATCH >= 1, "BATCH >= 1");

    const uint32_t in_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    const auto in = TensorAccessor(in_args, in_addr);
    const uint32_t tile_bytes = in.get_aligned_page_size();

    CircularBuffer cb(CB_IN);
    uint32_t done = 0;
    while (done < count) {
        const uint32_t n = (count - done) < BATCH ? (count - done) : BATCH;
        cb.reserve_back(2 * n);
        uint32_t l1 = get_write_ptr(CB_IN);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t o = start + done + i;
            const uint32_t row = o / NP;
            const uint32_t p = o - row * NP;
            const uint32_t base = row * 2 * NP;
            const uint32_t g = PAIR ? base + 2 * p : base + p;
            const uint32_t u = PAIR ? g + 1 : g + NP;
            noc_async_read(in.get_noc_addr(g), l1, tile_bytes);
            noc_async_read(in.get_noc_addr(u), l1 + tile_bytes, tile_bytes);
            l1 += 2 * tile_bytes;
        }
        noc_async_read_barrier();
        cb.push_back(2 * n);
        done += n;
    }
}
