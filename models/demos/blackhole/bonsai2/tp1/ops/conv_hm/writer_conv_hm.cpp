// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Writer for conv_hm (one core per half state tile, see the reader).
//   state : this half of every state row shifted in place, st[j] <- old st[j+1] (j < K-1), st[K-1] <- the new row.
//           Safe in place: this core is the only reader and writer of these 1 KB, and the reader finished its reads
//           before pushing cb_wst.
//   q/k/v : row i of this half of the fp32 conv tile is head h = h0 + i; it becomes row 0 of tile (h_local*4 + c_t)
//           of the [heads, 1, 128] output (q heads [0, NK), k [NK, 2NK), v [2NK, 2NK + NV)), whose padding rows stay
//           zero (the buffers are persistent and zeroed once).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_out = 3, cb_wst = 4;

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t NK = get_compile_time_arg_val(1);
    constexpr auto st_a = TensorAccessorArgs<2>();
    constexpr auto o_a = TensorAccessorArgs<st_a.next_compile_time_args_offset()>();  // q, k, v share it

    // runtime: 0..K-1 state addrs, K q, K+1 k, K+2 v, K+3 r_t, K+4 c_t, K+5 half
    const uint32_t r_t = get_arg_val<uint32_t>(K + 3);
    const uint32_t c_t = get_arg_val<uint32_t>(K + 4);
    const uint32_t half = get_arg_val<uint32_t>(K + 5);
    const uint32_t tile = r_t * 4 + c_t;
    const uint32_t h0 = 32 * r_t + 16 * half;

    CircularBuffer w(cb_wst), o(cb_out);
    const uint32_t tb = get_tile_size(cb_wst);  // bf16
    const uint32_t ob = get_tile_size(cb_out);  // fp32 4096 B
    Noc noc;

    w.wait_front(K);
    for (uint32_t j = 0; j < K; j++) {
        const auto st = TensorAccessor(st_a, get_arg_val<uint32_t>(j), tb);
        noc.async_write(w, st, 1024, {.offset_bytes = j * tb + half * 1024}, {.page_id = tile, .offset_bytes = half * 1024});
    }

    o.wait_front(1);
    const auto oq = TensorAccessor(o_a, get_arg_val<uint32_t>(K), ob);
    const auto ok = TensorAccessor(o_a, get_arg_val<uint32_t>(K + 1), ob);
    const auto ov = TensorAccessor(o_a, get_arg_val<uint32_t>(K + 2), ob);
    for (uint32_t i = 0; i < 16; i++) {
        const uint32_t h = h0 + i;
        const uint32_t src = (2 * half * 256 + i * 16) * 4;  // fp32 face 2*half, row i
        const auto& dst = (h < NK) ? oq : ((h < 2 * NK) ? ok : ov);
        const uint32_t hl = (h < NK) ? h : ((h < 2 * NK) ? h - NK : h - 2 * NK);
        noc.async_write(o, dst, 64, {.offset_bytes = src}, {.page_id = hl * 4 + c_t, .offset_bytes = 0});
        noc.async_write(o, dst, 64, {.offset_bytes = src + 1024}, {.page_id = hl * 4 + c_t, .offset_bytes = 1024});
    }
    noc.async_write_barrier();
    w.pop_front(K);
    o.pop_front(1);
}
