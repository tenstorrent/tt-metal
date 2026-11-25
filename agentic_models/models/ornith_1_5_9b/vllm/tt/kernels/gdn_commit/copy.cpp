// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Commit a chained verify of B requests x T candidates: request u (pool row u) accepted candidate a_u (device index
// tensor, -1 = leave the request untouched), so the program is trace-safe. Work items:
//  - recurrent: (layer, u) -> copy chain heads (u*T + a_u)*8 .. +7 (16 FP32 tiles each) to live heads u*8 .. u*8+7;
//  - history: one item per layer (one core owns all rows of the layer's shared tiles: no concurrent read-modify-write
//    of a tile) -> row u*T + a_u of the chain taps into row u of the live taps, for every committing u.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

static inline uint32_t row_offset(uint32_t r) { return (r / 16) * 1024 + (r % 16) * 32; }

void kernel_main() {
    constexpr auto ia = TensorAccessorArgs<0>();
    constexpr auto sa = TensorAccessorArgs<ia.next_compile_time_args_offset()>();
    constexpr auto ha = TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto index = TensorAccessor(ia, get_arg_val<uint32_t>(0), 128);
    const uint32_t n_rec = get_arg_val<uint32_t>(1), n_hist = get_arg_val<uint32_t>(2), T = get_arg_val<uint32_t>(3);
    CircularBuffer buf(0), tiles(1);
    buf.reserve_back(16);
    tiles.reserve_back(2);
    const uint32_t slab = get_write_ptr(0), pair = get_write_ptr(1);
    noc_async_read(index.get_noc_addr(0), pair, 128);
    noc_async_read_barrier();
    int32_t accept[32];
    for (uint32_t u = 0; u < 32; ++u) {
        accept[u] = reinterpret_cast<volatile tt_l1_ptr int32_t*>(pair)[u];
    }
    uint32_t arg = 4;
    for (uint32_t i = 0; i < n_rec; ++i, arg += 3) {
        const auto chain = TensorAccessor(sa, get_arg_val<uint32_t>(arg), 4096);
        const auto live = TensorAccessor(sa, get_arg_val<uint32_t>(arg + 1), 4096);
        const uint32_t u = get_arg_val<uint32_t>(arg + 2);
        if (accept[u] < 0) {
            continue;
        }
        const uint32_t a = static_cast<uint32_t>(accept[u]);
        for (uint32_t h = 0; h < 8; ++h) {
            for (uint32_t t = 0; t < 16; ++t) {
                noc_async_read(chain.get_noc_addr(((u * T + a) * 8 + h) * 16 + t), slab + t * 4096, 4096);
            }
            noc_async_read_barrier();
            for (uint32_t t = 0; t < 16; ++t) {
                noc_async_write(slab + t * 4096, live.get_noc_addr((u * 8 + h) * 16 + t), 4096);
            }
            noc_async_write_barrier();
        }
    }
    for (uint32_t i = 0; i < n_hist; ++i, arg += 2) {
        const auto chain = TensorAccessor(ha, get_arg_val<uint32_t>(arg), 2048);
        const auto live = TensorAccessor(ha, get_arg_val<uint32_t>(arg + 1), 2048);
        for (uint32_t tile = 0; tile < 3 * 64; ++tile) {
            noc_async_read(chain.get_noc_addr(tile), pair, 2048);
            noc_async_read(live.get_noc_addr(tile), pair + 2048, 2048);
            noc_async_read_barrier();
            for (uint32_t u = 0; u < 32; ++u) {
                if (accept[u] < 0) {
                    continue;
                }
                volatile tt_l1_ptr uint32_t* s =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pair + row_offset(u * T + accept[u]));
                volatile tt_l1_ptr uint32_t* d =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pair + 2048 + row_offset(u));
                for (uint32_t w = 0; w < 8; ++w) {
                    d[w] = s[w];
                    d[w + 128] = s[w + 128];
                }
            }
            noc_async_write(pair + 2048, live.get_noc_addr(tile), 2048);
            noc_async_write_barrier();
        }
    }
}
