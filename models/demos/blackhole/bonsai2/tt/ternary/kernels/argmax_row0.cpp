// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Greedy token of one decode step (argmax.py): the argmax of row 0 of a bf16 TILE logits row [.., 32, Npad] over its
// first V columns. Each core scans row 0 of its tiles (two 64-B reads per tile: face 0 and face 1, rows 0-1) on the
// RISC, keeping the largest value (bf16 bits mapped to an order-preserving key; the first index wins ties); every
// core sends (key, index) to core 0, which reduces them and writes the index.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "core_span.hpp"

namespace {
FORCE_INLINE uint32_t order_key(uint16_t b) { return (b & 0x8000) ? (uint32_t)(uint16_t)~b : (uint32_t)(b | 0x8000); }
}  // namespace

void kernel_main() {
    // common args: logits, out, Nt, ncores, grid_x, V, hub NOC x, hub NOC y
    const uint32_t y_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t ncores = get_common_arg_val<uint32_t>(3);
    const uint32_t V = get_common_arg_val<uint32_t>(5);
    const uint32_t hub_x = get_common_arg_val<uint32_t>(6), hub_y = get_common_arg_val<uint32_t>(7);
    uint32_t first, n;
    core_span(get_common_arg_val<uint32_t>(2), ncores, get_common_arg_val<uint32_t>(4), 1, first, n);
    const uint32_t c = core_index(get_common_arg_val<uint32_t>(4));
    constexpr auto y_args = TensorAccessorArgs<0>();
    constexpr auto o_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
    const auto ys = TensorAccessor(y_args, y_addr, 2048);
    constexpr uint32_t cb_s = 0, cb_p = 1;  // scratch rows; partials (16 B per core, on the hub)
    const uint32_t sc = get_write_ptr(cb_s), parts = get_write_ptr(cb_p);

    uint32_t best_key = 0, best_idx = 0xFFFFFFFFu;
    constexpr uint32_t BATCH = 16;  // tiles per read batch (128 B each)
    for (uint32_t t0 = 0; t0 < n; t0 += BATCH) {
        const uint32_t nb = n - t0 < BATCH ? n - t0 : BATCH;
        for (uint32_t k = 0; k < nb; ++k) {
            const uint64_t a = ys.get_noc_addr(first + t0 + k);
            noc_async_read(a, sc + k * 128, 64);
            noc_async_read(a + 512, sc + k * 128 + 64, 64);
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < nb; ++k) {
            const uint32_t col0 = (first + t0 + k) * 32;
            const volatile tt_l1_ptr uint16_t* r = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(sc + k * 128);
            for (uint32_t j = 0; j < 32; ++j) {
                const uint32_t col = col0 + j;
                if (col >= V) {
                    break;
                }
                const uint32_t key = order_key(r[j < 16 ? j : 32 + (j - 16)]);  // face 1 row 0 starts 64 B in
                if (key > best_key || (key == best_key && col < best_idx)) {
                    best_key = key;
                    best_idx = col;
                }
            }
        }
    }
    const uint32_t sem = get_semaphore(0);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(parts + c * 16);
    p[0] = best_key;
    p[1] = best_idx;
    if (c != 0) {
        noc_async_write(parts + c * 16, get_noc_addr(hub_x, hub_y, parts + c * 16), 16);
        noc_async_writes_flushed();
        noc_semaphore_inc(get_noc_addr(hub_x, hub_y, sem), 1);
        noc_async_atomic_barrier();
        return;
    }
    noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), ncores - 1);
    invalidate_l1_cache();
    for (uint32_t i = 1; i < ncores; ++i) {
        const volatile tt_l1_ptr uint32_t* q = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(parts + i * 16);
        if (q[0] > best_key || (q[0] == best_key && q[1] < best_idx)) {
            best_key = q[0];
            best_idx = q[1];
        }
    }
    volatile tt_l1_ptr uint32_t* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sc);
    for (uint32_t i = 0; i < 16; ++i) {
        out[i] = i == 0 ? best_idx : 0;
    }
    const auto os = TensorAccessor(o_args, o_addr, 64);
    noc_async_write(sc, os.get_noc_addr(0), 64);
    noc_async_write_barrier();
}
