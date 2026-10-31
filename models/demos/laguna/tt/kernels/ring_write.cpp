// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// DFlash context K/V ring write (Laguna; ring_write.py): rows 0..W-1 of each new [1, nkv, 32, hd] K or V tensor go
// to ring slots slots[0..W-1] of its [1, nkv, RING, hd] ring. Core u handles ring u / (nkv * hdt), head
// (u / hdt) % nkv, column tile u % hdt: it reads the source tile, then for each ring tile row holding a target slot
// reads that ring tile, copies the rows in and writes it back. Row r of a 32x32 bf16 tile is 32 bytes at
// face (r / 16) * 2, offset (r % 16) * 32, and 32 more in the next face (512 bytes later).
// Common runtime args: slots address, W, then the R source addresses, then the R ring addresses.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t grid_x = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t nkv = get_compile_time_arg_val(2);
    constexpr uint32_t hdt = get_compile_time_arg_val(3);       // head_dim tiles
    constexpr uint32_t ring_rows = get_compile_time_arg_val(4);  // ring tile rows (RING / 32)
    constexpr auto i_args = TensorAccessorArgs<5>();
    constexpr auto s_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    const uint32_t u = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    if (u >= R * nkv * hdt) {
        return;
    }
    const uint32_t ring = u / (nkv * hdt), h = (u / hdt) % nkv, j = u % hdt;
    const uint32_t W = get_common_arg_val<uint32_t>(1);
    const auto idx = TensorAccessor(i_args, get_common_arg_val<uint32_t>(0), 128);
    const auto src = TensorAccessor(s_args, get_common_arg_val<uint32_t>(2 + ring), 2048);
    const auto dst = TensorAccessor(r_args, get_common_arg_val<uint32_t>(2 + R + ring), 2048);
    const uint32_t sbuf = get_write_ptr(0), dbuf = sbuf + 2048, ibuf = get_write_ptr(1);
    noc_async_read(idx.get_noc_addr(0), ibuf, 128);
    noc_async_read(src.get_noc_addr(h * hdt + j), sbuf, 2048);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* slots = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ibuf);
    uint32_t done = 0;  // bit i: row i written
    for (uint32_t i = 0; i < W; ++i) {
        if (done & (1u << i)) {
            continue;
        }
        const uint32_t tr = slots[i] / 32;
        const uint32_t tile = (h * ring_rows + tr) * hdt + j;
        noc_async_read(dst.get_noc_addr(tile), dbuf, 2048);
        noc_async_read_barrier();
        for (uint32_t r = i; r < W; ++r) {
            if (slots[r] / 32 != tr) {
                continue;
            }
            done |= 1u << r;
            const uint32_t d = slots[r] % 32;
            const uint32_t so = (r / 16) * 1024 + (r % 16) * 32, dofs = (d / 16) * 1024 + (d % 16) * 32;
            for (uint32_t half = 0; half < 2; ++half) {
                volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sbuf + so + half * 512);
                volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dbuf + dofs + half * 512);
                for (uint32_t w = 0; w < 8; ++w) {
                    o[w] = s[w];
                }
            }
        }
        noc_async_write(dbuf, dst.get_noc_addr(tile), 2048);
        noc_async_write_barrier();
    }
}
