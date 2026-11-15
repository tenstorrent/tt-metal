// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Reader for MoeCombine. Reads the [E, 32] bf16 weight table once into cb_tab (one 64-byte row per expert), zeroes
// the cb_d / cb_w slots once, then per output column tile c (this core's units) and per local expert e pushes a
// weight tile (column 0 of row u = w[e][u], zeros elsewhere) into cb_w and the down tile (e, c) into cb_d -- the
// read is skipped (the zero slot goes in) when no user < R selected e, so the product is zero and the tile is
// never fetched. The expert count is fixed: the compute kernel's math core cannot read a data-dependent one.
//
// compile-time args: [cb_d, cb_w, cb_tab, E, R, cols, w_page_bytes, slots] + TensorAccessorArgs(down, w)
// runtime args: [down_addr, w_addr, start column, column count]
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    const uint32_t down_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t start = get_arg_val<uint32_t>(2);
    const uint32_t count = get_arg_val<uint32_t>(3);
    constexpr uint32_t cb_d = get_compile_time_arg_val(0);
    constexpr uint32_t cb_w = get_compile_time_arg_val(1);
    constexpr uint32_t cb_tab = get_compile_time_arg_val(2);
    constexpr uint32_t E = get_compile_time_arg_val(3);
    constexpr uint32_t R = get_compile_time_arg_val(4);
    constexpr uint32_t cols = get_compile_time_arg_val(5);
    constexpr uint32_t w_page = get_compile_time_arg_val(6);
    constexpr uint32_t slots = get_compile_time_arg_val(7);
    constexpr auto down_args = TensorAccessorArgs<8>();
    constexpr auto w_args = TensorAccessorArgs<down_args.next_compile_time_args_offset()>();
    constexpr uint32_t tile_bytes = 2048;
    constexpr uint32_t row_elems = w_page / 2;

    const auto s_down = TensorAccessor(down_args, down_addr, tile_bytes);
    const auto s_w = TensorAccessor(w_args, w_addr, w_page);
    CircularBuffer cbd(cb_d), cbw(cb_w), cbt(cb_tab);

    cbt.reserve_back(1);
    const uint32_t table_addr = get_write_ptr(cb_tab);
    for (uint32_t e = 0; e < E; ++e) {
        noc_async_read(s_w.get_noc_addr(e), table_addr + e * w_page, w_page);
    }
    noc_async_read_barrier();
    cbt.push_back(1);
    volatile tt_l1_ptr uint16_t* table = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(table_addr);
    bool active[E > 0 ? E : 1];
    for (uint32_t e = 0; e < E; ++e) {
        active[e] = false;
        for (uint32_t u = 0; u < R; ++u) {
            if ((table[e * row_elems + u] & 0x7FFF) != 0) {  // a non-zero bf16 (either sign)
                active[e] = true;
                break;
            }
        }
    }
    // zero every cb_d and cb_w slot once; afterwards only column 0 of a cb_w slot and the fetched cb_d tiles change
    for (uint32_t cb : {cb_d, cb_w}) {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb));
        for (uint32_t i = 0; i < slots * tile_bytes / 4; ++i) {
            z[i] = 0;
        }
    }
    Noc noc;
    for (uint32_t c = start; c < start + count; ++c) {
        for (uint32_t e = 0; e < E; ++e) {
            cbd.reserve_back(1);
            cbw.reserve_back(1);
            volatile tt_l1_ptr uint16_t* wt = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_w));
            if (active[e]) {
                noc.async_read(s_down, cbd, tile_bytes, {.page_id = e * cols + c}, {.offset_bytes = 0});
                // weight tile: column 0 of row u lives in face (u / 16) * 2 at element (u % 16) * 16
                for (uint32_t u = 0; u < 32; ++u) {
                    wt[(u >> 4) * 512 + (u & 15) * 16] = (u < R) ? table[e * row_elems + u] : uint16_t(0);
                }
                noc.async_read_barrier();
            } else {
                // a skipped expert: zero weights (the cb_d slot holds a finite stale tile; the product is zero)
                for (uint32_t u = 0; u < 32; ++u) {
                    wt[(u >> 4) * 512 + (u & 15) * 16] = 0;
                }
            }
            cbd.push_back(1);
            cbw.push_back(1);
        }
    }
}
