// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Group B writer.
//   * A z unit (kind 1..12, value head kind - 1) writes cb_sig's four bf16 tiles to `sig` [1, 1, T, 1536]
//     token-major, page c * 48 + 4 hv + d (the same page map as `v`); sig takes no row mask, as in the chain.
//   * The a/b unit (kind 0) turns the fp32 beta and g tiles (tokens down the rows, heads across the columns) into the
//     prims' [12, NC, 32, 1] form: head h's 32 values go into column 0 of a zeroed scratch tile (cb_col, restored
//     after each page) written to page h * NC + c. Rows at or past `rows` stay exact zeros, the +0 of the chain's
//     fp32 row-mask multiply.
//
// Compile-time args: 0 chunks (NC); then TensorAccessorArgs of beta_c, g_c, sig.  Runtime args: 0 beta_c, 1 g_c,
// 2 sig addresses, 3 rows, 4 units, then (tile row, kind) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

namespace {
constexpr uint32_t cb_beta = 23, cb_g = 24, cb_sig = 25, cb_col = 26;
constexpr uint32_t head_tiles = 4, value_heads = 12, value_tiles = 48;
constexpr uint32_t tile_rows = 32, face_elements = 256, face_columns = 16, bf16_tile_bytes = 2048,
                   fp32_tile_bytes = 4096;

constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * face_elements + (row & 15) * face_columns + (col & 15);
}

void fill_words(uint32_t l1, uint32_t words, uint32_t value) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t k = 0; k < words; ++k) {
        p[k] = value;
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t chunks = get_compile_time_arg_val(0);
    constexpr auto beta_args = TensorAccessorArgs<1>();
    constexpr auto g_args = TensorAccessorArgs<beta_args.next_compile_time_args_offset()>();
    constexpr auto sig_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    const uint32_t beta_addr = get_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_arg_val<uint32_t>(1);
    const uint32_t sig_addr = get_arg_val<uint32_t>(2);
    const uint32_t rows = get_arg_val<uint32_t>(3);
    const uint32_t units = get_arg_val<uint32_t>(4);
    uint32_t arg = 5;

    const auto beta_c = TensorAccessor(beta_args, beta_addr);
    const auto g_c = TensorAccessor(g_args, g_addr);
    const auto sig = TensorAccessor(sig_args, sig_addr);

    const uint32_t last_chunk = (rows - 1) / tile_rows;
    const uint32_t partial = rows % tile_rows;
    // The column scratch: never pushed or popped, so its write pointer is stable for the whole kernel.
    const uint32_t scratch = get_write_ptr(cb_col);
    fill_words(scratch, fp32_tile_bytes / 4, 0);

    for (uint32_t unit = 0; unit < units; ++unit) {
        const uint32_t chunk = get_arg_val<uint32_t>(arg++);
        const uint32_t kind = get_arg_val<uint32_t>(arg++);
        const uint32_t keep = chunk > last_chunk ? 0 : (chunk == last_chunk && partial != 0 ? partial : tile_rows);

        if (kind == 0) {
            cb_wait_front(cb_beta, 1);
            cb_wait_front(cb_g, 1);
            const uint32_t sources[2] = {get_read_ptr(cb_beta), get_read_ptr(cb_g)};
            volatile tt_l1_ptr uint32_t* column = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
            for (uint32_t which = 0; which < 2; ++which) {
                volatile tt_l1_ptr uint32_t* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sources[which]);
                for (uint32_t head = 0; head < value_heads; ++head) {
                    for (uint32_t row = 0; row < keep; ++row) {
                        column[face_element(row, 0)] = source[face_element(row, head)];
                    }
                    const uint32_t page = head * chunks + chunk;
                    noc_async_write_page(page, which == 0 ? beta_c : g_c, scratch);
                    noc_async_write_barrier();
                    for (uint32_t row = 0; row < keep; ++row) {
                        column[face_element(row, 0)] = 0;
                    }
                }
            }
            cb_pop_front(cb_beta, 1);
            cb_pop_front(cb_g, 1);
        } else {
            const uint32_t head = kind - 1;
            cb_wait_front(cb_sig, head_tiles);
            // sig carries no row mask: the chain's z sigmoid has none
            const uint32_t l1 = get_read_ptr(cb_sig);
            for (uint32_t d = 0; d < head_tiles; ++d) {
                noc_async_write_page(chunk * value_tiles + head * head_tiles + d, sig, l1 + d * bf16_tile_bytes);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_sig, head_tiles);
        }
    }
}
