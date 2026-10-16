// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Group B writer.
//
//   * A z unit (kind 1..12, value head kind - 1) writes the four bf16 tiles of CB_SIG to `sig` [1, 1, T, 1536]
//     token-major, page c * 48 + 4 hv + d -- the same flat page map as `v`.  sig takes NO row mask: the chain's
//     z sigmoid (_gate_and_project_rows) has none.
//   * The a/b unit (kind 0) turns the fp32 beta and g tiles -- 32 tokens down the rows, the 12 value heads across
//     the columns, columns 12..31 never read -- into the prims' [12, NC, 32, 1] form: head h's 32 values are copied
//     into COLUMN 0 of a zeroed 4 KB tile and written to page h * NC + c (the prep reader's g/beta address
//     hc * Ct at Ct = 1).  That is data movement, so it is bitwise; the scratch tile is CB_COL, zeroed once and
//     restored after each page, so a head costs 64 word stores instead of a full 1024-word clear.
//
// Rows at or past `rows` are left as exact zeros in beta and g: the chain multiplies those rows by the 0.0 of its
// fp32 row mask, whose product the die pins at +0.  That falls out of the zeroed scratch (only rows below `rows`
// are copied into column 0).
//
// Compile-time args: 0 chunks (NC); then TensorAccessorArgs of beta_c, g_c, sig.  Runtime args: 0 beta_c, 1 g_c,
// 2 sig addresses, 3 rows, 4 units, then (tile row, kind) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"

namespace {
constexpr uint32_t CB_BETA = 23, CB_G = 24, CB_SIG = 25, CB_COL = 26;
constexpr uint32_t HEAD_TILES = 4, HEADS = 12, VALUE_TILES = 48;
constexpr uint32_t TILE_ROWS = 32, FACE_ELEMS = 256, FACE_COLS = 16, BF16_TILE = 2048, FP32_TILE = 4096;

constexpr uint32_t face_element(uint32_t row, uint32_t col) {
    return ((row >> 4) * 2 + (col >> 4)) * FACE_ELEMS + (row & 15) * FACE_COLS + (col & 15);
}

void fill_words(uint32_t l1, uint32_t words, uint32_t value) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t k = 0; k < words; ++k) {
        p[k] = value;
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t CHUNKS = get_compile_time_arg_val(0);
    constexpr auto b_args = TensorAccessorArgs<1>();
    constexpr auto g_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    const uint32_t b_addr = get_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_arg_val<uint32_t>(2);
    const uint32_t rows = get_arg_val<uint32_t>(3);
    const uint32_t units = get_arg_val<uint32_t>(4);
    uint32_t arg = 5;

    const auto beta_c = TensorAccessor(b_args, b_addr);
    const auto g_c = TensorAccessor(g_args, g_addr);
    const auto sig = TensorAccessor(s_args, s_addr);

    const uint32_t last_chunk = (rows - 1) / TILE_ROWS;
    const uint32_t partial = rows % TILE_ROWS;
    // The column scratch: never pushed or popped, so its write pointer is stable for the whole kernel.
    const uint32_t scratch = get_write_ptr(CB_COL);
    fill_words(scratch, FP32_TILE / 4, 0);

    for (uint32_t unit = 0; unit < units; ++unit) {
        const uint32_t chunk = get_arg_val<uint32_t>(arg++);
        const uint32_t kind = get_arg_val<uint32_t>(arg++);
        const uint32_t keep = chunk > last_chunk ? 0 : (chunk == last_chunk && partial != 0 ? partial : TILE_ROWS);

        if (kind == 0) {
            FUSED_ZONE("fz_gpr_wg_gates");
            cb_wait_front(CB_BETA, 1);
            cb_wait_front(CB_G, 1);
            const uint32_t sources[2] = {get_read_ptr(CB_BETA), get_read_ptr(CB_G)};
            volatile tt_l1_ptr uint32_t* column = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
            for (uint32_t which = 0; which < 2; ++which) {
                volatile tt_l1_ptr uint32_t* source = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sources[which]);
                for (uint32_t head = 0; head < HEADS; ++head) {
                    for (uint32_t row = 0; row < keep; ++row) {
                        column[face_element(row, 0)] = source[face_element(row, head)];
                    }
                    const uint32_t page = head * CHUNKS + chunk;
                    noc_async_write_page(page, which == 0 ? beta_c : g_c, scratch);
                    noc_async_write_barrier();
                    for (uint32_t row = 0; row < keep; ++row) {
                        column[face_element(row, 0)] = 0;
                    }
                }
            }
            cb_pop_front(CB_BETA, 1);
            cb_pop_front(CB_G, 1);
        } else {
            FUSED_ZONE("fz_gpr_wg_sig");
            const uint32_t head = kind - 1;
            cb_wait_front(CB_SIG, HEAD_TILES);
            // sig carries NO row mask: _gate_and_project_rows applies none to the z sigmoid (only q/k/v/beta/g are
            // masked, in _make_chunk_inputs), so every row of every tile row is written as computed
            const uint32_t l1 = get_read_ptr(CB_SIG);
            for (uint32_t d = 0; d < HEAD_TILES; ++d) {
                noc_async_write_page(chunk * VALUE_TILES + head * HEAD_TILES + d, sig, l1 + d * BF16_TILE);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_SIG, HEAD_TILES);
        }
    }
}
