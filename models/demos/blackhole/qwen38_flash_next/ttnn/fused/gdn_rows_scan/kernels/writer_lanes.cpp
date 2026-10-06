// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// The lanes form of gdn_rows_scan's writer: one (value head, lane) item per core, the whole head per core (VBT = HT),
// the compute kernel unchanged (its rows r = 0 .. R-1 are lane u's rows at tile rows t = lane * R + r).  Row r's o
// tiles arrive from the compute (CB_OBF, four tiles; the q S matmul is over the whole tile, so the row that matters
// is tile row t) and row t of each is copied into the head's assembled o-rows tiles (CB_OROWS, zeroed first: every
// other lane's rows and the pad rows stay exactly zero through the gated norm); the state after r + 1 rows arrives
// as CB_OUTS and goes to PREFIX SLOT t (the lanes prefix [B * R, 12, 128, 128]: slot lane * R + r = lane u after r + 1
// rows, so the page formula is the single form's with r -> t); after the last row the o-rows tiles go to the
// compute's gated norm and ONLY lane u's rows of the head's four gated column tiles are written to the output (8 R
// row pieces of 32 B by page + offset; the other lanes' cores write theirs, and the cores of the last lane write the
// pad rows B * R .. 31 as exact zeros).  No cross-core exchange: one core owns one lane of one head.
// Compile-time args: 0 ROWS (R, rows per lane), 1 VBT (= 4), 2 LANES (B); then TensorAccessorArgs of prefix
// ([B * R, 12, 128, 128] fp32) and out ([1, 1, 32, 1536] bf16).  Runtime args: 0 prefix, 1 out addresses, 2 items,
// then (head, lane) pairs.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"
#include "../../kernels/row_mask.h"

namespace {
constexpr uint32_t CB_OUTS = 25, CB_OUTG = 26, CB_OROWS = 29, CB_OBF = 31;
constexpr uint32_t ROWS = get_compile_time_arg_val(0);
constexpr uint32_t VBT = get_compile_time_arg_val(1);
constexpr uint32_t LANES = get_compile_time_arg_val(2);
constexpr uint32_t HEADS = 12, HT = 4, ST = HT * VBT, STATE_TILES = HT * HT, TILE_ROWS = 32;
constexpr uint32_t BF16_TILE = 2048, FP32_TILE = 4096;
constexpr uint32_t FACE_BYTES = fused_rows::FACE_BYTES, HALF_ROW = fused_rows::HALF_ROW_BYTES;
static_assert(VBT == HT, "the lanes form runs the whole head on one core");
static_assert(ROWS >= 1 && LANES >= 1 && LANES * ROWS <= TILE_ROWS, "B lanes x R rows in one tile");
}  // namespace

void kernel_main() {
    constexpr auto prefix_args = TensorAccessorArgs<3>();
    constexpr auto out_args = TensorAccessorArgs<prefix_args.next_compile_time_args_offset()>();
    uint32_t arg = 0;
    const uint32_t prefix_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t items = get_arg_val<uint32_t>(arg++);
    const auto prefix = TensorAccessor(prefix_args, prefix_addr);
    const auto out = TensorAccessor(out_args, out_addr);

    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t lane = get_arg_val<uint32_t>(arg++);
        const uint32_t row0 = lane * ROWS;

        cb_reserve_back(CB_OROWS, HT);
        const uint32_t orows = get_write_ptr(CB_OROWS);
        fused_rows::fill_words(orows, HT * BF16_TILE / 4, 0);  // every row not lane u's stays exactly zero

        auto drain_state = [&](uint32_t r) {  // lane u's state after r + 1 rows -> prefix slot row0 + r
            cb_wait_front(CB_OUTS, ST);
            const uint32_t l1 = get_read_ptr(CB_OUTS);
            for (uint32_t kt = 0; kt < HT; ++kt) {
                for (uint32_t c = 0; c < VBT; ++c) {
                    const uint32_t page = ((row0 + r) * HEADS + head) * STATE_TILES + kt * HT + c;
                    noc_async_write_page(page, prefix, l1 + (kt * VBT + c) * FP32_TILE);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(CB_OUTS, ST);
        };

        {
            FUSED_ZONE("fz_gsc_lw_rows");
            for (uint32_t r = 0; r < ROWS; ++r) {
                cb_wait_front(CB_OBF, VBT);
                const uint32_t src = get_read_ptr(CB_OBF);
                for (uint32_t j = 0; j < VBT; ++j) {
                    fused_rows::copy_row_bf16(orows + j * BF16_TILE, src + j * BF16_TILE, row0 + r);
                }
                cb_pop_front(CB_OBF, VBT);
                if (r + 1 < ROWS) {
                    drain_state(r);
                }
            }
        }
        cb_push_back(CB_OROWS, HT);  // the gated norm may start; the last prefix state drains meanwhile
        {
            FUSED_ZONE("fz_gsc_lw_state");
            drain_state(ROWS - 1);
        }
        {
            FUSED_ZONE("fz_gsc_lw_gated");
            cb_wait_front(CB_OUTG, HT);
            const uint32_t l1 = get_read_ptr(CB_OUTG);
            // lane u's rows of the head's four gated column tiles: two 32 B face pieces per row per tile
            for (uint32_t c = 0; c < HT; ++c) {
                const uint32_t page = head * HT + c;
                for (uint32_t r = 0; r < ROWS; ++r) {
                    const uint32_t off = fused_rows::bf16_row_offset(row0 + r);
                    noc_async_write(l1 + c * BF16_TILE + off, out.get_noc_addr(page, off), HALF_ROW);
                    noc_async_write(
                        l1 + c * BF16_TILE + off + FACE_BYTES, out.get_noc_addr(page, off + FACE_BYTES), HALF_ROW);
                }
                if (lane + 1 == LANES) {  // the pad rows past B * R: exact zeros (the o-rows tiles' untouched rows)
                    for (uint32_t t = LANES * ROWS; t < TILE_ROWS; ++t) {
                        const uint32_t off = fused_rows::bf16_row_offset(t);
                        noc_async_write(orows + c * BF16_TILE + off, out.get_noc_addr(page, off), HALF_ROW);
                        noc_async_write(
                            orows + c * BF16_TILE + off + FACE_BYTES,
                            out.get_noc_addr(page, off + FACE_BYTES),
                            HALF_ROW);
                    }
                }
            }
            noc_async_write_barrier();
            cb_pop_front(CB_OUTG, HT);
        }
    }
}
