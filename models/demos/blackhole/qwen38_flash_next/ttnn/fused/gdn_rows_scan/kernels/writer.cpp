// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Per (value head, state column block) item: row r's o tiles arrive from the compute (CB_OBF, VBT tiles) and row r
// of each is copied into the head's assembled o-rows tiles (CB_OROWS, four bf16 tiles; the item's own tiles are
// zeroed first, so rows >= ROWS stay exactly zero); the state after r + 1 rows arrives as CB_OUTS (ST fp32 tiles) and
// goes to prefix slot r; after the last row the o-rows tiles are handed to the compute's gated norm and the item's
// gated tiles are written to the output.  In the column-block form (VBT = 1) the four cores of one head first
// exchange their o-rows tiles through L1: each writes its tile into the peers' CB_OROWS slot (the same L1 address on
// every core of the program), then increments the peers' semaphore and waits for the three increments of its own, so
// every core runs the gated norm on the same four tiles in the same order and writes only its own column tile.
// Compile-time args: 0 ROWS, 1 VBT; then TensorAccessorArgs of prefix ([ROWS, 12, 128, 128] fp32) and out
// ([1, 1, 32, 1536] bf16).  Runtime args: 0 prefix, 1 out addresses, 2 items, then per item (head, vb) and the PEERS
// peers' (noc x, noc y).  Semaphore 0: the exchange counter.

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include "../../kernels/zones.h"
#include "../../kernels/row_mask.h"

namespace {
constexpr uint32_t CB_OUTS = 25, CB_OUTG = 26, CB_DBG = 28, CB_OROWS = 29, CB_OBF = 31;
constexpr uint32_t ROWS = get_compile_time_arg_val(0);
constexpr uint32_t VBT = get_compile_time_arg_val(1);
constexpr uint32_t HEADS = 12, HT = 4, ST = HT * VBT, STATE_TILES = HT * HT;
constexpr uint32_t PEERS = HT / VBT - 1;  // 0 for the whole-head item, 3 for one column block of four
constexpr uint32_t BF16_TILE = 2048, FP32_TILE = 4096;
constexpr uint32_t DBG_TILES =
    11 + 7 * ROWS;  // per item: conv q0, conv v0, q0..3, k0..3, kcol0; per row 2 gate + 5 taps
static_assert(VBT == HT || VBT == 1, "a whole head or one of four column blocks");
}  // namespace

void kernel_main() {
    constexpr auto prefix_args = TensorAccessorArgs<2>();
    constexpr auto out_args = TensorAccessorArgs<prefix_args.next_compile_time_args_offset()>();
#ifdef DEBUG_TAPS
    constexpr auto dbg_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
#endif
    uint32_t arg = 0;
    const uint32_t prefix_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
#ifdef DEBUG_TAPS
    const uint32_t dbg_addr = get_arg_val<uint32_t>(arg++);
    const auto dbg = TensorAccessor(dbg_args, dbg_addr);
#endif
    const uint32_t items = get_arg_val<uint32_t>(arg++);
    const auto prefix = TensorAccessor(prefix_args, prefix_addr);
    const auto out = TensorAccessor(out_args, out_addr);
    const uint32_t sem_addr = get_semaphore(0);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_addr);

    for (uint32_t item = 0; item < items; ++item) {
        const uint32_t head = get_arg_val<uint32_t>(arg++);
        const uint32_t vb = get_arg_val<uint32_t>(arg++);
        uint32_t peer_x[HT], peer_y[HT];
        for (uint32_t p = 0; p < PEERS; ++p) {
            peer_x[p] = get_arg_val<uint32_t>(arg++);
            peer_y[p] = get_arg_val<uint32_t>(arg++);
        }

        cb_reserve_back(CB_OROWS, HT);
        const uint32_t orows = get_write_ptr(CB_OROWS);
        const uint32_t mine = orows + vb * VBT * BF16_TILE;  // the item's own column tiles; peers fill the others
        fused_rows::fill_words(mine, VBT * BF16_TILE / 4, 0);

        auto drain_state = [&](uint32_t r) {  // the state after r + 1 rows -> prefix slot r
            cb_wait_front(CB_OUTS, ST);
            const uint32_t l1 = get_read_ptr(CB_OUTS);
            for (uint32_t kt = 0; kt < HT; ++kt) {
                for (uint32_t c = 0; c < VBT; ++c) {
                    const uint32_t page = (r * HEADS + head) * STATE_TILES + kt * HT + vb * VBT + c;
                    noc_async_write_page(page, prefix, l1 + (kt * VBT + c) * FP32_TILE);
                }
            }
            noc_async_write_barrier();
            cb_pop_front(CB_OUTS, ST);
        };

        {
            FUSED_ZONE("fz_gsc_w_rows");
            for (uint32_t r = 0; r < ROWS; ++r) {
                cb_wait_front(CB_OBF, VBT);
                const uint32_t src = get_read_ptr(CB_OBF);
                for (uint32_t j = 0; j < VBT; ++j) {
                    fused_rows::copy_row_bf16(mine + j * BF16_TILE, src + j * BF16_TILE, r);
                }
                cb_pop_front(CB_OBF, VBT);
                if (r + 1 < ROWS) {
                    drain_state(r);
                }
            }
        }
        if constexpr (PEERS > 0) {
            FUSED_ZONE("fz_gsc_w_exchange");
            for (uint32_t p = 0; p < PEERS; ++p) {
                noc_async_write(mine, get_noc_addr(peer_x[p], peer_y[p], mine), VBT * BF16_TILE);
            }
            noc_async_write_barrier();
            for (uint32_t p = 0; p < PEERS; ++p) {
                noc_semaphore_inc(get_noc_addr(peer_x[p], peer_y[p], sem_addr), 1);
            }
            noc_semaphore_wait_min(sem, PEERS);
            noc_semaphore_set(sem, 0);
        }
        cb_push_back(CB_OROWS, HT);  // the gated norm may start; the last prefix state drains meanwhile
        {
            FUSED_ZONE("fz_gsc_w_state");
            drain_state(ROWS - 1);
        }
        {
            FUSED_ZONE("fz_gsc_w_gated");
            cb_wait_front(CB_OUTG, HT);
            const uint32_t l1 = get_read_ptr(CB_OUTG);
            for (uint32_t c = 0; c < VBT; ++c) {
                const uint32_t col = vb * VBT + c;
                noc_async_write_page(head * HT + col, out, l1 + col * BF16_TILE);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_OUTG, HT);
        }
#ifdef DEBUG_TAPS
        cb_wait_front(CB_DBG, DBG_TILES);
        {
            const uint32_t l1 = get_read_ptr(CB_DBG);
            for (uint32_t t = 0; t < DBG_TILES; ++t) {
                noc_async_write_page((head * (HT / VBT) + vb) * DBG_TILES + t, dbg, l1 + t * FP32_TILE);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(CB_DBG, DBG_TILES);
#endif
    }
}
