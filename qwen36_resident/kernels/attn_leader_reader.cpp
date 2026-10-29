// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attention leader reader: reads every user's rope tables from the token state; per attention layer a,
// reads the layer's q norm weight (copy a % copies) and waits for the streamers' projection rows
// [q heads | gates | k | v] (batch x 32 tiles); per user u places head h of u's q and gate into row h of
// cb_qraw / cb_gate, then releases the group heads' partial sums to compute once all of them have
// written theirs. User u + 1's rows are placed before user u's partials are awaited.
//
// Runtime args: 0 row buffer address, 1 children (group heads of the reduction tree), 2 optional
// timeline buffer (0 = off; per layer 8 wall-clock words: [1] rows arrived, [2] rows placed,
// [3] partials arrived), 3 token state address, 4 rope offset in it, 5 q norm weight address (DRAM,
// [copies][Dt] bf16 tiles).

#include "api/dataflow/dataflow_api.h"
#include "attn_common.hpp"

using namespace resident_attn;

namespace {

// copy user u's row of a batch x 32 tile into row `row` of a 32x32 bf16 tile (two 16-element face rows)
FORCE_INLINE void place_row(uint32_t src, uint32_t u, uint32_t tile, uint32_t row) {
    const uint32_t off = row_offset(row);
    src += user_offset(u);
    noc_async_read(get_noc_addr(src), tile + off, kFaceRow);
    noc_async_read(get_noc_addr(src + batch * kFaceRow), tile + kFace + off, kFaceRow);
}

FORCE_INLINE void zero_tiles(uint32_t cb, uint32_t n) {
    auto& iface = get_local_cb_interface(cb);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(iface.fifo_limit - iface.fifo_size);
    for (uint32_t i = 0; i < n * kTile / 4; i++) {
        p[i] = 0;
    }
}

FORCE_INLINE void push_resident(uint32_t cb, uint32_t n) {
    cb_reserve_back(cb, n);
    cb_push_back(cb, n);
}

}  // namespace

void kernel_main() {
    const uint32_t rows = get_arg_val<uint32_t>(0);
    const uint32_t children = get_arg_val<uint32_t>(1);
    const uint32_t ts_addr = get_arg_val<uint32_t>(2);
    const uint32_t tok_addr = get_arg_val<uint32_t>(3);
    const uint32_t rope_off = get_arg_val<uint32_t>(4);
    const InterleavedAddrGenFast<true> qw_dram{
        .bank_base_address = get_arg_val<uint32_t>(5), .page_size = kTile, .data_format = DataFormat::Float16_b};
    volatile tt_l1_ptr uint32_t* rows_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_rows));
    volatile tt_l1_ptr uint32_t* part_sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_part));
    volatile tt_l1_ptr uint32_t* ts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ts_addr);
    volatile uint32_t* clk = reinterpret_cast<volatile uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    auto mark = [&](uint32_t l, uint32_t i) {
        if (ts_addr) {
            ts[l * 8 + i] = *clk;
        }
    };

    zero_tiles(cb_qraw, 2 * Dt);
    zero_tiles(cb_gate, 2 * Dt);
    push_resident(cb_mean, 1);
    {
        const InterleavedAddrGen<true> tok{.bank_base_address = tok_addr, .page_size = rope_off + 3 * batch * kTile};
        cb_reserve_back(cb_rope, 3 * batch);
        noc_async_read(tok.get_noc_addr(0, rope_off), get_write_ptr(cb_rope), 3 * batch * kTile);
        noc_async_read_barrier();
        cb_push_back(cb_rope, 3 * batch);
    }

    constexpr uint32_t gate0 = heads * Dt;
    auto place = [&](uint32_t u) {
        cb_reserve_back(cb_qraw, Dt);
        cb_reserve_back(cb_gate, Dt);
        const uint32_t q = get_write_ptr(cb_qraw), g = get_write_ptr(cb_gate);
        for (uint32_t h = 0; h < heads; h++) {
            for (uint32_t d = 0; d < Dt; d++) {
                place_row(rows + (h * Dt + d) * kRowTile, u, q + d * kTile, h);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_qraw, Dt);
        for (uint32_t h = 0; h < heads; h++) {
            for (uint32_t d = 0; d < Dt; d++) {
                place_row(rows + (gate0 + h * Dt + d) * kRowTile, u, g + d * kTile, h);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_gate, Dt);
    };
    for (uint32_t i = 0; i < iters; i++) {
        const uint32_t l = i / batch, u = i % batch;
        if (u == 0) {
            cb_reserve_back(cb_qw, Dt);
            for (uint32_t d = 0; d < Dt; d++) {
                noc_async_read_page((l % copies) * Dt + d, qw_dram, get_write_ptr(cb_qw) + d * kTile);
            }
            noc_async_read_barrier();
            cb_push_back(cb_qw, Dt);
            noc_semaphore_wait_min(rows_sem, num_streamers * (l + 1));
            mark(l, 1);
            place(0);
            mark(l, 2);
        }
        // the next user's rows go in before this user's partials arrive, so compute prepares its q meanwhile
        if (u + 1 < batch) {
            place(u + 1);
        }

        noc_semaphore_wait_min(part_sem, children * (i + 1));
        if (u + 1 == batch) {
            mark(l, 3);
        }
        push_resident(cb_parts, fanin * kPart);
    }
}
