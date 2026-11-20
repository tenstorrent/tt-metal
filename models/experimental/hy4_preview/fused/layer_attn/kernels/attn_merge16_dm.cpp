// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The data-movement kernel of merge core i in program P2 at 16 token rows (attn_merge_dm.cpp for two sub-passes of
// 8 rows with a row mask; fused/layer_attn/PLAN.md section 19; attn_rows.hpp). Merge core i serves row i of every
// sub-pass, token rows i and 8 + i, one after the other: for each sub-pass p in which its row is live it runs what
// attn_merge_dm.cpp runs once, with
//   local     byte p of A_LOCAL; the attention core keeps the partial of row 8 p + i at partial + p x 8,704
//   column    bits 8 p + r of A_COLMASK; the partial is at place(mask, p, i) of slot p of the landing on core (5, r)
//   row       the column result goes to place(mask, p, i) of row staging p on the sender core, byte p of C_ROW is
//             counted; bits 8 p + c of A_ROWMASK, slot p of the row landing
//   output    rows 8 p + i of the [16 users, 32] tiles of the post hub's x buffer, 32 bytes at a time; A_CTX + 1
// Before the first chain it tells the merge kernel how many chains it will run (cb_ctl, one word): the merge kernel
// runs exactly as many (hy4r::chains). A row that is not live: nothing is waited for, read, written or counted.
// Every wait on another core is bounded as in attn_merge_dm.cpp, and every chain delivers exactly the partials the
// merge kernel waits for whatever happens.
//
// Compile-time args: n_col, n_row, cb_acc_ms, cb_acc_o, cb_send_ms, cb_send_o, cb_sink, cb_final, cb_in, cb_ctl.
// Runtime args: merge core index, own column, expected column sources (bits), expected row sources, attention core
// (i, own group) NoC x | y << 16, partial address there, column landing address, row landing address, sender core,
// row staging address, post hub core, ctx address there, then the 8 merge cores (NoC x | y << 16).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "attn_rows_dm.hpp"
#ifdef HY4A_FIXED_ORDER
#include "attn_order.hpp"
#endif

constexpr uint32_t n_col = get_compile_time_arg_val(0);
constexpr uint32_t n_row = get_compile_time_arg_val(1);
constexpr uint32_t cb_acc_ms = get_compile_time_arg_val(2);
constexpr uint32_t cb_acc_o = get_compile_time_arg_val(3);
constexpr uint32_t cb_send_ms = get_compile_time_arg_val(4);
constexpr uint32_t cb_send_o = get_compile_time_arg_val(5);
constexpr uint32_t cb_sink = get_compile_time_arg_val(6);
constexpr uint32_t cb_final = get_compile_time_arg_val(7);
constexpr uint32_t cb_in = get_compile_time_arg_val(8);
constexpr uint32_t cb_ctl = get_compile_time_arg_val(9);

namespace {
constexpr uint32_t TINY = 512;  // one [8, 32] BF16 tile
constexpr uint32_t O_TILES = 16;
constexpr uint32_t PART_TILES = 1 + O_TILES;
constexpr uint32_t PART_BYTES = PART_TILES * TINY;        // 8,704
constexpr uint32_t PIECE = 32;                            // one row of a face
constexpr uint32_t GROUP_BYTES = hy4r::SUB * PART_BYTES;  // a landing slot: one source's partials of a sub-pass
constexpr uint32_t FACE16 = hy4r::TOKEN_ROWS * PIECE;     // a face of the post hub's [16, 32] tiles
constexpr uint32_t MERGE_ARGS = 12;
constexpr uint32_t SINK_ARGS = MERGE_ARGS + 8;  // HY4A_SINK_ARGS: the sink peer's max of the chip's 8 heads (BF16 bits)

FORCE_INLINE uint64_t core_addr(uint32_t core, uint32_t address) {
    return get_noc_addr(core & 0xFFFF, core >> 16, address);
}

// Wait for `pages` pages of the merge kernel's output, bounded.
inline bool wait_pages(uint32_t cb, uint32_t pages) {
    for (uint32_t polls = 0; polls < HY4L_WAIT_LIMIT; ++polls) {
        if (cb_pages_available_at_front(cb, pages)) {
            return true;
        }
    }
    return false;
}

// Deliver EXACTLY `count` partials to the merge kernel, in every outcome: the partial at `place` of the landing
// slot at `address` on every landing core whose bit of this core's word `w` appears (the bits of `mask`, which are
// the sources' indices shifted by `shift` = 8 x the sub-pass), until all bits of `mask` have; if the bounded wait runs
// out, what is missing is delivered as the buffer holds it. The merge kernel waits for count * PART_TILES tiles of
// cb_in in this stage and must get them whatever happens (a compute kernel left waiting cannot be recovered without a
// board reset: fused/README.md, fourth batch).
inline void pull_stage(
    uint32_t w, uint32_t mask, uint32_t shift, uint32_t address, uint32_t place, uint32_t code, uint32_t count) {
#ifdef HY4A_FIXED_ORDER
    // HY4A_FIXED_ORDER (PLAN.md section 19.14): the partners in index order whatever order they land in (the stage
    // waits for the lowest one still missing), by attn_order.hpp's loop, which the host test runs: one sequence of
    // merges on every replay of a traced step. The counts delivered and the status are those of the loop below.
    {
        const uint32_t missing = attn_order::stage<true>(
            mask,
            count,
            HY4L_WAIT_LIMIT,
            [&]() {
                invalidate_l1_cache();
                return hy4a::word(w)[0];
            },
            [&](uint32_t r) {
                cb_reserve_back(cb_in, PART_TILES);
                noc_async_read(
                    core_addr(get_arg_val<uint32_t>(MERGE_ARGS + r - shift), address + place * PART_BYTES),
                    get_write_ptr(cb_in),
                    PART_BYTES);
                noc_async_read_barrier();
                cb_push_back(cb_in, PART_TILES);
            },
            [&]() {
                cb_reserve_back(cb_in, PART_TILES);
                cb_push_back(cb_in, PART_TILES);
            });
        if (missing != 0) {
            hy4a::fail(code, missing);
        } else if (mask != 0) {
            hy4a::take_word(w, mask);
        }
        return;
    }
#endif
    uint32_t remaining = mask;
    uint32_t pushed = 0;
    uint32_t polls = 0;
    while (remaining != 0 && pushed < count) {
        invalidate_l1_cache();
        const uint32_t seen = hy4a::word(w)[0] & remaining;
        if (seen != 0) {
            const uint32_t r = __builtin_ctz(seen);
            cb_reserve_back(cb_in, PART_TILES);
            noc_async_read(
                core_addr(get_arg_val<uint32_t>(MERGE_ARGS + r - shift), address + place * PART_BYTES),
                get_write_ptr(cb_in),
                PART_BYTES);
            noc_async_read_barrier();
            cb_push_back(cb_in, PART_TILES);
            remaining &= ~(1u << r);
            ++pushed;
            polls = 0;
        } else if (++polls >= HY4L_WAIT_LIMIT) {
            hy4a::fail(code, remaining);
            break;
        }
    }
    for (; pushed < count; ++pushed) {
        cb_reserve_back(cb_in, PART_TILES);
        cb_push_back(cb_in, PART_TILES);
    }
    if (remaining == 0 && mask != 0) {
        hy4a::take_word(w, mask);
    }
}
}  // namespace

void kernel_main() {
    const uint32_t index = get_arg_val<uint32_t>(0);
    const uint32_t col = get_arg_val<uint32_t>(1);
    const uint32_t col_mask = get_arg_val<uint32_t>(2);
    const uint32_t row_mask = get_arg_val<uint32_t>(3);
    const uint32_t att = get_arg_val<uint32_t>(4);
    const uint32_t partial = get_arg_val<uint32_t>(5);
    const uint32_t col_landing = get_arg_val<uint32_t>(6);
    const uint32_t row_landing = get_arg_val<uint32_t>(7);
    const uint32_t sender = get_arg_val<uint32_t>(8);
    const uint32_t row_staging = get_arg_val<uint32_t>(9);
    const uint32_t post_hub = get_arg_val<uint32_t>(10);
    const uint32_t ctx = get_arg_val<uint32_t>(11);

    hy4l::stamp(0);
    const uint32_t mask = hy4a::row_mask();
    {
        // how many chains the merge kernel is to run: one per sub-pass in which this core's row is live
        cb_reserve_back(cb_ctl, 1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl))[0] = hy4r::chains(mask, index);
        cb_push_back(cb_ctl, 1);
    }
    for (uint32_t p = 0; p < hy4r::SUBPASSES; ++p) {
        if (!hy4r::live(mask, p, index)) {
            continue;
        }
        const uint32_t shift = hy4r::SUB * p;
        const uint32_t place = hy4r::place(mask, p, index);
        const uint32_t row = shift + index;  // the token row
#ifdef HY4A_SINK_ARGS
        {
            // The sink peer, built here from 8 runtime arguments while the attention cores still work: max = the head's
            // sink / scale, sum = 1, O = 0. In an [8, 32] tile's memory the first 128 values are columns 0..15 of the 8
            // rows: row h's columns 0 (max) and 1 (sum) are one 32-bit word, at word 8 h.
            cb_reserve_back(cb_sink, PART_TILES);
            volatile tt_l1_ptr uint32_t* peer = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_sink));
            for (uint32_t i = 0; i < PART_BYTES / 4; ++i) {
                peer[i] = 0;
            }
            for (uint32_t h = 0; h < 8; ++h) {
                peer[8 * h] = (get_arg_val<uint32_t>(SINK_ARGS + h) & 0xFFFFu) | (0x3F80u << 16);
            }
            cb_push_back(cb_sink, PART_TILES);
        }
#endif
        // the chip's own partial of its head group: the first operand of the first merge
        if (hy4a::wait_field(hy4a::A_LOCAL, p, 1, hy4a::ST_LOCAL)) {
            hy4a::take_field(hy4a::A_LOCAL, p, 1);
        }
        [[maybe_unused]] constexpr bool local_to_send = n_col == 0 && n_row > 0;
        constexpr uint32_t local_ms = local_to_send ? cb_send_ms : cb_acc_ms;
        constexpr uint32_t local_o = local_to_send ? cb_send_o : cb_acc_o;
        cb_reserve_back(local_ms, 1);
        cb_reserve_back(local_o, O_TILES);
        noc_async_read(core_addr(att, partial + p * PART_BYTES), get_write_ptr(local_ms), TINY);
        noc_async_read(core_addr(att, partial + p * PART_BYTES + TINY), get_write_ptr(local_o), O_TILES * TINY);
        noc_async_read_barrier();
        cb_push_back(local_ms, 1);
        cb_push_back(local_o, O_TILES);
#ifndef HY4A_SINK_ARGS
        cb_reserve_back(cb_sink, PART_TILES);  // (a host-written tensor lies under the buffer)
        cb_push_back(cb_sink, PART_TILES);
#endif
        hy4l::stamp(1);

        if constexpr (n_col > 0) {
            pull_stage(
                hy4a::A_COLMASK, col_mask << shift, shift, col_landing + p * GROUP_BYTES, place, hy4a::ST_COL, n_col);
        }
        hy4l::stamp(2);

        if constexpr (n_row > 0) {
            // the column result (or the local partial): to the other chips of the row, and kept as the first operand
            // of the row merges; the merge kernel does not touch it before the first row partial is delivered
            if (!wait_pages(cb_send_ms, 1) || !wait_pages(cb_send_o, O_TILES)) {
                hy4a::fail(hy4a::ST_ROW, 0xFFFF);
            }
            const uint32_t slot = row_staging + p * GROUP_BYTES + place * PART_BYTES;
            noc_async_write(get_read_ptr(cb_send_ms), core_addr(sender, slot), TINY);
            noc_async_write(get_read_ptr(cb_send_o), core_addr(sender, slot + TINY), O_TILES * TINY);
            noc_async_write_barrier();
            noc_semaphore_inc(core_addr(sender, hy4a::word_addr(hy4a::C_ROW)), hy4r::one(p));
            noc_async_atomic_barrier();
            pull_stage(
                hy4a::A_ROWMASK, row_mask << shift, shift, row_landing + p * GROUP_BYTES, place, hy4a::ST_ROW, n_row);
        }
        hy4l::stamp(3);

        // the normalised result of the head group: rows = heads; this chip owns heads 2 col and 2 col + 1 of it
        const bool merged = wait_pages(cb_final, O_TILES);
        if (!merged) {
            hy4a::fail(hy4a::ST_CTX, 0xFFFF);
        }
        const uint32_t result = get_read_ptr(cb_final);
        for (uint32_t j = 0; merged && j < 2; ++j) {
            const uint32_t head_row = 2 * col + j;
            for (uint32_t tf = 0; tf < 2 * O_TILES; ++tf) {
                noc_async_write(
                    result + tf * (TINY / 2) + head_row * PIECE,
                    core_addr(post_hub, ctx + j * O_TILES * 2 * FACE16 + tf * FACE16 + row * PIECE),
                    PIECE);
            }
        }
        noc_async_write_barrier();
        if (merged) {
            cb_pop_front(cb_final, O_TILES);
        }
        noc_semaphore_inc(core_addr(post_hub, hy4a::word_addr(hy4a::A_CTX)), 1);
        noc_async_atomic_barrier();
    }
    hy4l::stamp(4);
}
