// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// The index query's assembly in program P1 of a full-indexer layer (fused/layer_attn/program.py; PLAN.md section
// 21.3), on the 4 idx cores, the same source on both data-movement processors of each (compile-time arg 0: 0 on
// processor 0 / NoC 0, 1 on processor 1 / NoC 1): ONE token row per processor. No circular buffer, no compute kernel.
//
// What lands on the FIRST idx core with Xa: every chip h's 4 row tiles of index head h's query for the pass's token
// rows (2,048 B at 8 token rows, at landing + h * that), each chip's arrival counted in that core's word A_IDX; chip
// 22's pool tile, the head weights [token rows x 32 heads], counted in its word A_IDXW.
//   the master (processor 0 of the first idx core): waits (bounded) for the arrivals, takes the counts off, releases
//                the other 7 processors: one atomic per idx core into its word A_IDXGO (1 on its own core, 2 on each
//                other); a released processor takes 1 off by an atomic of its own.
//   every processor, for its token row u: the row's 256 pieces of 32 B (chip h, tile t, half f, row u -> tile t,
//                face 2 (h / 16) + f, row h % 16: attn_idx_rows.hpp, the text the host test compiles) by one-packet
//                NoC reads from the landing straight into the row's 4 tiles in this core's arena (a word copied by
//                the RISC costs 13 ns here: 110 us for 4 rows, measured; the NoC moves a piece at once), and the 2
//                pieces of the head-weight row; read barrier; the 4 tiles to user u's 4 pages of the scan's query
//                tensor (DRAM, interleaved: page 4 u + t); the head-weight row (the 32 BF16 multipliers) to bytes
//                64..127 of user u's page of the scan's step tensor (its words 16..31; the other bytes of the page
//                are the host's). The writes are drained before the kernel ends: the scan is this chip's next
//                program and reads both tensors from DRAM at its start.
// A wait that runs out leaves its code in its core's status word and the kernel carries on with what the landing
// holds (the scan then runs on a query that is partly the step before's: wrong ids, nothing waits for ever). A
// released processor's bound is three of the master's, which makes two bounded waits before it releases.
//
// Compile-time args: processor (0 | 1), token rows of a row tile (8).
// Runtime args (per core): its token row on processor 0 (processor 1: that + 1), 1 on the first idx core, chips
// whose index-query tiles to wait for, head-weight tiles to wait for (0 | 1), the first idx core's NoC x | y << 16,
// the landing's address there, the head-weight tile's, this core's two rows' query buffers (8,192 B each), their
// multiplier words (64 B each), the scan's query tensor's address, its step tensor's, then the 4 idx cores' NoC
// x | y << 16.
// Stamps (HY4L_TIMING), per core: processor 0: 0 start, 1 released (the master: arrivals complete), 2 its row
// written (drained); processor 1: 3 released, 4 its row written (drained).

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"

#include "attn_common.hpp"
#include "attn_idx_rows.hpp"

constexpr uint32_t part = get_compile_time_arg_val(0);
constexpr uint32_t rows = get_compile_time_arg_val(1);

namespace {
constexpr uint32_t A_IDX = 0;     // chips whose index-query tiles are in the landing (Xa; counted by the fabric)
constexpr uint32_t A_IDXW = 2;    // the head-weight tile has landed (Xa)
constexpr uint32_t A_IDXGO = 10;  // the master -> the processors of an idx core: the landing is complete
constexpr uint32_t ST_IDX = 0xDFu << 24;    // index-query tiles missing (low bits: the arrivals seen)
constexpr uint32_t ST_IDXW = 0xE0u << 24;   // the head-weight tile missing
constexpr uint32_t ST_IDXGO = 0xE1u << 24;  // a processor was not released
constexpr uint32_t IDX_CORES = 4;
constexpr uint32_t PIECE = 4 * idx_rows::PIECE_WORDS;  // 32 B
constexpr uint32_t TILE16 = 2048;                      // a standard BF16 tile
constexpr uint32_t STEP_PAGE = 128;                    // a user's page of the scan's step tensor
constexpr uint32_t MULT_BYTES = 2 * PIECE;             // its words 16..31

FORCE_INLINE uint64_t dram_page(uint32_t base, uint32_t page, uint32_t page_bytes, uint32_t offset) {
    return get_noc_addr_from_bank_id<true>(page % NUM_DRAM_BANKS, base + (page / NUM_DRAM_BANKS) * page_bytes + offset);
}
}  // namespace

void kernel_main() {
    const uint32_t u = get_arg_val<uint32_t>(0) + part;
    const bool master = part == 0 && get_arg_val<uint32_t>(1) != 0;
    const uint32_t n_idx = get_arg_val<uint32_t>(2);
    const uint32_t n_w = get_arg_val<uint32_t>(3);
    const uint32_t land_core = get_arg_val<uint32_t>(4);
    const uint32_t landing = get_arg_val<uint32_t>(5);
    const uint32_t weights = get_arg_val<uint32_t>(6);
    const uint32_t query = get_arg_val<uint32_t>(7) + part * 4 * idx_rows::QUERY_WORDS;
    const uint32_t words = get_arg_val<uint32_t>(8) + part * MULT_BYTES;
    const uint32_t qt_addr = get_arg_val<uint32_t>(9);
    const uint32_t step_addr = get_arg_val<uint32_t>(10);

    if constexpr (part == 0) {
        hy4l::stamp(0);
    }
    if (master) {
        if (hy4a::wait_word(A_IDX, n_idx, ST_IDX)) {
            hy4a::take_word(A_IDX, n_idx);
        }
        if (n_w != 0 && hy4a::wait_word(A_IDXW, n_w, ST_IDXW)) {
            hy4a::take_word(A_IDXW, n_w);
        }
        for (uint32_t k = 0; k < IDX_CORES; ++k) {
            const uint32_t core = get_arg_val<uint32_t>(11 + k);
            const uint32_t n = core == land_core ? 1u : 2u;  // (this core's processor 0 is the master itself)
            noc_semaphore_inc(get_noc_addr(core & 0xFFFF, core >> 16, hy4a::word_addr(A_IDXGO)), n);
        }
        noc_async_atomic_barrier();
    } else {
        volatile tt_l1_ptr uint32_t* go = hy4a::word(A_IDXGO);
        uint32_t seen = 0;
        for (uint32_t round = 0; round < 3 && seen == 0; ++round) {
            for (uint32_t polls = 0; polls < HY4L_WAIT_LIMIT && seen == 0; ++polls) {
                invalidate_l1_cache();
                seen = go[0];
            }
        }
        if (seen != 0) {
            hy4a::take_word(A_IDXGO, 1);
        } else {
            hy4a::fail(ST_IDXGO, 0);
        }
    }
    hy4l::stamp(part == 0 ? 1 : 3);

    const uint32_t lx = land_core & 0xFFFF;
    const uint32_t ly = land_core >> 16;
    idx_rows::pieces(rows, u, [&](uint32_t a, uint32_t b) {
        noc_async_read_one_packet(get_noc_addr(lx, ly, landing + 4 * a), query + 4 * b, PIECE);
    });
    for (uint32_t f = 0; f < 2; ++f) {
        noc_async_read_one_packet(
            get_noc_addr(lx, ly, weights + 4 * idx_rows::weight_word(rows, f, u)), words + f * PIECE, PIECE);
    }
    noc_async_read_barrier();
    for (uint32_t t = 0; t < idx_rows::TILES; ++t) {
        noc_async_write(query + t * TILE16, dram_page(qt_addr, u * idx_rows::TILES + t, TILE16, 0), TILE16);
    }
    noc_async_write(words, dram_page(step_addr, u, STEP_PAGE, 4 * idx_rows::STEP_MULTIPLIER_WORD), MULT_BYTES);
    noc_async_write_barrier();
    hy4l::stamp(part == 0 ? 2 : 4);
}
