// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Cross-chip log-sum-exp merge (fused/merges/lse_merge.py): the data-movement kernel of merge core u (one per
// user) of the chip in row r, column c of the merged rectangle. The chip's row owns head group r.
//
//   local     the chip's own partial (u, r) -> the running partial (cb_acc; cb_send when there is no column stage)
//   column    wait for the partials (u, r) of the other chips of the column (lse_sender.cpp of each of them, one
//             packet into this core's column landing) and give the landing to the merge kernel
//   row       hand the column-merged partial to the collective's sender (all-gather over the chips of the row),
//             wait for the row landing, give it to the merge kernel
//   sink      one more peer (max = sink / scale, sum = 1, O = 0), resident in L1, merged last with the division
//   output    of the normalised result [8 heads, 512] the rows of the heads this chip owns go to the chip's ctx
//             tiles ([8 users, 32] per head and latent tile column: row u of each), 32 bytes at a time
// Every wait of this kernel on another core is bounded (collective.hpp); a wait that runs out is flagged in the
// state tile and the pages are published all the same, so the merge kernel and the program end.
//
// Compile-time args: column members, row members, cb_acc_ms, cb_acc_o, cb_send_ms, cb_send_o, cb_sink,
// cb_final, cb_col0, cb_col1 (the column landing's parities), cb_row0, cb_row1, then TensorAccessorArgs of the
// attention output.
// Runtime args: user, row slot r, column slot c, attention output address, ctx core NoC x, y, ctx address,
// state address, column Use (10 words), row Use (10 words).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "collective.hpp"
#include "lse_common.hpp"

using namespace hy4_lse;

constexpr uint32_t col_members = get_compile_time_arg_val(0);
constexpr uint32_t row_members = get_compile_time_arg_val(1);
constexpr uint32_t cb_acc_ms = get_compile_time_arg_val(2);
constexpr uint32_t cb_acc_o = get_compile_time_arg_val(3);
constexpr uint32_t cb_send_ms = get_compile_time_arg_val(4);
constexpr uint32_t cb_send_o = get_compile_time_arg_val(5);
constexpr uint32_t cb_sink = get_compile_time_arg_val(6);
constexpr uint32_t cb_final = get_compile_time_arg_val(7);
constexpr uint32_t cb_col[2] = {get_compile_time_arg_val(8), get_compile_time_arg_val(9)};
constexpr uint32_t cb_row[2] = {get_compile_time_arg_val(10), get_compile_time_arg_val(11)};
constexpr uint32_t heads_out = 8 / row_members;
static_assert(row_members >= 1 && 8 % row_members == 0 && col_members >= 1);

namespace {
FORCE_INLINE void publish(uint32_t cb, uint32_t pages) {
    cb_reserve_back(cb, pages);
    cb_push_back(cb, pages);
}

// Wait for `pages` pages of the merge kernel's output, bounded like the collective's waits.
inline bool wait_pages(uint32_t cb, uint32_t pages) {
#if HY4_COLLECTIVE_WAIT_LIMIT != 0
    for (uint32_t polls = 0; polls < HY4_COLLECTIVE_WAIT_LIMIT; ++polls) {
        if (cb_pages_available_at_front(cb, pages)) {
            return true;
        }
    }
    return false;
#else
    cb_wait_front(cb, pages);
    return true;
#endif
}
}  // namespace

void kernel_main() {
#ifdef HY4_MERGE_NOP
    return;
#else
    constexpr auto out_args = TensorAccessorArgs<12>();
    size_t arg = 0;
    const uint32_t user = get_arg_val<uint32_t>(arg++);
    const uint32_t row = get_arg_val<uint32_t>(arg++);
    const uint32_t col = get_arg_val<uint32_t>(arg++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg++);
    const uint32_t ctx_x = get_arg_val<uint32_t>(arg++);
    const uint32_t ctx_y = get_arg_val<uint32_t>(arg++);
    const uint32_t ctx = get_arg_val<uint32_t>(arg++);
    const uint32_t state = get_arg_val<uint32_t>(arg++);
    const hy4_collective::Use col_use = hy4_collective::read_use(arg);
    const hy4_collective::Use row_use = hy4_collective::read_use(arg);

    volatile tt_l1_ptr uint32_t* st = hy4_collective::word(state);
    uint32_t status = 0;
    st[ST_STATUS] = 0;
    st[ST_STAMPS + 0] = hy4_collective::now();

    // The chip's own partial of its head group: the first operand of the first merge.
    WAYPOINT("LSRD");
    const auto out = TensorAccessor(out_args, out_addr);
    const bool local_to_send = col_members == 1 && row_members > 1;
    const uint32_t local_ms = local_to_send ? cb_send_ms : cb_acc_ms;
    const uint32_t local_o = local_to_send ? cb_send_o : cb_acc_o;
    cb_reserve_back(local_ms, 1);
    cb_reserve_back(local_o, O_TILES);
    const uint32_t page = user * 8 + row;
    noc_async_read(out.get_noc_addr(page, O_BYTES), get_write_ptr(local_ms), TINY_BYTES);
    noc_async_read(out.get_noc_addr(page, 0), get_write_ptr(local_o), O_BYTES);
    noc_async_read_barrier();
    cb_push_back(local_ms, 1);
    cb_push_back(local_o, O_TILES);
    publish(cb_sink, PART_TILES);
    st[ST_STAMPS + 1] = hy4_collective::now();

    if constexpr (col_members > 1) {
        WAYPOINT("LSCW");
        uint32_t parity = 0;
        if (!hy4_collective::wait_landed(col_use, parity)) {
            status |= 1;
        }
        st[ST_STAMPS + 2] = hy4_collective::now();
        publish(cb_col[parity], col_members * PART_TILES);
    }

    if constexpr (row_members > 1) {
        // The column-merged partial (or the local one): to the other chips of the row, and kept as the first
        // operand of the row merges. The merge kernel does not touch it before the row landing is published.
        WAYPOINT("LSRH");
        if (!wait_pages(cb_send_ms, 1) || !wait_pages(cb_send_o, O_TILES)) {
            status |= 4;  // the merge kernel did not deliver: what is handed over is not the column result
        }
        st[ST_STAMPS + 3] = hy4_collective::now();
        const uint32_t sx = row_use.sender_xy & 0xFFFF;
        const uint32_t sy = row_use.sender_xy >> 16;
        noc_async_write(get_read_ptr(cb_send_ms), get_noc_addr(sx, sy, row_use.staging), TINY_BYTES);
        noc_async_write(get_read_ptr(cb_send_o), get_noc_addr(sx, sy, row_use.staging + TINY_BYTES), O_BYTES);
        noc_async_write_barrier();
        // collective.hpp hand_over, for a payload in two pieces: the op's ready bit of this round's parity
        noc_semaphore_inc(
            get_noc_addr(sx, sy, get_semaphore(row_use.ready_id)),
            1u << (2 * row_use.op + hy4_collective::parity(row_use)));
        noc_async_atomic_barrier();
        st[ST_STAMPS + 4] = hy4_collective::now();
        uint32_t parity = 0;
        if (!hy4_collective::wait_landed(row_use, parity)) {
            status |= 2;
        }
        st[ST_STAMPS + 5] = hy4_collective::now();
        publish(cb_row[parity], row_members * PART_TILES);
    }

    // The normalised result of the head group: rows = heads. This chip owns `heads_out` of them.
    WAYPOINT("LSFW");
    const bool merged = wait_pages(cb_final, O_TILES);
    if (!merged) {
        status |= 8;  // the merge kernel did not deliver: no ctx is written
    }
    st[ST_STAMPS + 6] = hy4_collective::now();
    const uint32_t result = get_read_ptr(cb_final);
    for (uint32_t j = 0; merged && j < heads_out; ++j) {
        const uint32_t head_row = col * heads_out + j;
        for (uint32_t t = 0; t < O_TILES; ++t) {
            for (uint32_t face = 0; face < 2; ++face) {
                noc_async_write(
                    result + t * TINY_BYTES + face * FACE_BYTES + head_row * ROW_BYTES,
                    get_noc_addr(
                        ctx_x, ctx_y, ctx + (j * O_TILES + t) * TINY_BYTES + face * FACE_BYTES + user * ROW_BYTES),
                    ROW_BYTES);
            }
        }
    }
    noc_async_write_barrier();
    if (merged) {
        cb_pop_front(cb_final, O_TILES);
    }
    if constexpr (col_members > 1) {
        hy4_collective::finish(col_use);
    }
    if constexpr (row_members > 1) {
        hy4_collective::finish(row_use);
    }
    st[ST_STAMPS + 7] = hy4_collective::now();
    st[ST_STATUS] = status | 0x100;
    WAYPOINT("LSDN");
#endif
}
