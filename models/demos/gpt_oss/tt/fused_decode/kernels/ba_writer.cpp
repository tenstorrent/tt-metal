// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Block attention, writer (BRISC) (dflash/block_attention.py): tells compute its slice (one mailbox word), then
//   root (slice 0): writes its Q tile's 2 packed output tiles (tile t, d at page (g Tt + t) 2 + d of the packed
//   output);
//   other active slices: writes the partial (cb_send) into slot send_round of the parent's cb_recv and sets bit
//   send_round of the parent's tree semaphore.
//
// common runtime args: [out_addr, core index table (BA_NIDX words), noc (x << 8 | y) per core (2 per word),
// (BA_OUT_NOTIFY_N: the consumer cores' noc (x << 8 | y), 2 per word), (BA_USER_ARG_W: the user)]
// BA_OUT_ROWS (batch-32 verify): the root writes its packed tile's 4 rows into the rows layout [1, rows, heads, 64]
// instead (block row r of user u = out row r BA_Q_RSTRIDE + u, kv head g's 8 heads = tile rows 8 g .. 8 g + 7).
// BA_OUT_CONCAT: into the concatenated-heads layout [1, 1, rows, heads * 64] (BA_OUT_CONCAT column tiles) instead.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ba_common.hpp"
#include "ckernel.h"
#include "ckernel_defs.h"

namespace {
constexpr uint32_t rows8(uint32_t i, uint32_t f) { return ((i >> 1) * 2 + f) * 512 + (i & 1) * 256; }
}  // namespace

void kernel_main() {
    const uint32_t out_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t ci = ba_core_index(1);
    const uint32_t grp = ci / get_compile_time_arg_val(5), s = ci % get_compile_time_arg_val(5);
    const uint32_t g = (grp / BA_TQ) % BA_NKV;

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(1);
    constexpr uint32_t cb_send = get_compile_time_arg_val(2);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(3);
    constexpr uint32_t M = get_compile_time_arg_val(4);
    constexpr uint32_t S = get_compile_time_arg_val(5);
    constexpr uint32_t min_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t partial_tiles = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<8>();
    constexpr uint32_t tile = 2048;
    constexpr uint32_t Tt = M / 4;
    const uint32_t t = grp % BA_TQ;  // this core's packed Q tile
    const auto s_out = TensorAccessor(out_args, out_addr, tile);

    // the positions, from this core's reader (cb_scr: BA_CB_POS)
    cb_wait_front(cb_scr, 1);
    invalidate_l1_cache();
    volatile tt_l1_ptr int32_t* pos = reinterpret_cast<volatile tt_l1_ptr int32_t*>(get_read_ptr(cb_scr));
#ifdef BA_UNIFORM
    const uint32_t hi = static_cast<uint32_t>(pos[M - 1]);
#ifdef BA_WINDOW
    const uint32_t lo = hi + 1 >= BA_WINDOW ? hi + 1 - BA_WINDOW : 0;
#else
    const uint32_t lo = 0;
#endif
    const BaSlice sl = ba_slice(hi, hi, s, S, min_chunks, lo / 32);
    constexpr bool has_sink = false;
    // bit 13 (no sink here): chunk 0 of this slice is the window's first chunk with a cut start (mask e = 1)
    const bool win_first = sl.c_lo == lo / 32 && (lo % 32) && lo / 32 != hi / 32;  // (the window's first slice)
#elif defined(BA_SWIN)
    const uint32_t w_lo = static_cast<uint32_t>(pos[0]) + 1 >= BA_SWIN ? pos[0] + 1 - BA_SWIN : 0;
    const BaSlice sl =
        ba_slice(static_cast<uint32_t>(pos[0]), static_cast<uint32_t>(pos[M - 1]), s, S, min_chunks, w_lo / 32);
    constexpr bool has_sink = true;
    constexpr bool win_first = false;
#else
    const BaSlice sl = ba_slice(static_cast<uint32_t>(pos[0]), static_cast<uint32_t>(pos[M - 1]), s, S, min_chunks);
    constexpr bool has_sink = true;
    constexpr bool win_first = false;
#endif
    const uint32_t nc = sl.c_hi - sl.c_lo;
#ifdef BA_SWIN
    // the window's history is short: bits 18..24 the causal mask offset + 1 (as below, clamped to [-1, 125]), 25..31
    // the window mask offset + 1 (w_lo / 32 - c_lo: chunk j takes window mask e = j - offset when 0 <= e <= 1; 127:
    // none, the slice starts past the window's first 2 chunks)
    static_assert(BA_SWIN / 32 + 2 < 126, "the window's chunks fit the 7-bit offsets");
    int32_t off = static_cast<int32_t>(sl.c_block) - static_cast<int32_t>(sl.c_lo);
    off = off < -1 ? -1 : (off > 125 ? 125 : off);
    const int32_t woff = static_cast<int32_t>(w_lo / 32) - static_cast<int32_t>(sl.c_lo);
    const uint32_t woff_bits = woff < -1 ? 127u : static_cast<uint32_t>(woff + 1);
    const uint32_t info = nc | (s == 0 ? 1u << 13 : 0u) | (sl.send_round < 0 ? 1u << 14 : 0u) | (sl.recv_rounds << 15) |
                          (static_cast<uint32_t>(off + 1) << 18) | (woff_bits << 25);
#else
    // one word: bits 0..12 chunks, 13 sink, 14 root, 15..17 receive rounds, 18..31 mask offset + 1 (c_block - c_lo
    // clamped to [-1, 8190]: chunk j of the slice takes causal mask e = j - offset when 0 <= e <= 1)
    int32_t off = static_cast<int32_t>(sl.c_block) - static_cast<int32_t>(sl.c_lo);
    off = off < -1 ? -1 : (off > 8190 ? 8190 : off);
    const uint32_t info = nc | ((has_sink && s == 0) || win_first ? 1u << 13 : 0u) |
                          (sl.send_round < 0 ? 1u << 14 : 0u) | (sl.recv_rounds << 15) |
                          (static_cast<uint32_t>(off + 1) << 18);
#endif
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, info);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, info);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, info);
    if (nc == 0) {
        return;
    }
    if (sl.send_round < 0) {
        cb_wait_front(cb_out, 2);
        const uint32_t src = get_read_ptr(cb_out);
#ifdef BA_OUT_CONCAT
        // the concatenated-heads layout [1, 1, rows, heads * 64] (o_proj's input: row tile rt's column tile c at page
        // rt BA_OUT_CONCAT + c): packed row 8 i + j (block row 4 t + i, head 8 g + j) -> row R = (4 t + i) BA_Q_RSTRIDE
        // + user, column tiles 2 (8 g + j) + d, one 32-byte face row per (i, j, d, f)
        const uint32_t user = get_common_arg_val<uint32_t>(BA_USER_ARG_W) + grp / (BA_TQ * BA_NKV);
        for (uint32_t d = 0; d < 2; ++d) {
            for (uint32_t i = 0; i < 4; ++i) {
                const uint32_t R = (4 * t + i) * BA_Q_RSTRIDE + user, rr = R % 32;
                for (uint32_t j = 0; j < 8; ++j) {
                    const uint64_t dst = s_out.get_noc_addr((R / 32) * BA_OUT_CONCAT + 2 * (8 * g + j) + d);
                    const uint32_t pr = 8 * i + j;
                    for (uint32_t f = 0; f < 2; ++f) {
                        noc_async_write(
                            src + d * tile + ((pr / 16) * 2 + f) * 512 + (pr % 16) * 32,
                            dst + ((rr / 16) * 2 + f) * 512 + (rr % 16) * 32,
                            32);
                    }
                }
            }
        }
#elif defined(BA_OUT_ROWS)
        const uint32_t user = get_common_arg_val<uint32_t>(BA_USER_ARG_W) + grp / (BA_TQ * BA_NKV);
        for (uint32_t d = 0; d < 2; ++d) {
            for (uint32_t i = 0; i < 4; ++i) {
                const uint64_t dst = s_out.get_noc_addr(2 * ((4 * t + i) * BA_Q_RSTRIDE + user) + d);
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_write(src + d * tile + rows8(i, f), dst + rows8(g, f), 256);
                }
            }
        }
#else
        for (uint32_t d = 0; d < 2; ++d) {
            noc_async_write(src + d * tile, s_out.get_noc_addr((g * Tt + t) * 2 + d), tile);
        }
#endif
        noc_async_write_barrier();
#ifdef BA_OUT_NOTIFY_N
        // a consumer in the same program (the drafter's o_proj, dflash/fused_draft.py): increment its semaphore
        // BA_OUT_NOTIFY_SEM on each of its cores (common args BA_OUT_NOTIFY_ARG..: noc (x << 8 | y), 2 per word)
        for (uint32_t i = 0; i < BA_OUT_NOTIFY_N; ++i) {
            const uint32_t w = get_common_arg_val<uint32_t>(BA_OUT_NOTIFY_ARG + i / 2) >> (16 * (i % 2));
            noc_semaphore_inc(get_noc_addr((w >> 8) & 0xFF, w & 0xFF, get_semaphore(BA_OUT_NOTIFY_SEM)), 1);
        }
        noc_async_atomic_barrier();
#endif
        cb_pop_front(cb_out, 2);
    } else {
        constexpr uint32_t fp32_tile = 4096;
        const uint32_t r = static_cast<uint32_t>(sl.send_round);
        // the parent s - 2^r (r = the lowest set bit of s: fixed per slice)
        const uint32_t parent = grp * S + (s & (s - 1));
        const uint32_t nw = get_common_arg_val<uint32_t>(1 + BA_NIDX + parent / 2) >> (16 * (parent % 2));
        const uint32_t px = (nw >> 8) & 0xFF, py = nw & 0xFF;
        const uint32_t slot = r;
        const uint32_t dst = get_write_ptr(cb_recv) + slot * partial_tiles * fp32_tile;  // (no push on this RISC)
        cb_wait_front(cb_send, partial_tiles);
        noc_async_write(get_read_ptr(cb_send), get_noc_addr(px, py, dst), partial_tiles * fp32_tile);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(BA_TREE_SEM)), 1u << r);
        noc_async_atomic_barrier();
        cb_pop_front(cb_send, partial_tiles);
    }
}
