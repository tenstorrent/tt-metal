// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Activation gather + output scatter (BRISC, NOC1) of the TP=1 decode MoE (nd.py).
//
// mode 1 (gate|up): in0 = the flat BF16 normed hidden (boundary.py: value h at byte 2 h; its 32-value groups
//   are the 1x32 activation tiles, read in one piece) + a tile of `nbias` ones (multiplies the bias tile, K row kt) +
//   a zero tile (K row kt + 1, the bias tile's even-K copy). The routing scores go to the compute MATH thread through
//   the mailbox. Output tile i of a group (column col0 + i) of routed expert e -> row e, bytes 64 (col0 + i) of the
//   [k, I] BF16 row-major activation buffer.
// mode 0 (down): in0 = k segments of kt + 2 tiles: row e of the activation buffer (already score-weighted), a tile
//   holding w_e in its first `nbias` entries (sum_e w_e b_e), a zero tile. Output tile i of a group -> the flat BF16
//   partial sum at byte 64 (col0 + i); columns >= n_out (shard padding) are dropped.
//
// With streams = 2 this kernel also streams the odd K blocks of the weights into cb_w2 on NOC1 (nd_stream.hpp).
//
// runtime args: [in_addr, out_addr, scores_addr, n_groups, e0, w_addr, u_addr, b_addr, idx_addr, vc,
// col0 * n_groups]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "nd_stream.hpp"
#include "ckernel_defs.h"

void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t scores_addr = get_arg_val<uint32_t>(2);
    const uint32_t n_groups = get_arg_val<uint32_t>(3);
    const uint32_t e0 = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(2);
    constexpr uint32_t kt = get_compile_time_arg_val(3);
    constexpr uint32_t nbias = get_compile_time_arg_val(4);
    constexpr uint32_t num_sel = get_compile_time_arg_val(5);
    constexpr uint32_t in_page_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t out_page_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t mode = get_compile_time_arg_val(8);
    constexpr uint32_t g = get_compile_time_arg_val(9);
    constexpr uint32_t n_out = get_compile_time_arg_val(10);
    constexpr auto in_args = TensorAccessorArgs<11>();
    constexpr auto out_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto scores_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    // The weight-stream arguments, as nd_reader.cpp's (tile_bytes .. streams), then the w / u / b / idx accessors.
    constexpr uint32_t st = scores_args.next_compile_time_args_offset();
    constexpr uint32_t cb_w2 = get_compile_time_arg_val(st);
    constexpr uint32_t cb_idx2 = get_compile_time_arg_val(st + 1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(st + 2);
    constexpr uint32_t nt = get_compile_time_arg_val(st + 4);
    constexpr uint32_t nt_b = get_compile_time_arg_val(st + 5);
    constexpr uint32_t kb = get_compile_time_arg_val(st + 6);
    constexpr uint32_t sets = get_compile_time_arg_val(st + 9);
    constexpr uint32_t up_off = get_compile_time_arg_val(st + 10);
    constexpr uint32_t depth = get_compile_time_arg_val(st + 11);
    constexpr uint32_t pkt_tiles = get_compile_time_arg_val(st + 12);
    constexpr uint32_t streams = get_compile_time_arg_val(st + 13);
    constexpr auto w_args = TensorAccessorArgs<st + 14>();
    constexpr auto u_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<u_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr uint32_t col_arg0 = 10;

    constexpr uint32_t tiny = 64;  // one 1x32 BF16 tile
    constexpr uint32_t seg = kt + 2;

    const auto s_in = TensorAccessor(in_args, in_addr, in_page_bytes);
    const auto s_out = TensorAccessor(out_args, out_addr, out_page_bytes);
    const auto s_scores = TensorAccessor(scores_args, scores_addr, 64);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    noc_async_read(s_scores.get_noc_addr(0), scr, 64);

    constexpr uint32_t in0_tiles = mode == 1 ? seg : num_sel * seg;
    cb_reserve_back(cb_in0, in0_tiles);
    const uint32_t in0 = get_write_ptr(cb_in0);
    if constexpr (mode == 1) {
        noc_async_read(s_in.get_noc_addr(0), in0, kt * tiny);
    } else {
        for (uint32_t e = 0; e < num_sel; ++e) {
            noc_async_read(s_in.get_noc_addr(e), in0 + e * seg * tiny, kt * tiny);
        }
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* scores = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(scr);
    for (uint32_t e = 0; e < (mode == 1 ? 1 : num_sel); ++e) {
        volatile tt_l1_ptr uint16_t* aug = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(in0 + (e * seg + kt) * tiny);
        const uint16_t w = mode == 1 ? 0x3F80 : scores[e];  // BF16 1.0 / w_e
        for (uint32_t i = 0; i < 32; ++i) {
            aug[i] = i < nbias ? w : 0;
        }
        for (uint32_t i = 32; i < 64; ++i) {
            aug[i] = 0;
        }
    }
    cb_push_back(cb_in0, in0_tiles);
    if constexpr (mode == 1) {
        for (uint32_t e = 0; e < num_sel; ++e) {
            ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, static_cast<uint32_t>(scores[e0 + e]) << 16);
        }
    }

    if constexpr (streams == 2) {
        const auto s_w = TensorAccessor(w_args, get_arg_val<uint32_t>(5), tile_bytes);
        const auto s_u = TensorAccessor(u_args, get_arg_val<uint32_t>(6), tile_bytes);
        const auto s_b = TensorAccessor(b_args, get_arg_val<uint32_t>(7), tile_bytes);
        const auto s_idx = TensorAccessor(idx_args, get_arg_val<uint32_t>(8), 64);
        cb_reserve_back(cb_idx2, 1);
        const uint32_t idx_l1 = get_write_ptr(cb_idx2);
        noc_async_read(s_idx.get_noc_addr(0), idx_l1, 64);
        noc_async_read_barrier();
        volatile tt_l1_ptr uint16_t* ids = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(idx_l1);
        uint32_t experts[num_sel];
        for (uint32_t e = 0; e < num_sel; ++e) {
            experts[e] = ids[e0 + e];
        }
        gpt_oss_nd::nd_stream_slots<tile_bytes, kt, nt, nt_b, kb, g, num_sel, sets, up_off, depth, pkt_tiles, 2>(
            cb_w2, s_w, s_u, s_b, experts, n_groups, col_arg0, 1, get_arg_val<uint32_t>(9));
    }

    for (uint32_t gi = 0; gi < n_groups; ++gi) {
        const uint32_t col0 = get_arg_val<uint32_t>(col_arg0 + gi);
        for (uint32_t e = 0; e < (mode == 1 ? num_sel : 1); ++e) {
            cb_wait_front(cb_out, g);
            uint32_t src = get_read_ptr(cb_out);
            for (uint32_t i = 0; i < g; ++i) {
                const uint32_t n = col0 + i;
                if (n >= n_out) {
                    break;
                }
                const uint64_t dst = mode == 1
                                         ? s_out.get_noc_addr(e0 + e, n * tiny)
                                         : s_out.get_noc_addr(n * tiny / out_page_bytes, (n * tiny) % out_page_bytes);
                noc_async_write(src, dst, tiny);
                src += tiny;
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_out, g);
        }
    }
    noc_async_write_barrier();
}
