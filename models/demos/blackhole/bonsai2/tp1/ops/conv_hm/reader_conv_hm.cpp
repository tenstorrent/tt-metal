// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader for conv_hm (fused head-major GDN decode conv). One core per HALF state tile: 16 heads x 32 channels,
// i.e. faces 0|1 (heads 32*r_t + 0..15) or faces 2|3 (heads 32*r_t + 16..31) of tile (r_t, c_t). A face pair is a
// contiguous 1 KB half of a bf16 tile, so two cores own disjoint bytes of one tile and the in-place state update
// cannot race. Every row of a half is a real head (80 heads = 5 halves), so nothing needs zeroing.
//
//   new row : head h's 32 channels of chunk c_t are row 0 of qkv tile h*4 + c_t (row 0 of faces 0 and 1, i.e. bytes
//             0..31 and 512..543): ONE 1 KB read per head into a scratch slot, then two 32 B runs copied into
//             row (h % 16) of this half's faces.
//   cb_cst  : this half of state rows 1..K-1, cb_tap: this half of the K taps (1 KB each, at its offset in the page)
//   cb_wst  : this half of state rows 1..K-1 and the new row again, for the writer (one consumer per CB)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_new = 0, cb_cst = 1, cb_tap = 2, cb_out = 3, cb_wst = 4, cb_scr = 5;

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);  // conv taps (4)
    constexpr auto qkv_a = TensorAccessorArgs<1>();
    constexpr auto st_a = TensorAccessorArgs<qkv_a.next_compile_time_args_offset()>();   // the K state tensors share it
    constexpr auto tap_a = TensorAccessorArgs<st_a.next_compile_time_args_offset()>();   // the K tap tensors share it

    // runtime: 0 qkv, 1..K state addrs, K+1..2K tap addrs, 2K+1 r_t, 2K+2 c_t, 2K+3 half (0: faces 0|1, 1: faces 2|3)
    const uint32_t qkv_addr = get_arg_val<uint32_t>(0);
    const uint32_t r_t = get_arg_val<uint32_t>(2 * K + 1);
    const uint32_t c_t = get_arg_val<uint32_t>(2 * K + 2);
    const uint32_t half = get_arg_val<uint32_t>(2 * K + 3);
    const uint32_t tile = r_t * 4 + c_t;  // page of this core's tile in every [1, 80, 128] tensor
    const uint32_t hoff = half * 1024;    // byte offset of this core's half in a bf16 tile
    const uint32_t h0 = 32 * r_t + 16 * half;

    const uint32_t tb = get_tile_size(cb_new);  // bf16 2048 B
    const auto qkv = TensorAccessor(qkv_a, qkv_addr, tb);
    Noc noc;

    CircularBuffer cn(cb_new), cc(cb_cst), ct(cb_tap), cw(cb_wst), scr(cb_scr);
    cn.reserve_back(1);
    cc.reserve_back(K - 1);
    ct.reserve_back(K);
    cw.reserve_back(K);
    for (uint32_t i = 0; i < 16; i++) {
        noc.async_read(qkv, scr, 1024, {.page_id = (h0 + i) * 4 + c_t, .offset_bytes = 0}, {.offset_bytes = i * 1024});
    }
    auto read_half = [&](uint32_t cb_id, uint32_t page, uint32_t addr, const auto& args) {
        CircularBuffer cb(cb_id);
        const auto acc = TensorAccessor(args, addr, tb);
        noc.async_read(acc, cb, 1024, {.page_id = tile, .offset_bytes = hoff}, {.offset_bytes = page * tb + hoff});
    };
    for (uint32_t j = 1; j < K; j++) {
        read_half(cb_cst, j - 1, get_arg_val<uint32_t>(1 + j), st_a);
        read_half(cb_wst, j - 1, get_arg_val<uint32_t>(1 + j), st_a);
    }
    for (uint32_t j = 0; j < K; j++) {
        read_half(cb_tap, j, get_arg_val<uint32_t>(1 + K + j), tap_a);
    }
    noc.async_read_barrier();

    // Place each head's two 32 B runs (scratch slot i: words 0..7 = face 0 row 0, words 128..135 = face 1 row 0)
    // into row i of this half's faces (face 2*half and 2*half + 1), in both copies of the new row.
    const CoreLocalMem<volatile uint32_t> sw(scr.get_write_ptr());
    auto place = [&](uint32_t cb_id, uint32_t page) {
        CircularBuffer cb(cb_id);
        const CoreLocalMem<volatile uint32_t> w(cb.get_write_ptr() + page * tb + hoff);
        for (uint32_t i = 0; i < 16; i++) {
            for (uint32_t x = 0; x < 8; x++) {
                w[i * 8 + x] = sw[i * 256 + x];              // face 2*half,   row i (16 bf16 = 8 words per row)
                w[128 + i * 8 + x] = sw[i * 256 + 128 + x];  // face 2*half+1, row i
            }
        }
    };
    place(cb_new, 0);
    place(cb_wst, K - 1);
    cn.push_back(1);
    cc.push_back(K - 1);
    ct.push_back(K);
    cw.push_back(K);
}
