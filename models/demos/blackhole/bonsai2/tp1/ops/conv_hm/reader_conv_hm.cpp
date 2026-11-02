// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader for conv_hm (fused head-major GDN decode conv). One core per head-major state tile (r_t, c_t): rows
// 32*r_t .. 32*r_t + rows - 1 are heads, columns 32*c_t .. 32*c_t + 31 their channels.
//
//   new row : gathered from the qkv ROW [1,1,C]: head h's 32 channels of tile c_t are row 0 of qkv tile h*4 + c_t,
//             i.e. two 16-element runs (face 0 and face 1 of that tile); they land in row (h - 32 r_t) of a zero tile.
//   cb_cst  : old state rows 1..K-1 (tile (r_t, c_t) of st[1..K-1]) for compute
//   cb_tap  : the K taps' tile (r_t, c_t)
//   cb_wst  : old state rows 1..K-1 and the new row again, for the writer (one consumer per CB)

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_new = 0, cb_cst = 1, cb_tap = 2, cb_out = 3, cb_wst = 4;

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);  // conv taps (4)
    constexpr auto qkv_a = TensorAccessorArgs<1>();
    constexpr auto st_a = TensorAccessorArgs<qkv_a.next_compile_time_args_offset()>();   // the K state tensors share it
    constexpr auto tap_a = TensorAccessorArgs<st_a.next_compile_time_args_offset()>();   // the K tap tensors share it

    // runtime: 0 qkv, 1..K state addrs, K+1..2K tap addrs, 2K+1 r_t, 2K+2 c_t, 2K+3 rows
    const uint32_t qkv_addr = get_arg_val<uint32_t>(0);
    const uint32_t r_t = get_arg_val<uint32_t>(2 * K + 1);
    const uint32_t c_t = get_arg_val<uint32_t>(2 * K + 2);
    const uint32_t rows = get_arg_val<uint32_t>(2 * K + 3);
    const uint32_t tile = r_t * 4 + c_t;  // page of this core's tile in every [1, 80, 128] tensor

    const uint32_t tb = get_tile_size(cb_new);  // bf16 2048 B
    const auto qkv = TensorAccessor(qkv_a, qkv_addr, tb);
    Noc noc;

    // The new row tile, assembled twice (compute and writer each consume their own copy).
    auto gather_new = [&](uint32_t cb_id, uint32_t page) {
        CircularBuffer cb(cb_id);
        const CoreLocalMem<volatile uint32_t> w(cb.get_write_ptr() + page * tb);
        for (uint32_t i = 0; i < tb / 4; i++) {
            w[i] = 0;  // padding rows must be finite: they go through the SFPU and get written back to the state
        }
        for (uint32_t i = 0; i < rows; i++) {
            const uint32_t h = 32 * r_t + i;
            const uint32_t f = (i < 16) ? 0u : 2u;  // faces 0|1 hold rows 0-15, faces 2|3 rows 16-31
            const uint32_t dst = page * tb + (f * 256 + (i % 16) * 16) * 2;
            noc.async_read(qkv, cb, 32, {.page_id = h * 4 + c_t, .offset_bytes = 0}, {.offset_bytes = dst});
            noc.async_read(qkv, cb, 32, {.page_id = h * 4 + c_t, .offset_bytes = 512}, {.offset_bytes = dst + 512});
        }
    };
    auto read_tile = [&](uint32_t cb_id, uint32_t page, uint32_t addr, const auto& args) {
        CircularBuffer cb(cb_id);
        const auto acc = TensorAccessor(args, addr, tb);
        noc.async_read(acc, cb, tb, {.page_id = tile}, {.offset_bytes = page * tb});
    };

    CircularBuffer cn(cb_new), cc(cb_cst), ct(cb_tap), cw(cb_wst);
    cn.reserve_back(1);
    cc.reserve_back(K - 1);
    ct.reserve_back(K);
    cw.reserve_back(K);
    gather_new(cb_new, 0);
    gather_new(cb_wst, K - 1);
    for (uint32_t j = 1; j < K; j++) {
        read_tile(cb_cst, j - 1, get_arg_val<uint32_t>(1 + j), st_a);
        read_tile(cb_wst, j - 1, get_arg_val<uint32_t>(1 + j), st_a);
    }
    for (uint32_t j = 0; j < K; j++) {
        read_tile(cb_tap, j, get_arg_val<uint32_t>(1 + K + j), tap_a);
    }
    noc.async_read_barrier();
    cn.push_back(1);
    cc.push_back(K - 1);
    ct.push_back(K);
    cw.push_back(K);
}
