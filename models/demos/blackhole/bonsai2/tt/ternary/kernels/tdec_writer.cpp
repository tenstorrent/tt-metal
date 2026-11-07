// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Ternary decode linear, writer (BRISC). First builds the block-diagonal activation in cb_act from the rotated
// activation row x' (bf16, K values): tile kt is zero except row (kt / 4) % 32, which holds x'[32kt .. 32kt+31].
// Zero-fill: one tile by stores, then doubling local NoC copies. Also fills the bf16 ones tile the row sum reads.
// Then streams the odd output-tile records into c_6 (tdec_stream.hpp), then writes this core's output tiles.
#include "tdec_stream.hpp"

void kernel_main() {
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t first_nt = get_arg_val<uint32_t>(2);
    const uint32_t num_nt = get_arg_val<uint32_t>(3);
    const uint32_t w_addr = get_arg_val<uint32_t>(4);
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t ppr = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr auto x_args = TensorAccessorArgs<3>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto w_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_act = tt::CBIndex::c_0;
    constexpr uint32_t cb_x = tt::CBIndex::c_5;
    constexpr uint32_t cb_ones = tt::CBIndex::c_3;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t A_TILE = 2048;
    constexpr uint32_t X_BYTES = Kt * 64;

    const auto xs = TensorAccessor(x_args, x_addr, X_BYTES);
    cb_reserve_back(cb_x, 1);
    const uint32_t x_l1 = get_write_ptr(cb_x);
    noc_async_read(xs.get_noc_addr(0), x_l1, X_BYTES);

    cb_reserve_back(cb_act, Kt);
    const uint32_t base = get_write_ptr(cb_act);
    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    for (uint32_t i = 0; i < A_TILE / 4; ++i) {
        z[i] = 0;
    }
    uint32_t done = 1;
    while (done < Kt) {
        const uint32_t n = (Kt - done < done) ? (Kt - done) : done;
        noc_async_read(get_noc_addr(base), base + done * A_TILE, n * A_TILE);
        noc_async_read_barrier();
        done += n;
    }
    noc_async_read_barrier();  // x' too
    const volatile tt_l1_ptr uint32_t* xw = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(x_l1);
    for (uint32_t kt = 0; kt < Kt; ++kt) {
        const uint32_t r = (kt >> 2) & 31;
        const uint32_t row = base + kt * A_TILE + (r >> 4) * 1024 + (r & 15) * 32;
        volatile tt_l1_ptr uint32_t* d0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row);
        volatile tt_l1_ptr uint32_t* d1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(row + 512);
        const volatile tt_l1_ptr uint32_t* s = xw + kt * 16;
        for (uint32_t j = 0; j < 8; ++j) {
            d0[j] = s[j];
            d1[j] = s[8 + j];
        }
    }
    cb_push_back(cb_act, Kt);

    cb_reserve_back(cb_ones, 1);
    volatile tt_l1_ptr uint32_t* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ones));
    for (uint32_t i = 0; i < A_TILE / 4; ++i) {
        ones[i] = 0x3F803F80;  // two bf16 1.0
    }
    cb_push_back(cb_ones, 1);

    const auto w = TensorAccessor(w_args, w_addr, 4096);
    stream_records<ppr, NB>(tt::CBIndex::c_6, w, first_nt, num_nt, 1);

    const uint32_t bytes = get_tile_size(cb_out);
    const auto out = TensorAccessor(out_args, out_addr, bytes);
    for (uint32_t i = 0; i < num_nt; ++i) {
        cb_wait_front(cb_out, 1);
        noc_async_write(get_read_ptr(cb_out), out.get_noc_addr(first_nt + i), bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
