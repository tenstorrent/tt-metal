// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Post-SDPA glue, rope cores (one per (user t, rope column tile j in {0,1})): builds the x tile for the inverse RoPE of
// dims 448 + 32 j .. +31 of the NH head rows of user t straight from the row-major SDPA output [1,32,T,512] (page h*T +
// t, 1 KB), and the pair rotation matrix R from row 0 of the cos / (negated) sin table tiles -- the exact
// rope_reader.cpp construction.  The matmul x @ R (post_compute.cpp == rope_compute.cpp) then yields the rotated rows
// (row h = head h).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);   // scratch, 1 page: cos / sin row pieces
    constexpr uint32_t cb_xs = get_compile_time_arg_val(3);  // scratch, 1 page: NH x 64 B head row pieces
    constexpr uint32_t T = get_compile_time_arg_val(4);
    constexpr uint32_t NH = get_compile_time_arg_val(5);
    constexpr uint32_t TS = get_compile_time_arg_val(6);   // table tile pages per user (512 / 32)
    constexpr uint32_t ROW = get_compile_time_arg_val(7);  // bytes of one SDPA output row
    constexpr auto o_args = TensorAccessorArgs<8>();
    constexpr auto c_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();

    const uint32_t w = get_arg_val<uint32_t>(0);
    const uint32_t o_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t t = w >> 1, j = w & 1;

    Noc noc;
    const auto o_acc = TensorAccessor(o_args, o_addr, ROW);
    const auto c_acc = TensorAccessor(c_args, c_addr, 2048);
    const auto s_acc = TensorAccessor(s_args, s_addr, 2048);
    experimental::CB cbx(cb_x), cbr(cb_r), cbs(cb_s), cbxs(cb_xs);

    cbx.reserve_back(1);
    cbr.reserve_back(1);
    cbs.reserve_back(1);
    cbxs.reserve_back(1);
    for (uint32_t h = 0; h < NH; ++h) {
        noc.async_read(o_acc, cbxs, 64, {.page_id = h * T + t, .offset_bytes = 896 + 64 * j}, {.offset_bytes = h * 64});
    }
    const uint32_t tp = t * TS + 14 + j;
    noc.async_read(c_acc, cbs, 64, {.page_id = tp, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read(c_acc, cbs, 64, {.page_id = tp, .offset_bytes = 512}, {.offset_bytes = 64});
    noc.async_read(s_acc, cbs, 64, {.page_id = tp, .offset_bytes = 0}, {.offset_bytes = 128});
    noc.async_read(s_acc, cbs, 64, {.page_id = tp, .offset_bytes = 512}, {.offset_bytes = 192});
    noc.async_write_zeros(cbr, 2048, {.offset_bytes = 0});
    noc.async_write_zeros(cbx, 2048, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    noc.async_read_barrier();

    // x tile rows 0..NH-1: bf16 columns 0..15 of the 64 B piece -> face (h>>4)*2 row h&15, columns 16..31 -> face +1
    volatile tt_l1_ptr uint32_t* xs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbxs.get_write_ptr());
    volatile tt_l1_ptr uint32_t* xt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbx.get_write_ptr());
    for (uint32_t h = 0; h < NH; ++h) {
        volatile tt_l1_ptr uint32_t* d = xt + ((h >> 4) << 8) + ((h & 15) << 3);
        volatile tt_l1_ptr uint32_t* s = xs + h * 16;
        for (uint32_t q = 0; q < 8; ++q) {
            d[q] = s[q];
            d[128 + q] = s[8 + q];
        }
    }

    volatile tt_l1_ptr uint16_t* sc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbs.get_write_ptr());
    volatile tt_l1_ptr uint16_t* R = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbr.get_write_ptr());
    // element (row r, col q) of a bf16 tile: face (r>>4)*2 + (q>>4), within-face (r&15)*16 + (q&15)
#define EL(r, q) ((((((r) >> 4) << 1) + ((q) >> 4)) << 8) + (((r) & 15) << 4) + ((q) & 15))
    for (uint32_t e = 0; e < 32; e += 2) {
        const uint32_t half = e >> 4;                      // which face half of the table row
        const uint16_t c = sc[half * 32 + (e & 15)];       // cos at the even column
        const uint16_t s = sc[64 + half * 32 + (e & 15)];  // sin at the even column
        R[EL(e, e)] = c;
        R[EL(e + 1, e + 1)] = c;
        R[EL(e, e + 1)] = s;
        R[EL(e + 1, e)] = s ^ 0x8000;
    }
#undef EL
    cbr.push_back(1);
    cbx.push_back(1);
}
