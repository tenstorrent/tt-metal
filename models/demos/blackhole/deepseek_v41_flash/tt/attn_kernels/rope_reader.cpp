// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Partial RoPE (adjacent pairs) on one 32-col tile (user t, tile col 14+j of a 512-wide row), as x @ R with the 32x32
// rotation matrix R built here from row 0 of the cos / sin table tiles:
//   y[2i] = c x[2i] - s x[2i+1],  y[2i+1] = c x[2i+1] + s x[2i]   =>  R[2i,2i]=R[2i+1,2i+1]=c, R[2i,2i+1]=s,
//   R[2i+1,2i]=-s.
// Core w handles t = w>>1, j = w&1. x page = t*XS + 14 + j (XS = 16 head layout [1,T,32,512], 0 row layout
// [1,1,T,512]), table page = t*TS + 14 + j.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);  // scratch, 1 page
    constexpr uint32_t XS = get_compile_time_arg_val(3);
    constexpr uint32_t TS = get_compile_time_arg_val(4);
    constexpr auto x_args = TensorAccessorArgs<5>();
    constexpr auto c_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();

    const uint32_t w = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t t = w >> 1, j = w & 1;

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, 2048);
    const auto c_acc = TensorAccessor(c_args, c_addr, 2048);
    const auto s_acc = TensorAccessor(s_args, s_addr, 2048);
    experimental::CB cbx(cb_x), cbr(cb_r), cbs(cb_s);

    cbx.reserve_back(1);
    cbr.reserve_back(1);
    cbs.reserve_back(1);
    noc.async_read(x_acc, cbx, 2048, {.page_id = t * XS + 14 + j, .offset_bytes = 0}, {.offset_bytes = 0});
    // row 0 of the table tiles: 64 B aligned reads of face 0 (cols 0..15) and face 1 (cols 16..31), rows 0-1
    const uint32_t tp = t * TS + 14 + j;
    noc.async_read(c_acc, cbs, 64, {.page_id = tp, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read(c_acc, cbs, 64, {.page_id = tp, .offset_bytes = 512}, {.offset_bytes = 64});
    noc.async_read(s_acc, cbs, 64, {.page_id = tp, .offset_bytes = 0}, {.offset_bytes = 128});
    noc.async_read(s_acc, cbs, 64, {.page_id = tp, .offset_bytes = 512}, {.offset_bytes = 192});
    noc.async_write_zeros(cbr, 2048, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    noc.async_read_barrier();

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
