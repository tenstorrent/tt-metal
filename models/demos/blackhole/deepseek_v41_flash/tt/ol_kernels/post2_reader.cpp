// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// otherlayers: post-SDPA rope reader for the stock-unfused rotation (UNF: T > 8 and T % 8 != 0, k = 5 / T = 12),
// variant 2 of tt/attn_kernels/post_reader.cpp: the cos / sin tables are read as raw tiles and broadcast from row 0 by
// the unpacker in qpost2_compute.cpp (no RISC replicate loops over the 31 other rows of two tiles); with the SDPA
// output in L1 (DIRECT, compile arg 2) the 32 B head row pieces are read straight into the face rows of the x tile (no
// scratch, no RISC copy loop). Same argument layout as post_reader.cpp (compile arg 2 = the DIRECT flag instead of the
// scratch CB).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ol_zones.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);  // raw c tile
    constexpr bool DIRECT = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t cb_xs = get_compile_time_arg_val(3);  // scratch: NH x 64 B head row pieces
    constexpr uint32_t T = get_compile_time_arg_val(4);
    constexpr uint32_t NH = get_compile_time_arg_val(5);
    constexpr uint32_t TS = get_compile_time_arg_val(6);   // table tile pages per user (512 / 32)
    constexpr uint32_t ROW = get_compile_time_arg_val(7);  // bytes of one SDPA output row
    constexpr uint32_t cb_r2 = get_compile_time_arg_val(9);
    constexpr uint32_t cb_ri = get_compile_time_arg_val(10);  // raw s tile
    constexpr auto o_args = TensorAccessorArgs<11>();
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
    OLZ("Z_RALL");
    experimental::CB cbx(cb_x), cbr(cb_r), cbxs(cb_xs), cbr2(cb_r2), cbri(cb_ri);
    cbx.reserve_back(1);
    cbr.reserve_back(1);
    cbxs.reserve_back(1);
    cbr2.reserve_back(1);
    cbri.reserve_back(1);
    // rows NH..31 of the x tile and the P tile are zero: zero first (the head rows are written by reads below)
    noc.async_write_zeros(cbx, 2048, {.offset_bytes = 0});
    noc.async_write_zeros(cbr2, 2048, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    const uint32_t tp = t * TS + 14 + j;
    noc.async_read(c_acc, cbr, 2048, {.page_id = tp, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read(s_acc, cbri, 2048, {.page_id = tp, .offset_bytes = 0}, {.offset_bytes = 0});
    if constexpr (DIRECT) {
        for (uint32_t h = 0; h < NH; ++h) {
            const uint32_t d0 = (((h >> 4) << 1) * 512) + ((h & 15) * 32);
            noc.async_read(o_acc, cbx, 32, {.page_id = h * T + t, .offset_bytes = 896 + 64 * j}, {.offset_bytes = d0});
            noc.async_read(
                o_acc, cbx, 32, {.page_id = h * T + t, .offset_bytes = 896 + 64 * j + 32}, {.offset_bytes = d0 + 512});
        }
    } else {
        for (uint32_t h = 0; h < NH; ++h) {
            noc.async_read(
                o_acc, cbxs, 64, {.page_id = h * T + t, .offset_bytes = 896 + 64 * j}, {.offset_bytes = h * 64});
        }
    }
    // the signed pair swap P (p = x @ P is exact), built while the reads are in flight
    {
        volatile tt_l1_ptr uint16_t* R2 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbr2.get_write_ptr());
#define EL(rr, qq) ((((((rr) >> 4) << 1) + ((qq) >> 4)) << 8) + (((rr) & 15) << 4) + ((qq) & 15))
        for (uint32_t e = 0; e < 32; e += 2) {
            R2[EL(e, e + 1)] = 0x3F80;
            R2[EL(e + 1, e)] = 0xBF80;
        }
#undef EL
    }
    {
        OLZ("Z_WAIT");
        noc.async_read_barrier();
    }
    if constexpr (!DIRECT) {
        OLZ("Z_XCOPY");
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
    }
    cbr2.push_back(1);
    cbri.push_back(1);
    cbr.push_back(1);
    cbx.push_back(1);
}
