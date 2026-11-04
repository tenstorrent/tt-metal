// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// otherlayers: reader of the indexer-q RoPE program. One core per output tile (r, c) of q [R,1,32,128] (user row r,
// 32-column tile c of the 128-wide head dim):
//   x tile  row h (head h, 0..31) = row r of tile column h*4 + c of the wq_b linear output qlin [1,1,R,4096] (the
//   'reshape [1,1,R,4096] -> [R,1,32,128]' of SpecIndexer.project, 20 us as a stock op), 64 B reads
//           of the row pair (r & ~1, r | 1) of both faces, then the row copy into the x tile
//   c tile  = row 0 of tile (r, c) of the cos table st["C"] [R,1,1,128] replicated over the 32 rows (the broadcast
//   operand of ttnn.multiply(q, C)) s tile  = row 0 of tile (r, c) of st["S"] replicated over 32 rows (the broadcast
//   operand of ttnn.addcmul(m, p, S)) P tile  = the signed pair swap P (x @ P exact) for the rotated tile columns c >=
//   2 (dims 64..127), zero for c < 2 (cos 1, sin 0: out = x)
// Same operand construction as tt/attn_kernels/post_reader.cpp (UNF branch), which the compute kernel
// tt/attn_kernels/post_compute.cpp (UNF) consumes.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ol_zones.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_r = get_compile_time_arg_val(1);   // c replicated
    constexpr uint32_t cb_s = get_compile_time_arg_val(2);   // scratch, 256 B: table row pieces
    constexpr uint32_t cb_xs = get_compile_time_arg_val(3);  // scratch, 4096 B: 64 x 64 B pieces of the qlin rows
    constexpr uint32_t cb_r2 = get_compile_time_arg_val(4);  // P
    constexpr uint32_t cb_ri = get_compile_time_arg_val(5);  // s replicated
    constexpr auto q_args = TensorAccessorArgs<6>();
    constexpr auto c_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();

    const uint32_t t = ol_core_idx(get_common_arg_val<uint32_t>(3));  // tile index r * 4 + c
    const uint32_t q_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t c_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t s_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t r = t >> 2, c = t & 3;

    Noc noc;
    const auto q_acc = TensorAccessor(q_args, q_addr, 2048);
    const auto c_acc = TensorAccessor(c_args, c_addr, 2048);
    const auto s_acc = TensorAccessor(s_args, s_addr, 2048);
    OLZ("Z_RALL");
    experimental::CB cbx(cb_x), cbr(cb_r), cbs(cb_s), cbxs(cb_xs), cbr2(cb_r2), cbri(cb_ri);
    cbx.reserve_back(1);
    cbr.reserve_back(1);
    cbs.reserve_back(1);
    cbxs.reserve_back(1);
    cbr2.reserve_back(1);
    cbri.reserve_back(1);

    // the row pair (r & ~1, r | 1) of face (r >> 4) * 2 + f of tile column h * 4 + c: 64 B at face * 512 + ((r & 15) &
    // ~1) * 32
    const uint32_t roff = (((r >> 4) << 1) * 512) + (((r & 15) & ~1u) * 32);
    for (uint32_t h = 0; h < 32; ++h) {
        noc.async_read(q_acc, cbxs, 64, {.page_id = h * 4 + c, .offset_bytes = roff}, {.offset_bytes = (h * 2) * 64});
        noc.async_read(
            q_acc, cbxs, 64, {.page_id = h * 4 + c, .offset_bytes = roff + 512}, {.offset_bytes = (h * 2 + 1) * 64});
    }
    // table row 0 of tile (r, c): face 0 and face 1, 64 B each (rows 0 and 1; row 0 is used)
    noc.async_read(c_acc, cbs, 64, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read(c_acc, cbs, 64, {.page_id = t, .offset_bytes = 512}, {.offset_bytes = 64});
    noc.async_read(s_acc, cbs, 64, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = 128});
    noc.async_read(s_acc, cbs, 64, {.page_id = t, .offset_bytes = 512}, {.offset_bytes = 192});
    noc.async_write_zeros(cbr, 2048, {.offset_bytes = 0});
    noc.async_write_zeros(cbx, 2048, {.offset_bytes = 0});
    noc.async_write_zeros(cbr2, 2048, {.offset_bytes = 0});
    noc.async_write_zeros(cbri, 2048, {.offset_bytes = 0});
    {
        OLZ("Z_WAIT");
        noc.write_zeros_l1_barrier();
        noc.async_read_barrier();
    }

    {
        OLZ("Z_XCOPY");
        // x tile: row h of face (h >> 4) * 2 + f from the piece of row r (offset (r & 1) * 32 B in the 64 B chunk)
        volatile tt_l1_ptr uint32_t* xs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbxs.get_write_ptr());
        volatile tt_l1_ptr uint32_t* xt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbx.get_write_ptr());
        const uint32_t po = (r & 1) * 8;  // words
        for (uint32_t h = 0; h < 32; ++h) {
            volatile tt_l1_ptr uint32_t* d = xt + ((h >> 4) << 8) + ((h & 15) << 3);
            volatile tt_l1_ptr uint32_t* s0 = xs + (h * 2) * 16 + po;
            volatile tt_l1_ptr uint32_t* s1 = xs + (h * 2 + 1) * 16 + po;
            for (uint32_t q = 0; q < 8; ++q) {
                d[q] = s0[q];
                d[128 + q] = s1[q];
            }
        }
    }
    {
        OLZ("Z_TAB");
        volatile tt_l1_ptr uint16_t* sc = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbs.get_write_ptr());
        volatile tt_l1_ptr uint16_t* R = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbr.get_write_ptr());
        volatile tt_l1_ptr uint16_t* R2 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbr2.get_write_ptr());
        volatile tt_l1_ptr uint16_t* RI = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbri.get_write_ptr());
        // element (row r, col q) of a bf16 tile: face (r>>4)*2 + (q>>4), within-face (r&15)*16 + (q&15)
#define EL(rr, qq) ((((((rr) >> 4) << 1) + ((qq) >> 4)) << 8) + (((rr) & 15) << 4) + ((qq) & 15))
        if (c >= 2) {
            for (uint32_t e = 0; e < 32; e += 2) {
                R2[EL(e, e + 1)] = 0x3F80;
                R2[EL(e + 1, e)] = 0xBF80;
            }
        }
        for (uint32_t q = 0; q < 32; ++q) {
            R[EL(0, q)] = sc[(q >> 4) * 32 + (q & 15)];
            RI[EL(0, q)] = sc[64 + (q >> 4) * 32 + (q & 15)];
        }
#undef EL
        volatile tt_l1_ptr uint32_t* W = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbri.get_write_ptr());
        volatile tt_l1_ptr uint32_t* V = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbr.get_write_ptr());
        for (uint32_t rr = 1; rr < 32; ++rr) {
            for (uint32_t f = 0; f < 2; ++f) {
                for (uint32_t q = 0; q < 8; ++q) {
                    const uint32_t d = (((rr >> 4) << 1) + f) * 128 + (rr & 15) * 8 + q;
                    W[d] = W[f * 128 + q];
                    V[d] = V[f * 128 + q];
                }
            }
        }
    }
    cbr2.push_back(1);
    cbri.push_back(1);
    cbr.push_back(1);
    cbx.push_back(1);
}
