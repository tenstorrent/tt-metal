// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Reader for gate_hm v2 (GDN decode gated norm + SiLU(z) gate). One core per value head h. Head h's D = 32*Dt
// channels sit in row 0 of Dt tiles (o: [heads, 1, D] fp32 pages h*Dt + t; z: bf16 row pages h*Dt + t; w: bf16
// pages t). Only those rows are read, and they are packed so the compute kernel works on ONE tile per head:
//   cb_po / cb_pz / cb_pw: row t = row 0 of tile t (rows 0..Dt-1, faces 0 and 1)
//   cb_o4: tile t has o's row 0 of tile t in rows 0..Dt-1, so the transposed matmul's [r, 0] is sum(o^2) for every
//          packed row r (the per-row broadcast source for rinv)
// Rows past Dt are never written and never read back. A NoC transfer keeps source and destination at the same offset
// within a 64 B line: fp32 rows are 64 B, so o lands directly; bf16 rows are 32 B, so z/w are staged in 64 B slots
// and moved into place locally.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_o4 = 0, cb_pz = 1, cb_pw = 2, cb_po = 5, cb_stage = 6;

void kernel_main() {
    constexpr uint32_t Dt = get_compile_time_arg_val(0);
    constexpr auto o_a = TensorAccessorArgs<1>();
    constexpr auto z_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<z_a.next_compile_time_args_offset()>();
    const uint32_t h = get_arg_val<uint32_t>(3);
    const uint32_t tf = get_tile_size(cb_o4), tb = get_tile_size(cb_pz);  // fp32 4096, bf16 2048
    constexpr uint32_t ff = 1024, fb = 512;                                // face bytes, fp32 / bf16

    const auto o = TensorAccessor(o_a, get_arg_val<uint32_t>(0), tf);
    const auto z = TensorAccessor(z_a, get_arg_val<uint32_t>(1), tb);
    const auto w = TensorAccessor(w_a, get_arg_val<uint32_t>(2), tb);
    Noc noc;
    CircularBuffer c4(cb_o4), cpo(cb_po), cpz(cb_pz), cpw(cb_pw), st(cb_stage);
    c4.reserve_back(Dt);
    cpo.reserve_back(1);
    cpz.reserve_back(1);
    cpw.reserve_back(1);
    for (uint32_t t = 0; t < Dt; t++) {
        for (uint32_t f = 0; f < 2; f++) {
            for (uint32_t r = 0; r < Dt; r++) {
                noc.async_read(o, c4, 64, {.page_id = h * Dt + t, .offset_bytes = f * ff},
                               {.offset_bytes = t * tf + f * ff + r * 64});
            }
            noc.async_read(o, cpo, 64, {.page_id = h * Dt + t, .offset_bytes = f * ff}, {.offset_bytes = f * ff + t * 64});
            const uint32_t slot = (t * 2 + f) * 64;  // z slots, then w slots at +2*Dt*64
            noc.async_read(z, st, 32, {.page_id = h * Dt + t, .offset_bytes = f * fb}, {.offset_bytes = slot});
            noc.async_read(w, st, 32, {.page_id = t, .offset_bytes = f * fb}, {.offset_bytes = 2 * Dt * 64 + slot});
        }
    }
    noc.async_read_barrier();
    const CoreLocalMem<volatile uint32_t> s(st.get_write_ptr());
    const CoreLocalMem<volatile uint32_t> pz(cpz.get_write_ptr());
    const CoreLocalMem<volatile uint32_t> pw(cpw.get_write_ptr());
    for (uint32_t t = 0; t < Dt; t++) {
        for (uint32_t f = 0; f < 2; f++) {
            const uint32_t s_w = (t * 2 + f) * 16, d_w = (f * fb + t * 32) / 4;
            for (uint32_t x = 0; x < 8; x++) {
                pz[d_w + x] = s[s_w + x];
                pw[d_w + x] = s[2 * Dt * 16 + s_w + x];
            }
        }
    }
    c4.push_back(Dt);
    cpo.push_back(1);
    cpz.push_back(1);
    cpw.push_back(1);
}
