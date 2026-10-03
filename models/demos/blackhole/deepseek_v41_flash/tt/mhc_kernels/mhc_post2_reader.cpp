// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// mHC post kernel v2, reader (one core): the NCONST constant tiles, the NPG packed partial pages (page g*PGPG + n holds
// the token-group-g partials of PPT cores, TG rows each) and the G summation tiles SSEL_g[g*TG + tl, p*TG + tl] = 1 (p
// < PPT).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_consts = get_compile_time_arg_val(0);
    constexpr uint32_t cb_q = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ssel = get_compile_time_arg_val(2);
    constexpr uint32_t NPG = get_compile_time_arg_val(3);
    constexpr uint32_t NCONST = get_compile_time_arg_val(4);
    constexpr uint32_t G = get_compile_time_arg_val(5);
    constexpr uint32_t TG = get_compile_time_arg_val(6);
    constexpr uint32_t PPT = get_compile_time_arg_val(7);
    constexpr uint32_t cb_log = get_compile_time_arg_val(8);
    constexpr uint32_t cb_skin = get_compile_time_arg_val(9);
    constexpr uint32_t T = get_compile_time_arg_val(10);
    constexpr auto c_args = TensorAccessorArgs<11>();
    constexpr auto p_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE = 4096;

    const uint32_t c_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t p_addr = get_common_arg_val<uint32_t>(1);

    Noc noc;
    const auto c_acc = TensorAccessor(c_args, c_addr, TILE);
    const auto p_acc = TensorAccessor(p_args, p_addr, TILE);
    experimental::CB cbc(cb_consts), cbp(cb_q), cbs(cb_ssel), cbl(cb_log), cbk(cb_skin);

    cbc.reserve_back(NCONST);
    for (uint32_t i = 0; i < NCONST; ++i) {
        noc.async_read(c_acc, cbc, TILE, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = i * TILE});
    }
    cbp.reserve_back(NPG);
    for (uint32_t n = 0; n < NPG; ++n) {
        noc.async_read(p_acc, cbp, TILE, {.page_id = n, .offset_bytes = 0}, {.offset_bytes = n * TILE});
    }
    cbs.reserve_back(G);
    noc.async_write_zeros(cbs, G * TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    {
        volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbs.get_write_ptr());
        for (uint32_t g = 0; g < G; ++g) {
            for (uint32_t p = 0; p < PPT; ++p) {
                for (uint32_t tl = 0; tl < TG; ++tl) {
                    const uint32_t row = g * TG + tl;
                    const uint32_t col = p * TG + tl;
                    const uint32_t face = (row >= 16 ? 2 : 0) + (col >= 16 ? 1 : 0);
                    s[g * 1024 + face * 256 + (row % 16) * 16 + (col % 16)] = 0x3F800000u;
                }
            }
        }
    }
    cbs.push_back(G);
    noc.async_read_barrier();
    cbc.push_back(NCONST);
    cbp.push_back(NPG);

    // ---- permute the comb logits (token rows [t, e]) into the SFPU layout of mhc_sinkhorn_sfpu.h (element e = dst
    // vector e, lane t):
    //      word (e >> 3) * 256 + ((e & 7) >> 1) * 64 + (t >> 3) * 16 + (t & 7) * 2 + (e & 1)   (zeros elsewhere: unused
    //      lanes -> exp(0) = 1)
    cbk.reserve_back(1);
    noc.async_write_zeros(cbk, TILE, {.offset_bytes = 0});
    noc.write_zeros_l1_barrier();
    cbl.wait_front(1);
    {
        volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbl.get_read_ptr());
        volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbk.get_write_ptr());
        for (uint32_t t = 0; t < T; ++t) {
            const uint32_t rw = (((t) >> 4) << 9) + (((t) & 15) << 4);
            const uint32_t dw = ((t >> 3) << 4) + ((t & 7) << 1);
            for (uint32_t e = 0; e < 16; ++e) {
                dst[(e >> 3) * 256 + (((e & 7) >> 1) << 6) + dw + (e & 1)] = src[rw + e];
            }
        }
    }
    cbk.push_back(1);
    cbl.pop_front(1);
}
