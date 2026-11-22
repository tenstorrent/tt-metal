// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention output prep, reader: rows h0 .. h0 + nh - 1 of the attention output (bf16 TILE, DRAM, 128 value
// columns) -> one dense bf16 page (head k at elements 128k ..), and the matching gate columns (projection row 0) ->
// a second dense page.
// merge (global layers, nh = 3 = one kv group g): the attention output is the far partial [1, 1, 24, 160] (A | a);
// also the near partial o_near [1, 8, 3, 128] fp32 rows -> dense fp32 page, and per head the far mass a and the
// sink correction r ([1, 8, 3, 1] fp32) broadcast over the head's 128 elements (fp32 pages). far_valid == 0 (the
// far cache is empty): A and a read as zero.
#include "attn_io.hpp"

namespace {
FORCE_INLINE void fill(uint32_t dst, uint32_t bits, uint32_t n) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t i = 0; i < n; ++i) {
        p[i] = bits;
    }
}
}  // namespace

void kernel_main() {
    constexpr uint32_t nh = get_named_compile_time_arg_val("nh");
    constexpr uint32_t gate_tile = get_named_compile_time_arg_val("gate_tile");  // projection tile of gate col 0
    constexpr uint32_t merge = get_named_compile_time_arg_val("merge");
    constexpr uint32_t cb_a = 0, cb_g = 1, cb_m = 2, cb_r = 3, cb_o = 4, cb_stage = 24;

    const uint32_t attn = get_arg_val<uint32_t>(0);
    const uint32_t proj = get_arg_val<uint32_t>(1);
    const uint32_t h0 = get_arg_val<uint32_t>(2);
    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);

    cb_reserve_back(cb_a, 1);
    cb_reserve_back(cb_g, 1);
    const uint32_t aw = get_write_ptr(cb_a), gw = get_write_ptr(cb_g);
    attnio::zero_l1(aw + nh * 256, 2048 - nh * 256);
    attnio::zero_l1(gw + nh * 256, 2048 - nh * 256);
    attnio::gather_row(proj, 2048, 2, gate_tile + h0 * 4, nh * 4, 0, gw, stage);
    uint32_t valid = 1;
    if constexpr (merge) {
        const uint32_t o_near = get_arg_val<uint32_t>(3);
        const uint32_t r_addr = get_arg_val<uint32_t>(4);
        const uint32_t valid_addr = get_arg_val<uint32_t>(5);
        const uint32_t g = get_arg_val<uint32_t>(6);  // kv group
        valid = attnio::read_u32(valid_addr, 4096, stage) != 0;
        cb_reserve_back(cb_m, 1);
        cb_reserve_back(cb_r, 1);
        cb_reserve_back(cb_o, 1);
        const uint32_t mw = get_write_ptr(cb_m), rw = get_write_ptr(cb_r), ow = get_write_ptr(cb_o);
        attnio::zero_l1(mw + nh * 512, 4096 - nh * 512);
        attnio::zero_l1(rw + nh * 512, 4096 - nh * 512);
        attnio::zero_l1(ow + nh * 512, 4096 - nh * 512);
        if (!valid) {
            attnio::zero_l1(aw, nh * 256);
        }
        const InterleavedAddrGen<true> rg = {.bank_base_address = r_addr, .page_size = 4096};
        for (uint32_t k = 0; k < nh; ++k) {
            attnio::gather_row(o_near, 4096, 4, 4 * g, 4, k, ow + k * 512, stage);
            noc_async_read(rg.get_noc_addr(g, k * 64), stage, 64);  // r at (k, 0) of tile g
            noc_async_read_barrier();
            fill(rw + k * 512, *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage), 128);
            uint32_t a_bits = 0;
            if (valid) {
                const uint32_t e = attnio::read_elem_block(attn, 2048, 2, h0 + k, 128, stage);
                a_bits = static_cast<uint32_t>(*reinterpret_cast<volatile tt_l1_ptr uint16_t*>(e)) << 16;
            }
            fill(mw + k * 512, a_bits, 128);
        }
    }
    if (valid) {
        for (uint32_t k = 0; k < nh; ++k) {
            attnio::gather_row(attn, 2048, 2, 0, 4, h0 + k, aw + k * 256, stage);
        }
    }
    noc_async_read_barrier();
    if constexpr (merge) {
        cb_push_back(cb_m, 1);
        cb_push_back(cb_r, 1);
        cb_push_back(cb_o, 1);
    }
    cb_push_back(cb_a, 1);
    cb_push_back(cb_g, 1);
}
