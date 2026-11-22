// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention output prep, reader: rows h0 .. h0 + nh - 1 of the attention output [1, 1, 24, 128] (bf16 TILE,
// DRAM) -> one dense bf16 page (head k at elements 128k ..), and the matching gate columns (projection row 0) ->
// a second dense page.
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t nh = get_named_compile_time_arg_val("nh");
    constexpr uint32_t gate_tile = get_named_compile_time_arg_val("gate_tile");  // projection tile of gate col 0
    constexpr uint32_t cb_a = 0, cb_g = 1, cb_stage = 24;

    const uint32_t attn = get_arg_val<uint32_t>(0);
    const uint32_t proj = get_arg_val<uint32_t>(1);
    const uint32_t h0 = get_arg_val<uint32_t>(2);
    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);

    cb_reserve_back(cb_a, 1);
    cb_reserve_back(cb_g, 1);
    const uint32_t aw = get_write_ptr(cb_a), gw = get_write_ptr(cb_g);
    attnio::zero_l1(aw + nh * 256, 2048 - nh * 256);
    attnio::zero_l1(gw + nh * 256, 2048 - nh * 256);
    for (uint32_t k = 0; k < nh; ++k) {
        attnio::gather_row(attn, 2048, 2, 0, 4, h0 + k, aw + k * 256, stage);
    }
    attnio::gather_row(proj, 2048, 2, gate_tile + h0 * 4, nh * 4, 0, gw, stage);
    cb_push_back(cb_a, 1);
    cb_push_back(cb_g, 1);
}
