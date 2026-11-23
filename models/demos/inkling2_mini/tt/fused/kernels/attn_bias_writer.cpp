// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention glue, bias role, writer: mask tiles j0 .. j0 + nj - 1 -> mask [1, 1, 24, 1024] bf16 (DRAM), or
// with grp the same rows in kv-group layout [1, 8, 32, 1024] (rows 0..2 of each group).
// has_sink: also the sdpa sink tile [24, 32] bf16, row h = sink logit h (projection row 0, 24 columns), and
// with sink_blk (global layers) the sink logit block [1, 8, 3, 32] fp32 of the near softmax: col 0 = sink logit,
// cols 1..31 = neg (rows 3..31 of each tile stay as allocated, zero).
#include "attn_io.hpp"

constexpr bool proj_dram = get_named_compile_time_arg_val("proj_dram");  // projection output in DRAM (else L1)

void kernel_main() {
    constexpr uint32_t nj = get_named_compile_time_arg_val("nj");
    constexpr uint32_t has_sink = get_named_compile_time_arg_val("has_sink");
    constexpr uint32_t sink_tile = get_named_compile_time_arg_val("sink_tile");  // projection tile of sink col 0
    constexpr uint32_t sink_blk = get_named_compile_time_arg_val("sink_blk");
    constexpr uint32_t neg = get_named_compile_time_arg_val("neg");  // fp32 bits
    constexpr uint32_t grp = get_named_compile_time_arg_val("grp");  // mask in kv-group layout (global layers)
    constexpr uint32_t cb_mask = 21, cb_sink = 6, cb_blk = 7, cb_stage = 25;

    const uint32_t mask = get_arg_val<uint32_t>(0);
    const uint32_t j0 = get_arg_val<uint32_t>(1);
    const uint32_t proj = get_arg_val<uint32_t>(2);
    const uint32_t sink = get_arg_val<uint32_t>(3);
    const uint32_t blk = get_arg_val<uint32_t>(4);

    if constexpr (has_sink) {
        const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);
        cb_reserve_back(cb_sink, 1);
        const uint32_t w = get_write_ptr(cb_sink);
        attnio::zero_l1(w, 2048);
        attnio::gather_row<proj_dram>(proj, 2048, 2, sink_tile, 1, 0, stage + 256, stage);  // barriers (zeros too)
        volatile tt_l1_ptr uint16_t* v = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage + 256);
        for (uint32_t h = 0; h < 24; ++h) {
            const uint32_t word = static_cast<uint32_t>(v[h]) * 0x10001u;
            const uint32_t face = h < 16 ? 0 : 2;
            volatile tt_l1_ptr uint32_t* row =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w + (face * 256 + (h & 15) * 16) * 2);
            for (uint32_t k = 0; k < 8; ++k) {
                row[k] = word;        // face (cols 0..15)
                row[128 + k] = word;  // face + 1 (cols 16..31)
            }
        }
        const InterleavedAddrGen<true> dst = {.bank_base_address = sink, .page_size = 2048};
        noc_async_write(w, dst.get_noc_addr(0), 2048);
        if constexpr (sink_blk) {
            cb_reserve_back(cb_blk, 1);
            const uint32_t b = get_write_ptr(cb_blk);  // 24 rows x (face 0 chunk, face 1 chunk), 64 B each
            const InterleavedAddrGen<true> bd = {.bank_base_address = blk, .page_size = 4096};
            for (uint32_t h = 0; h < 24; ++h) {
                volatile tt_l1_ptr uint32_t* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(b + h * 128);
                row[0] = static_cast<uint32_t>(v[h]) << 16;
                for (uint32_t k = 1; k < 32; ++k) {
                    row[k] = neg;
                }
                const uint32_t g = h / 3, k = h % 3;
                noc_async_write(b + h * 128, bd.get_noc_addr(g, k * 64), 64);
                noc_async_write(b + h * 128 + 64, bd.get_noc_addr(g, 1024 + k * 64), 64);
            }
        }
    }

    const InterleavedAddrGen<true> dst = {.bank_base_address = mask, .page_size = 2048};
    for (uint32_t j = 0; j < nj; ++j) {
        cb_wait_front(cb_mask, 1);
        const uint32_t src = get_read_ptr(cb_mask);
        if constexpr (grp) {  // [1, 8, 32, 1024]: head h -> row h % 3 of kv group h / 3 (other rows stay zero)
            for (uint32_t h = 0; h < 24; ++h) {
                const uint32_t g = h / 3, k = h % 3, f0 = (h >> 4) * 2, r = h & 15;
                for (uint32_t f = 0; f < 2; ++f) {
                    noc_async_write(
                        src + ((f0 + f) * 256 + r * 16) * 2,
                        dst.get_noc_addr(g * 32 + j0 + j, (f * 256 + k * 16) * 2),
                        32);
                }
            }
        } else {
            noc_async_write(src, dst.get_noc_addr(j0 + j), 2048);
        }
        noc_async_write_barrier();
        cb_pop_front(cb_mask, 1);
    }
    noc_async_write_barrier();
}
