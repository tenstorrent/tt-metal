// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention glue, bias role, reader. Builds (all as 32x32 tiles):
//   cb_rel:   raw rel states as a [24 heads, 16] matrix (projection row 0, 24 chunks of 16)
//   cb_rs:    rel clip scale, same layout (fp32);  cb_g: per-head sum matrix (bf16);  cb_tau: tau broadcast (fp32)
//   cb_p:     nj profile tiles, rows = ring slots 32j .. 32j + 31, cols = rel dim: rows of the reversed profile
//             table starting at (32j - ring) mod 1024 (a contiguous run of 32 rows of 16 bf16)
//   cb_near:  the near-mask tiles of those slots (fp32, row 0)
#include "attn_io.hpp"

void kernel_main() {
    constexpr uint32_t nj = get_named_compile_time_arg_val("nj");
    constexpr uint32_t has_tau = get_named_compile_time_arg_val("has_tau");
    constexpr uint32_t rel_tile = get_named_compile_time_arg_val("rel_tile");  // projection tile of rel column 0
    constexpr uint32_t cb_rel = 0, cb_rs = 1, cb_g = 2, cb_tau = 3, cb_p = 4, cb_near = 5, cb_stage = 24;

    uint32_t a = 0;
    const uint32_t proj = get_arg_val<uint32_t>(a++);
    const uint32_t rs = get_arg_val<uint32_t>(a++);
    const uint32_t g = get_arg_val<uint32_t>(a++);
    const uint32_t tau = get_arg_val<uint32_t>(a++);
    const uint32_t near = get_arg_val<uint32_t>(a++);
    const uint32_t ring = get_arg_val<uint32_t>(a++);
    const uint32_t table = get_arg_val<uint32_t>(a++);
    const uint32_t j0 = get_arg_val<uint32_t>(a++);
    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);

    cb_reserve_back(cb_rs, 1);
    attnio::read_pages(rs, 4096, 0, 1, get_write_ptr(cb_rs));
    cb_reserve_back(cb_g, 1);
    attnio::read_pages(g, 2048, 0, 1, get_write_ptr(cb_g));
    cb_reserve_back(cb_near, nj);
    attnio::read_pages(near, 4096, j0, nj, get_write_ptr(cb_near));
    cb_reserve_back(cb_p, nj);
    const uint32_t pw = get_write_ptr(cb_p);
    attnio::zero_l1(pw, nj * 2048);

    // rel matrix: dense 384 bf16 at the tile start = heads 0..15 in face 0; heads 16..23 move to face 2
    cb_reserve_back(cb_rel, 1);
    const uint32_t rw = get_write_ptr(cb_rel);
    attnio::gather_row(proj, 2048, 2, rel_tile, 12, 0, rw, stage);
    noc_async_read(get_noc_addr(rw + 512), rw + 1024, 256);
    noc_async_read_barrier();
    attnio::zero_l1(rw + 512, 512);
    attnio::zero_l1(rw + 1280, 768);

    if constexpr (has_tau) {
        cb_reserve_back(cb_tau, 1);
        const uint32_t tw = get_write_ptr(cb_tau);
        const uint32_t bits = attnio::read_u32(tau, 4096, stage);  // tau at [0, 0] of its fp32 tile
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tw);
        for (uint32_t k = 0; k < 1024; ++k) {
            p[k] = bits;
        }
        cb_push_back(cb_tau, 1);
    }

    const uint32_t rpos = attnio::read_u32(ring, 4, stage);
    noc_async_read_barrier();  // profile tiles zeroed
    const InterleavedAddrGen<true> tab = {.bank_base_address = table, .page_size = 1};
    for (uint32_t j = 0; j < nj; ++j) {
        const uint32_t off = ((32 * (j0 + j) + 1024 - rpos) & 1023) * 32;
        const uint32_t blk = off & ~(attnio::ALIGN - 1);
        noc_async_read(tab.get_noc_addr(0, blk), stage, 1088);
        noc_async_read_barrier();
        const uint32_t s = stage + off - blk;
        noc_async_read(get_noc_addr(s), pw + j * 2048, 512);
        noc_async_read(get_noc_addr(s + 512), pw + j * 2048 + 1024, 512);
    }
    noc_async_read_barrier();
    cb_push_back(cb_rs, 1);
    cb_push_back(cb_g, 1);
    cb_push_back(cb_near, nj);
    cb_push_back(cb_rel, 1);
    cb_push_back(cb_p, nj);
}
