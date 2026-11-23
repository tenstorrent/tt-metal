// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention glue, bias role, reader. Builds (all as 32x32 tiles):
//   cb_rel:   raw rel states as a [24 heads, 16] matrix (projection row 0, 24 chunks of 16)
//   cb_rs:    rel clip scale, same layout (fp32);  cb_g: per-head sum matrix (bf16);  cb_tau: tau broadcast (fp32)
//   cb_p:     nj profile tiles, rows = ring slots 32j .. 32j + 31, cols = rel dim: rows of the reversed profile
//             table starting at (32j - ring) mod 1024 (a contiguous run of 32 rows of 16 bf16)
//   cb_near:  the near-mask tiles of those slots (fp32, row 0)
// slim: read only the bytes the compute uses, zeros elsewhere (as in the tensors), G built in L1: rel_scale rows
// 0..23 of columns 0..15 (faces 0, 2), the near tiles' row 0 (the compute broadcasts it).
#include "attn_io.hpp"

constexpr bool proj_dram = get_named_compile_time_arg_val("proj_dram");  // projection output in DRAM (else L1)

void kernel_main() {
    constexpr uint32_t nj = get_named_compile_time_arg_val("nj");
    constexpr uint32_t has_tau = get_named_compile_time_arg_val("has_tau");
    constexpr uint32_t rel_tile = get_named_compile_time_arg_val("rel_tile");  // projection tile of rel column 0
    constexpr uint32_t prow = get_named_compile_time_arg_val("prow");          // projection row (4 t: shared window)
    constexpr uint32_t slim = get_named_compile_time_arg_val("slim");
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

    // stage: [0, 1536) rel row blocks, [1536] tau, [1600] ring slot, [1664 + 1088 j) profile rows of slot tile j
    constexpr uint32_t st_tau = 1536, st_ring = 1600, st_prof = 1664;
    cb_reserve_back(cb_rs, 1);
    cb_reserve_back(cb_g, 1);
    cb_reserve_back(cb_near, nj);
    cb_reserve_back(cb_p, nj);
    cb_reserve_back(cb_rel, 1);
    const uint32_t pw = get_write_ptr(cb_p), rw = get_write_ptr(cb_rel);
    // wave 1: constants, zeros, rel row blocks, tau, ring slot - one read barrier
    if constexpr (slim) {
        const uint32_t rsw = get_write_ptr(cb_rs), gw = get_write_ptr(cb_g), nw = get_write_ptr(cb_near);
        const InterleavedAddrGen<true> rs_a = {.bank_base_address = rs, .page_size = 4096};
        noc_async_read(rs_a.get_noc_addr(0, 0), rsw, 1024);           // face 0: rows 0..15
        noc_async_read(rs_a.get_noc_addr(0, 2048), rsw + 2048, 512);  // face 2: rows 16..23
        attnio::zero_l1(rsw + 1024, 1024);
        attnio::zero_l1(rsw + 2560, 4096 - 2560);
        attnio::zero_l1(gw + 512, 2048 - 512);  // G: bf16 1.0 on face 0 (rows, cols 0..15), zero elsewhere
        volatile tt_l1_ptr uint32_t* gp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(gw);
        for (uint32_t k = 0; k < 128; ++k) {
            gp[k] = 0x3f803f80u;
        }
        const InterleavedAddrGen<true> near_a = {.bank_base_address = near, .page_size = 4096};
        for (uint32_t j = 0; j < nj; ++j) {  // row 0 = face 0 row 0 + face 1 row 0
            const uint32_t dst = nw + j * 4096;
            noc_async_read(near_a.get_noc_addr(j0 + j, 0), dst, 64);
            noc_async_read(near_a.get_noc_addr(j0 + j, 1024), dst + 1024, 64);
            attnio::zero_l1(dst + 64, 1024 - 64);
            attnio::zero_l1(dst + 1088, 4096 - 1088);
        }
    } else {
        attnio::read_pages(rs, 4096, 0, 1, get_write_ptr(cb_rs));
        attnio::read_pages(g, 2048, 0, 1, get_write_ptr(cb_g));
        attnio::read_pages(near, 4096, j0, nj, get_write_ptr(cb_near));
    }
    attnio::zero_l1(pw, nj * 2048);
    attnio::zero_l1(rw, 2048);
    attnio::stage_row<proj_dram>(proj, 2048, 2, rel_tile, 12, prow, stage);
    const InterleavedAddrGen<true> ring_a = {.bank_base_address = ring, .page_size = 4};
    noc_async_read(ring_a.get_noc_addr(0, 0), stage + st_ring, 64);
    if constexpr (has_tau) {
        const InterleavedAddrGen<true> tau_a = {.bank_base_address = tau, .page_size = 4096};
        noc_async_read(tau_a.get_noc_addr(0, 0), stage + st_tau, 64);
    }
    noc_async_read_barrier();

    // wave 2: rel matrix (head h's 16 values -> row h: face 0 for h < 16, face 2 after), the profile rows
    // (row prow sits at its offset inside each staged 64B block)
    const uint32_t rel_in = attnio::row_offset(prow, 2) & (attnio::ALIGN - 1);
    for (uint32_t h = 0; h < 24; ++h) {
        const uint32_t dst = h < 16 ? rw + 32 * h : rw + 1024 + 32 * (h - 16);
        noc_async_read(get_noc_addr(stage + (h >> 1) * 128 + (h & 1) * 64 + rel_in), dst, 32);
    }
    const uint32_t rpos = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + st_ring);
    const InterleavedAddrGen<true> tab = {.bank_base_address = table, .page_size = 1};
    uint32_t in_blk[nj];
    for (uint32_t j = 0; j < nj; ++j) {
        const uint32_t off = ((32 * (j0 + j) + 1024 - rpos) & 1023) * 32;
        const uint32_t blk = off & ~(attnio::ALIGN - 1);
        in_blk[j] = off - blk;
        noc_async_read(tab.get_noc_addr(0, blk), stage + st_prof + j * 1088, 1088);
    }
    if constexpr (has_tau) {
        cb_reserve_back(cb_tau, 1);
        const uint32_t bits = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + st_tau);  // tau at [0, 0]
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_tau));
        for (uint32_t k = 0; k < 1024; ++k) {
            p[k] = bits;
        }
        cb_push_back(cb_tau, 1);
    }
    noc_async_read_barrier();

    // wave 3: profile tiles (rows 0..15 -> face 0, 16..31 -> face 2)
    for (uint32_t j = 0; j < nj; ++j) {
        const uint32_t s = stage + st_prof + j * 1088 + in_blk[j];
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
