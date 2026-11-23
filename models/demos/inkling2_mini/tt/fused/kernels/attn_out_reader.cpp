// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention output prep, reader: rows h0 .. h0 + nh - 1 of the attention output (bf16 TILE, DRAM, 128 value
// columns) -> one dense bf16 page (head k at elements 128k ..), and the matching gate columns (projection row 0) ->
// a second dense page.
// merge (global layers, nh = 3 = one kv group g): the attention output is the far partial [1, 1, 24, 160] (A | a);
// also the near partial o_near [1, 8, 3, 128] fp32 rows -> dense fp32 page, and per head the far mass a and the
// sink correction r ([1, 8, 3, 1] fp32) broadcast over the head's 128 elements (fp32 pages). far_valid == 0 (the
// far cache is empty): A and a read as zero. far_grp: the far partial is per kv group, [1, 8, 3, 160] (rows 0..2 of
// tile row g).
#include "attn_io.hpp"

namespace {
FORCE_INLINE void fill(uint32_t dst, uint32_t bits, uint32_t n) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t i = 0; i < n; ++i) {
        p[i] = bits;
    }
}
}  // namespace

constexpr bool proj_dram = get_named_compile_time_arg_val("proj_dram");  // projection output in DRAM (else L1)

void kernel_main() {
    constexpr uint32_t nh = get_named_compile_time_arg_val("nh");
    constexpr uint32_t gate_tile = get_named_compile_time_arg_val("gate_tile");  // projection tile of gate col 0
    constexpr uint32_t merge = get_named_compile_time_arg_val("merge");
    constexpr uint32_t far_grp = get_named_compile_time_arg_val("far_grp");
    // rows (multi-token verify, local layers): the attention output is [1, 8R, 3, 128], token t of runtime arg 3 at
    // batch t * 8 + g (rows 0..2 of tile row t * 8 + g)
    constexpr uint32_t rows = get_named_compile_time_arg_val("rows");
    constexpr uint32_t cb_a = 0, cb_g = 1, cb_m = 2, cb_r = 3, cb_o = 4, cb_stage = 24;

    const uint32_t attn = get_arg_val<uint32_t>(0);
    const uint32_t proj = get_arg_val<uint32_t>(1);
    const uint32_t h0 = get_arg_val<uint32_t>(2);
    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);
    // attention output rows: r0 + k of the tile row starting at page t0
    const uint32_t t0 = far_grp ? (h0 / nh) * 5 : (rows ? (get_arg_val<uint32_t>(3) * 8 + h0 / nh) * 4 : 0);
    // merge with far_grp: the far partial's rows start at runtime arg 7 (3 t for token t of a shared verify far call)
    const uint32_t r0 = (far_grp && merge) ? get_arg_val<uint32_t>(7) : ((far_grp || rows) ? 0 : h0);

    // stage: [0, 128 nh*4) gate blocks; attn rows k at st_a + 512 k; merge: valid at st_v, o_near rows at
    // st_o + 512 k, r blocks at st_r + 64 k, a blocks at st_m + 64 k
    constexpr uint32_t st_a = 128 * nh * 4, st_v = st_a + 512 * nh, st_o = st_v + 64, st_r = st_o + 512 * nh;
    constexpr uint32_t st_m = st_r + 64 * nh;
    cb_reserve_back(cb_a, 1);
    cb_reserve_back(cb_g, 1);
    const uint32_t aw = get_write_ptr(cb_a), gw = get_write_ptr(cb_g);
    uint32_t mw = 0, rw = 0, ow = 0, g = 0, e_a[nh];
    // wave 1: zeros and every staged block - one read barrier
    attnio::zero_l1(aw + nh * 256, 2048 - nh * 256);
    attnio::zero_l1(gw + nh * 256, 2048 - nh * 256);
    attnio::stage_row<proj_dram>(proj, 2048, 2, gate_tile + h0 * 4, nh * 4, 0, stage);
    for (uint32_t k = 0; k < nh; ++k) {
        attnio::stage_row(attn, 2048, 2, t0, 4, r0 + k, stage + st_a + 512 * k);
    }
    if constexpr (merge) {
        const uint32_t o_near = get_arg_val<uint32_t>(3);
        const uint32_t r_addr = get_arg_val<uint32_t>(4);
        const uint32_t valid_addr = get_arg_val<uint32_t>(5);
        g = get_arg_val<uint32_t>(6);  // kv group
        const InterleavedAddrGen<true> va = {.bank_base_address = valid_addr, .page_size = 4096};
        noc_async_read(va.get_noc_addr(0, 0), stage + st_v, 64);
        cb_reserve_back(cb_m, 1);
        cb_reserve_back(cb_r, 1);
        cb_reserve_back(cb_o, 1);
        mw = get_write_ptr(cb_m), rw = get_write_ptr(cb_r), ow = get_write_ptr(cb_o);
        attnio::zero_l1(mw + nh * 512, 4096 - nh * 512);
        attnio::zero_l1(rw + nh * 512, 4096 - nh * 512);
        attnio::zero_l1(ow + nh * 512, 4096 - nh * 512);
        const InterleavedAddrGen<true> rg = {.bank_base_address = r_addr, .page_size = 4096};
        for (uint32_t k = 0; k < nh; ++k) {
            attnio::stage_row(o_near, 4096, 4, 4 * g, 4, k, stage + st_o + 512 * k);
            noc_async_read(rg.get_noc_addr(g, k * 64), stage + st_r + 64 * k, 64);  // r at (k, 0) of tile g
            e_a[k] = attnio::stage_elem_block(attn, 2048, 2, r0 + k, 128, stage + st_m + 64 * k, t0);
        }
    }
    noc_async_read_barrier();
    // wave 2: chunks into the pages, fills
    attnio::unstage_row(2, nh * 4, 0, gw, stage);
    const bool valid = !merge || *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + st_v) != 0;
    for (uint32_t k = 0; k < nh; ++k) {
        if (valid) {
            attnio::unstage_row(2, 4, r0 + k, aw + k * 256, stage + st_a + 512 * k);
        }
    }
    if (!valid) {  // far cache empty: A and a read as zero
        attnio::zero_l1(aw, nh * 256);
    }
    if constexpr (merge) {
        for (uint32_t k = 0; k < nh; ++k) {
            attnio::unstage_row(4, 4, k, ow + k * 512, stage + st_o + 512 * k);
            fill(rw + k * 512, *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + st_r + 64 * k), 128);
            const uint32_t a_bits =
                valid ? static_cast<uint32_t>(*reinterpret_cast<volatile tt_l1_ptr uint16_t*>(e_a[k])) << 16 : 0;
            fill(mw + k * 512, a_bits, 128);
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
