// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention output prep for the T tokens of a multi-token verify pass in one program, reader: per token the
// same reads and pages as attn_out_reader.cpp's verify forms (local: rows, batch b; global: merge + far_grp), one
// token after the other into 2-page CBs, so token t + 1's reads overlap token t's compute.
// Runtime args: proj, h0, then per token: attn, prow, and batch (local) or o_near, r, valid, kv group, far row
// (merge).
#include "attn_io.hpp"

namespace {
FORCE_INLINE void fill(uint32_t dst, uint32_t bits, uint32_t n) {
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    for (uint32_t i = 0; i < n; ++i) {
        p[i] = bits;
    }
}
}  // namespace

constexpr bool proj_dram = get_named_compile_time_arg_val("proj_dram");

void kernel_main() {
    constexpr uint32_t nh = get_named_compile_time_arg_val("nh");
    constexpr uint32_t gate_tile = get_named_compile_time_arg_val("gate_tile");
    constexpr uint32_t merge = get_named_compile_time_arg_val("merge");  // global: far_grp partial + near merge
    constexpr uint32_t tokens = get_named_compile_time_arg_val("tokens");
    constexpr uint32_t cb_a = 0, cb_g = 1, cb_m = 2, cb_r = 3, cb_o = 4, cb_stage = 24;
    uint32_t a = 0;
    const uint32_t proj = get_arg_val<uint32_t>(a++);
    const uint32_t h0 = get_arg_val<uint32_t>(a++);
    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);
    constexpr uint32_t st_a = 128 * nh * 4, st_v = st_a + 512 * nh, st_o = st_v + 64, st_r = st_o + 512 * nh;
    constexpr uint32_t st_m = st_r + 64 * nh;

    for (uint32_t t = 0; t < tokens; ++t) {
        const uint32_t attn = get_arg_val<uint32_t>(a++);
        const uint32_t prow = get_arg_val<uint32_t>(a++);
        uint32_t t0, r0, o_near = 0, r_addr = 0, valid_addr = 0, g = 0;
        if constexpr (merge) {
            o_near = get_arg_val<uint32_t>(a++);
            r_addr = get_arg_val<uint32_t>(a++);
            valid_addr = get_arg_val<uint32_t>(a++);
            g = get_arg_val<uint32_t>(a++);
            t0 = (h0 / nh) * 5;
            r0 = get_arg_val<uint32_t>(a++);
        } else {
            t0 = (get_arg_val<uint32_t>(a++) * 8 + h0 / nh) * 4;
            r0 = 0;
        }
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_g, 1);
        const uint32_t aw = get_write_ptr(cb_a), gw = get_write_ptr(cb_g);
        uint32_t mw = 0, rw = 0, ow = 0, e_a[nh];
        attnio::zero_l1(aw + nh * 256, 2048 - nh * 256);
        attnio::zero_l1(gw + nh * 256, 2048 - nh * 256);
        attnio::stage_row<proj_dram>(proj, 2048, 2, gate_tile + h0 * 4, nh * 4, prow, stage);
        for (uint32_t k = 0; k < nh; ++k) {
            attnio::stage_row(attn, 2048, 2, t0, 4, r0 + k, stage + st_a + 512 * k);
        }
        if constexpr (merge) {
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
                noc_async_read(rg.get_noc_addr(g, k * 64), stage + st_r + 64 * k, 64);
                e_a[k] = attnio::stage_elem_block(attn, 2048, 2, r0 + k, 128, stage + st_m + 64 * k, t0);
            }
        }
        noc_async_read_barrier();
        attnio::unstage_row(2, nh * 4, prow, gw, stage);
        const bool valid = !merge || *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage + st_v) != 0;
        for (uint32_t k = 0; k < nh; ++k) {
            if (valid) {
                attnio::unstage_row(2, 4, r0 + k, aw + k * 256, stage + st_a + 512 * k);
            }
        }
        if (!valid) {
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
}
