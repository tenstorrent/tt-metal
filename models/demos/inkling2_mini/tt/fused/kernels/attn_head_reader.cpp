// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Fused attention glue, head role, reader. Per item (one head): nlag rows of 128 projection columns -> dense bf16
// pages (cb_x), their taps / scale (dense fp32 pages, cb_t), and with `dyn` the per-token taps of global layers as
// dense fp32 pages filled with d_l (cb_d). Plus the reduce scaler, (has_tau) the tau tile and (mass) the
// far-cache mass page.
// slim: read only the bytes the compute uses and build the constant pages in L1 (same CB bytes as reading the
// whole pages): taps 512 of 4096 bytes per page, tau its first 64 bytes; the scaler and the mass page are built.
// The DRAM is busy with the expert stream here, so fewer bytes ends the reads sooner.
// pre (with slim; attn.py PREP_PRE): runtime args after the fixed five: scratch (an L1 shard on this core, the same
// address on every prep core) and use: this call's taps / tau are in the scratch (attn_prep_fetch.cpp, in the w_all
// program): taps of item i lag l at 512 (i nlag + l), tau's first 64 bytes at 2048; a 4 KB zero block at 4096. Then
// (projection in L1) the reader issues few NOC reads: zeros from that block, the projection rows straight from L1
// (no staging wave), the rest as before. Same bytes in the CBs.
#include "attn_io.hpp"

constexpr bool proj_dram = get_named_compile_time_arg_val("proj_dram");  // projection output in DRAM (else L1)

void kernel_main() {
    constexpr uint32_t nitems = get_named_compile_time_arg_val("nitems");
    constexpr uint32_t nlag = get_named_compile_time_arg_val("nlag");
    constexpr uint32_t dyn = get_named_compile_time_arg_val("dyn");
    constexpr uint32_t has_tau = get_named_compile_time_arg_val("has_tau");
    constexpr uint32_t mass = get_named_compile_time_arg_val("mass");  // far-cache mass chunk page (cb_mass)
    constexpr uint32_t slim = get_named_compile_time_arg_val("slim");
    constexpr uint32_t pre = get_named_compile_time_arg_val("pre");
    static_assert(!pre || slim, "pre needs slim");
    static_assert(nitems * nlag * 512 <= 2048, "scratch taps");
    constexpr uint32_t cb_x = 0, cb_t = 1, cb_d = 2, cb_scaler = 3, cb_tau = 4, cb_mass = 5, cb_stage = 24;
    constexpr uint32_t n = nitems * nlag;

    uint32_t a = 0;
    const uint32_t proj = get_arg_val<uint32_t>(a++);
    const uint32_t taps = get_arg_val<uint32_t>(a++);
    const uint32_t scaler = get_arg_val<uint32_t>(a++);
    const uint32_t tau = get_arg_val<uint32_t>(a++);
    const uint32_t mass_addr = get_arg_val<uint32_t>(a++);
    uint32_t scratch = 0, use = 0;
    if constexpr (pre) {
        scratch = get_arg_val<uint32_t>(a++);
        use = get_arg_val<uint32_t>(a++);
    }
    const uint32_t stage = align_power_of_2(get_write_ptr(cb_stage), attnio::ALIGN);
    const bool fast = pre && use && !proj_dram;
    auto zero = [&](uint32_t dst, uint32_t bytes) {
        if (fast) {
            attnio::zero_from(scratch + 4096, 4096, dst, bytes);
        } else {
            attnio::zero_l1(dst, bytes);
        }
    };

    cb_reserve_back(cb_scaler, 1);
    if constexpr (has_tau) {
        cb_reserve_back(cb_tau, 1);
    }
    if constexpr (mass) {
        cb_reserve_back(cb_mass, 1);
    }
    if constexpr (slim) {
        zero(get_write_ptr(cb_scaler), 4096);             // built after the wave 1 barrier
        if constexpr (has_tau) {                          // tau at [0, 0]: the compute uses f[0, 0] only
            zero(get_write_ptr(cb_tau) + 64, 4096 - 64);
            const InterleavedAddrGen<true> tau_a = {.bank_base_address = tau, .page_size = 4096};
            noc_async_read(use ? get_noc_addr(scratch + 2048) : tau_a.get_noc_addr(0, 0), get_write_ptr(cb_tau), 64);
        }
        if constexpr (mass) {
            zero(get_write_ptr(cb_mass), 4096);
        }
    } else {
        attnio::read_pages(scaler, 4096, 0, 1, get_write_ptr(cb_scaler));
        if constexpr (has_tau) {
            attnio::read_pages(tau, 4096, 0, 1, get_write_ptr(cb_tau));
        }
        if constexpr (mass) {
            attnio::read_pages(mass_addr, 4096, 0, 1, get_write_ptr(cb_mass));
        }
    }
    cb_reserve_back(cb_t, n);
    cb_reserve_back(cb_x, n);
    const uint32_t xw = get_write_ptr(cb_x);
    const uint32_t tw = get_write_ptr(cb_t);
    // wave 1: zeros, taps, the staged row blocks and (dyn) the d block - one read barrier
    if (fast) {  // the rows land in bytes 0..255 of each page
        for (uint32_t k = 0; k < n; ++k) {
            zero(xw + k * 2048 + 256, 2048 - 256);
        }
    } else {
        attnio::zero_l1(xw, n * 2048);
    }
    uint32_t t0s[nitems], r0s[nitems], dcol = 0, d_elem = 0, dw = 0;
    for (uint32_t i = 0; i < nitems; ++i) {
        t0s[i] = get_arg_val<uint32_t>(a++);               // projection tile of the head's first column
        r0s[i] = get_arg_val<uint32_t>(a++);               // window row of lag 0
        const uint32_t page = get_arg_val<uint32_t>(a++);  // first taps page
        if constexpr (slim) {  // a head's 128 taps are the page's first 512 bytes, the rest is zero
            const InterleavedAddrGen<true> taps_a = {.bank_base_address = taps, .page_size = 4096};
            for (uint32_t l = 0; l < nlag; ++l) {
                const uint32_t dst = tw + (i * nlag + l) * 4096;
                zero(dst + 512, 4096 - 512);
                const uint64_t src =
                    use ? get_noc_addr(scratch + (i * nlag + l) * 512) : taps_a.get_noc_addr(page + l, 0);
                noc_async_read(src, dst, 512);
            }
        } else {
            attnio::read_pages(taps, 4096, page, nlag, tw + i * nlag * 4096);
        }
        for (uint32_t l = 0; l < nlag; ++l) {
            if (fast) {
                attnio::l1_row(proj, 2048, 2, t0s[i], 4, r0s[i] + l, xw + (i * nlag + l) * 2048);
            } else {
                attnio::stage_row<proj_dram>(proj, 2048, 2, t0s[i], 4, r0s[i] + l, stage + (i * nlag + l) * 512);
            }
        }
        if constexpr (dyn) {                    // nitems == 1
            dcol = get_arg_val<uint32_t>(a++);  // column of d_0 (lags follow) in row r0
            cb_reserve_back(cb_d, nlag);
            dw = get_write_ptr(cb_d);
            zero(dw, nlag * 4096);
            d_elem = attnio::stage_elem_block<proj_dram>(proj, 2048, 2, r0s[i], dcol, stage + n * 512);
        }
    }
    noc_async_read_barrier();
    if constexpr (slim) {  // scaler: 1.0 on tile rows 0 and 16 (faces 0..3, row 0 of each); mass: 1.0 at [0, 0]
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_scaler));
        for (uint32_t f = 0; f < 4; ++f) {
            for (uint32_t k = 0; k < 16; ++k) {
                p[f * 256 + k] = 0x3f800000u;
            }
        }
        if constexpr (mass) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_mass)) = 0x3f800000u;
        }
    }
    // wave 2: chunks into the zeroed pages, d pages filled
    for (uint32_t i = 0; i < nitems && !fast; ++i) {
        for (uint32_t l = 0; l < nlag; ++l) {
            const uint32_t k = i * nlag + l;
            attnio::unstage_row(2, 4, r0s[i] + l, xw + k * 2048, stage + k * 512);
        }
    }
    if constexpr (dyn) {
        volatile tt_l1_ptr uint16_t* d16 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(d_elem);
        for (uint32_t l = 0; l < nlag; ++l) {
            const uint32_t bits = static_cast<uint32_t>(d16[l]) << 16;  // bf16 -> fp32
            volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dw + l * 4096);
            for (uint32_t k = 0; k < 128; ++k) {
                p[k] = bits;
            }
        }
        cb_push_back(cb_d, nlag);
    }
    noc_async_read_barrier();
    cb_push_back(cb_scaler, 1);
    if constexpr (has_tau) {
        cb_push_back(cb_tau, 1);
    }
    if constexpr (mass) {
        cb_push_back(cb_mass, 1);
    }
    cb_push_back(cb_t, n);
    cb_push_back(cb_x, n);
}
