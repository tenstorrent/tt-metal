// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// attention-front POC, BRISC (data movement of the fused chain, the 8 cores = the 8 DRAM-bank readers of the chip):
//   x (tiny 8x32 tiles in DRAM) -> cb_x;  gamma tiles, scaler / eps / zero tiles
//   mm1 output (7 tiny tiles) -> std 32x32 tiles in cb_ys (faces 0/1 hold the 8 rows)
//   partial sums of squares (std tiles from compute) -> every core's cb_part slot, semaphore
//   normalised tiles: q tiles -> every core's cb_in0b (tiny layout) + DRAM qr, kv tiles -> DRAM kvn
//   mm2 output (16 tiny tiles = one head) -> DRAM q (std pages, faces 0/1)
// Global tile ids: bank c owns wqkv output tiles [7c, 7c+7); tiles < NQT are q, the rest kv.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "api/debug/kernel_profiler.hpp"
#include "af_groups.hpp"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_y = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ys = get_compile_time_arg_val(2);
    constexpr uint32_t cb_pout = get_compile_time_arg_val(3);
    constexpr uint32_t cb_part = get_compile_time_arg_val(4);
    constexpr uint32_t cb_sc = get_compile_time_arg_val(5);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(6);
    constexpr uint32_t cb_g = get_compile_time_arg_val(7);
    constexpr uint32_t cb_n = get_compile_time_arg_val(8);
    constexpr uint32_t cb_in0b = get_compile_time_arg_val(9);
    constexpr uint32_t cb_q = get_compile_time_arg_val(10);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(11);
    constexpr uint32_t NC = get_compile_time_arg_val(12);   // 8 cores
    constexpr uint32_t PN1 = get_compile_time_arg_val(13);  // 7 tiles of wqkv output per core
    constexpr uint32_t NQT = get_compile_time_arg_val(14);  // 40 q tiles
    constexpr uint32_t KT1 = get_compile_time_arg_val(15);  // 160 x tiles
    constexpr uint32_t PN2 = get_compile_time_arg_val(16);  // 16 tiles of wq_b output per core
    constexpr uint32_t stage_raw = get_compile_time_arg_val(17);
    constexpr uint32_t stage = stage_raw & 15;
    constexpr bool x_l1 = ((stage_raw >> 4) & 1) != 0;   // x already in the cores L1 (tensor-backed cb_x)
    constexpr bool x_std = ((stage_raw >> 5) & 1) != 0;  // x = std 32x32 tiles (faces 0 / 1 carry the 8 rows)
    constexpr bool x_mcast = ((stage_raw >> 7) & 1) != 0;
    constexpr bool x_split =
        ((stage_raw >> 6) & 1) != 0;  // x_std: every core converts 1/NC of x and sends it to the others
    constexpr uint32_t eps_bits = get_compile_time_arg_val(18);
    constexpr auto x_args = TensorAccessorArgs<19>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto qr_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr auto kv_args = TensorAccessorArgs<qr_args.next_compile_time_args_offset()>();
    constexpr auto q_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr uint32_t MT = get_compile_time_arg_val(q_args.next_compile_time_args_offset());
    constexpr uint32_t cb_x1 = get_compile_time_arg_val(q_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t cb_in0b1 = get_compile_time_arg_val(q_args.next_compile_time_args_offset() + 2);
    constexpr uint32_t NKT = get_compile_time_arg_val(q_args.next_compile_time_args_offset() + 3);
    constexpr uint32_t G = get_compile_time_arg_val(q_args.next_compile_time_args_offset() + 4);

    const uint32_t c = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t g_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t qr_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t kv_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t q_addr = get_common_arg_val<uint32_t>(4);
    // physical NOC coordinates of the NC cores: common args 5 .. 5 + 2 NC
    auto nx = [&](uint32_t d) { return get_common_arg_val<uint32_t>(5 + 2 * d); };
    auto ny = [&](uint32_t d) { return get_common_arg_val<uint32_t>(6 + 2 * d); };

    const uint32_t gt0 = c * PN1;
    // global wqkv column tile of the local tile r: G = 1 contiguous; G = 2: bank c / 2 holds 8 tiles, the bank core the
    // even ones, the helper the odd ones
    auto tpage = [&](uint32_t r) { return G == 1 ? gt0 + r : (c / G) * 8 + 2 * r + (c % G); };
    // wq_b output tile j of this core: G = 1: head c, column j; G = 2: bank c / 2, column 2 j + (c % 2) (even columns
    // stay on the bank core)
    auto qpage = [&](uint32_t j) { return G == 1 ? c * PN2 + j : (c / G) * 16 + G * j + (c % G); };
    const uint32_t nq = gt0 >= NQT ? 0 : (NQT - gt0 >= PN1 ? PN1 : NQT - gt0);
    const uint32_t rem = gt0 + nq >= NKT ? 0 : NKT - gt0 - nq;
    const uint32_t nkv = rem < PN1 - nq ? rem : PN1 - nq;
    const uint32_t ntile = nq + nkv;

    Noc noc;
    const auto x_acc = TensorAccessor(x_args, x_addr, x_std ? 2048 : 512);
    const auto g_acc = TensorAccessor(g_args, g_addr, 2048);
    const auto qr_acc = TensorAccessor(qr_args, qr_addr, 2048);
    const auto kv_acc = TensorAccessor(kv_args, kv_addr, 2048);
    const auto q_acc = TensorAccessor(q_args, q_addr, 2048);
    experimental::CB cbx(cb_x), cby(cb_y), cbys(cb_ys), cbpout(cb_pout), cbpart(cb_part), cbsc(cb_sc), cbeps(cb_eps),
        cbg(cb_g), cbn(cb_n), cbin0b(cb_in0b), cbq(cb_q), cbzero(cb_zero), cbx1(cb_x1), cbin0b1(cb_in0b1);

    volatile tt_l1_ptr uint32_t* sem_part = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    volatile tt_l1_ptr uint32_t* sem_q = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(1));
    const uint32_t sem_part_addr = get_semaphore(0), sem_q_addr = get_semaphore(1);
    volatile tt_l1_ptr uint32_t* sem_x = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(2));
    const uint32_t sem_x_addr = get_semaphore(2);

    {
        DeviceZoneScopedN("B_XLOAD");
        cbx.reserve_back(KT1);
        if constexpr (MT == 2) {
            cbx1.reserve_back(KT1);
        }
        {
            DeviceZoneScopedN("B_XEX");
            if constexpr (!x_l1) {
                if constexpr (x_std && x_split) {
                    // every core converts its 1/NC of the x tiles (faces 0 / 1 -> tiny layout) and sends the contiguous
                    // slice to the other cores
                    constexpr uint32_t XPC = KT1 / NC;
                    for (uint32_t j = 0; j < XPC; ++j) {
                        const uint32_t i = c * XPC + j;
                        noc.async_read(x_acc, cbx, 256, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = i * 512});
                        noc.async_read(
                            x_acc, cbx, 256, {.page_id = i, .offset_bytes = 512}, {.offset_bytes = i * 512 + 256});
                        if constexpr (MT == 2) {
                            noc.async_read(
                                x_acc, cbx1, 256, {.page_id = i, .offset_bytes = 256}, {.offset_bytes = i * 512});
                            noc.async_read(
                                x_acc, cbx1, 256, {.page_id = i, .offset_bytes = 768}, {.offset_bytes = i * 512 + 256});
                        }
                    }
                    noc.async_read_barrier();
                    {
                        DeviceZoneScopedN("B_XSEND");
                        const uint32_t xslice = cbx.get_write_ptr() + c * XPC * 512;
                        const uint32_t xslice1 = cbx1.get_write_ptr() + c * XPC * 512;
                        if constexpr (x_mcast) {
                            // one multicast per column strip of cores (the strips also cover idle workers: their L1 at
                            // this transient CB address is overwritten, nothing lives there)
                            const uint32_t nrect = get_common_arg_val<uint32_t>(5 + 2 * NC);
                            for (uint32_t rr = 0; rr < nrect; ++rr) {
                                const uint32_t rx = get_common_arg_val<uint32_t>(6 + 2 * NC + 4 * rr);
                                const uint32_t ry0 = get_common_arg_val<uint32_t>(7 + 2 * NC + 4 * rr);
                                const uint32_t ry1 = get_common_arg_val<uint32_t>(8 + 2 * NC + 4 * rr);
                                const bool self_in = (rx == nx(c)) && (ny(c) >= ry0) && (ny(c) <= ry1);
                                const uint32_t nd =
                                    get_common_arg_val<uint32_t>(9 + 2 * NC + 4 * rr) - (self_in ? 1 : 0);
                                noc_async_write_multicast(
                                    xslice, get_noc_multicast_addr(rx, ry0, rx, ry1, xslice), XPC * 512, nd);
                                if constexpr (MT == 2) {
                                    noc_async_write_multicast(
                                        xslice1, get_noc_multicast_addr(rx, ry0, rx, ry1, xslice1), XPC * 512, nd);
                                }
                            }
                        } else {
                            for (uint32_t kk = 0; kk < NC; ++kk) {
                                const uint32_t d = (c + kk >= NC) ? c + kk - NC : c + kk;
                                if (d != c) {
                                    noc_async_write(xslice, get_noc_addr(nx(d), ny(d), xslice), XPC * 512);
                                    if constexpr (MT == 2) {
                                        noc_async_write(xslice1, get_noc_addr(nx(d), ny(d), xslice1), XPC * 512);
                                    }
                                }
                            }
                        }
                        noc_async_write_barrier();
                        for (uint32_t kk = 0; kk < NC; ++kk) {
                            const uint32_t d = (c + kk >= NC) ? c + kk - NC : c + kk;
                            if (d != c) {
                                noc_semaphore_inc(get_noc_addr(nx(d), ny(d), get_semaphore(5 + c)), 1);
                            }
                        }
                        noc_async_atomic_barrier();
                    }
                } else if constexpr (x_std) {
                    for (uint32_t i = 0; i < KT1; ++i) {
                        noc.async_read(x_acc, cbx, 256, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = i * 512});
                        noc.async_read(
                            x_acc, cbx, 256, {.page_id = i, .offset_bytes = 512}, {.offset_bytes = i * 512 + 256});
                        if constexpr (MT == 2) {
                            noc.async_read(
                                x_acc, cbx1, 256, {.page_id = i, .offset_bytes = 256}, {.offset_bytes = i * 512});
                            noc.async_read(
                                x_acc, cbx1, 256, {.page_id = i, .offset_bytes = 768}, {.offset_bytes = i * 512 + 256});
                        }
                    }
                } else {
                    for (uint32_t i = 0; i < KT1; ++i) {
                        noc.async_read(x_acc, cbx, 512, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = i * 512});
                    }
                }
            }
            noc.async_read_barrier();
            noc.async_write_barrier();
            if constexpr (x_std && x_split) {
                // push the x slices to the compute kernel in K order as they arrive (the matmul starts on the first K
                // blocks)
                DeviceZoneScopedN("B_XWAIT");
                constexpr uint32_t XPC = KT1 / NC;
                for (uint32_t s = 0; s < NC; ++s) {
                    if (s != c) {
                        volatile tt_l1_ptr uint32_t* sx =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(5 + s));
                        noc_semaphore_wait(sx, 1);
                        *sx = 0;
                    }
                    cbx.push_back(XPC);
                    if constexpr (MT == 2) {
                        cbx1.push_back(XPC);
                    }
                }
            }
        }
        if constexpr (!(x_std && x_split)) {
            cbx.push_back(KT1);
            if constexpr (MT == 2) {
                cbx1.push_back(KT1);
            }
        }
        if constexpr (stage >= 2) {
            // gamma tiles of this core's wqkv output tiles
            cbg.reserve_back(PN1);
            for (uint32_t r = 0; r < ntile; ++r) {
                noc.async_read(g_acc, cbg, 2048, {.page_id = tpage(r), .offset_bytes = 0}, {.offset_bytes = r * 2048});
            }
            // zero std tiles of cb_ys, one zero tile
            cbys.reserve_back(PN1);
            for (uint32_t r = 0; r < PN1; ++r) {
                noc.async_write_zeros(cbys, 2048, {.offset_bytes = r * 2048});
            }
            cbzero.reserve_back(1);
            noc.async_write_zeros(cbzero, 2048, {.offset_bytes = 0});
            // scaler tile: 1.0 in row 0 of every face (bf16), zeros elsewhere
            cbsc.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbsc.get_write_ptr());
                for (uint32_t i = 0; i < 512; ++i) {
                    p[i] = 0;
                }
                for (uint32_t f = 0; f < 4; ++f) {
                    for (uint32_t cc = 0; cc < 8; ++cc) {
                        p[f * 128 + cc] = 0x3F803F80u;
                    }
                }
            }
            cbsc.push_back(1);
            // eps tile (generate_bcast_col_scalar): bf16 eps in column 0 of faces 0 and 2
            cbeps.reserve_back(1);
            {
                volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbeps.get_write_ptr());
                for (uint32_t i = 0; i < 1024; ++i) {
                    p[i] = 0;
                }
                const uint16_t e = eps_bits >> 16;
                for (uint32_t k = 0; k < 4; k += 2) {
                    const uint32_t idx = k << 8;
                    for (uint32_t j = 0; j < 256; j += 16) {
                        p[idx + j] = e;
                    }
                }
            }
            cbeps.push_back(1);
            cbpart.reserve_back(2 * NC);
            cbin0b.reserve_back(NQT);
            if constexpr (MT == 2) {
                cbin0b1.reserve_back(NQT);
            }
        }
        noc.async_read_barrier();
        noc.async_write_barrier();
        if constexpr (stage >= 2) {
            cbg.push_back(PN1);
            cbzero.push_back(1);
        }
    }

    const uint32_t my_x = nx(c), my_y = ny(c);

    {
        DeviceZoneScopedN("B_CONV");
        const uint32_t y_ptr = cby.get_read_ptr();
        if constexpr (stage == 1) {
            cby.wait_front(PN1 * MT);
            // test output: all 56 tiles of y go to the qr tensor (pages 0..55 of a 56-page tensor, std layout faces
            // 0/1)
            for (uint32_t r = 0; r < PN1; ++r) {
                noc.async_write(cby, qr_acc, 256, {.offset_bytes = r * 512}, {.page_id = gt0 + r, .offset_bytes = 0});
                noc.async_write(
                    cby, qr_acc, 256, {.offset_bytes = r * 512 + 256}, {.page_id = gt0 + r, .offset_bytes = 512});
            }
            noc.async_write_barrier();
            cby.pop_front(PN1 * MT);
            return;
        }
        const uint32_t ys_ptr = cbys.get_write_ptr();
        {
            uint32_t r = 0, waited = 0;
            for (uint32_t g = 0; g < af::NG1; ++g) {  // convert group by group while the matmul runs on
                waited += af::GRP1[g] * MT;
                cby.wait_front(waited);
                for (uint32_t cc = 0; cc < af::GRP1[g]; ++cc, ++r) {
                    const uint32_t ya = af::out_idx(af::GRP1, MT, r, 0) * 512;
                    noc_async_write(y_ptr + ya, get_noc_addr(my_x, my_y, ys_ptr + r * 2048), 256);
                    noc_async_write(y_ptr + ya + 256, get_noc_addr(my_x, my_y, ys_ptr + r * 2048 + 512), 256);
                    if constexpr (MT == 2) {  // rows 8..15 of the std tile: second row tile
                        const uint32_t yb = af::out_idx(af::GRP1, MT, r, 1) * 512;
                        noc_async_write(y_ptr + yb, get_noc_addr(my_x, my_y, ys_ptr + r * 2048 + 256), 256);
                        noc_async_write(y_ptr + yb + 256, get_noc_addr(my_x, my_y, ys_ptr + r * 2048 + 768), 256);
                    }
                }
            }
        }
        noc_async_write_barrier();
        cby.pop_front(PN1 * MT);
        cbys.push_back(PN1);
    }

    {
        DeviceZoneScopedN("B_PARTIAL");
        const uint32_t part_base = cbpart.get_write_ptr();
        const uint32_t npo = (nq > 0 ? 1 : 0) + (nkv > 0 ? 1 : 0);
        cbpout.wait_front(npo);
        uint32_t po_ptr = cbpout.get_read_ptr();
        for (uint32_t kk = 0; kk < NC; ++kk) {
            const uint32_t d = (c + kk >= NC) ? c + kk - NC : c + kk;
            uint32_t p = po_ptr;
            if (nq > 0) {
                noc_async_write(p, get_noc_addr(nx(d), ny(d), part_base + c * 2048), 2048);
                p += 2048;
            }
            if (nkv > 0) {
                noc_async_write(p, get_noc_addr(nx(d), ny(d), part_base + (NC + c) * 2048), 2048);
            }
        }
        noc_async_write_barrier();
        for (uint32_t kk = 0; kk < NC; ++kk) {
            const uint32_t d = (c + kk >= NC) ? c + kk - NC : c + kk;
            noc_semaphore_inc(get_noc_addr(nx(d), ny(d), sem_part_addr), 1);
        }
        noc_async_atomic_barrier();
        noc_semaphore_wait(sem_part, NC);
        *sem_part = 0;
        cbpart.push_back(2 * NC);
        cbpout.pop_front(npo);
    }

    {
        DeviceZoneScopedN("B_NORMDIST");
        if (ntile > 0) {
            cbn.wait_front(1);
        }
        const uint32_t n_ptr = cbn.get_read_ptr();
        const uint32_t in0b_base = cbin0b.get_write_ptr();
        const uint32_t in0b1_base = MT == 2 ? cbin0b1.get_write_ptr() : 0;
        if (nq > 0) {
            cbn.wait_front(nq);
        }
        // per destination: one set_state, then the 2 (4 at MT = 2) face writes of every local q tile
        for (uint32_t kk = 0; kk < NC && nq > 0; ++kk) {
            const uint32_t d = (c + kk >= NC) ? c + kk - NC : c + kk;
            noc_async_write_one_packet_set_state<false>(get_noc_addr(nx(d), ny(d), 0), 256);
            for (uint32_t r = 0; r < nq; ++r) {
                const uint32_t gt = tpage(r);
                const uint32_t src = n_ptr + r * 2048;
                noc_async_write_one_packet_with_state<false>(src, in0b_base + gt * 512);
                noc_async_write_one_packet_with_state<false>(src + 512, in0b_base + gt * 512 + 256);
                if constexpr (MT == 2) {
                    noc_async_write_one_packet_with_state<false>(src + 256, in0b1_base + gt * 512);
                    noc_async_write_one_packet_with_state<false>(src + 768, in0b1_base + gt * 512 + 256);
                }
            }
        }
        for (uint32_t r = 0; r < nq; ++r) {
            const uint32_t gt = tpage(r);
            constexpr uint32_t FB = MT == 2 ? 512 : 256;  // bytes of a face written to DRAM: rows 0..15 / 0..7
            noc.async_write(cbn, qr_acc, FB, {.offset_bytes = r * 2048}, {.page_id = gt, .offset_bytes = 0});
            noc.async_write(cbn, qr_acc, FB, {.offset_bytes = r * 2048 + 512}, {.page_id = gt, .offset_bytes = 512});
        }
        for (uint32_t r = nq; r < ntile; ++r) {
            cbn.wait_front(r + 1);
            const uint32_t gt = tpage(r) - NQT;
            constexpr uint32_t FB = MT == 2 ? 512 : 256;
            noc.async_write(cbn, kv_acc, FB, {.offset_bytes = r * 2048}, {.page_id = gt, .offset_bytes = 0});
            noc.async_write(cbn, kv_acc, FB, {.offset_bytes = r * 2048 + 512}, {.page_id = gt, .offset_bytes = 512});
        }
        noc_async_write_barrier();
        noc.async_write_barrier();
        if (ntile > 0) {
            cbn.pop_front(ntile);
        }
    }
    if constexpr (stage == 2) {
        return;
    }
    {
        DeviceZoneScopedN("B_SEMQ");
        for (uint32_t kk = 0; kk < NC; ++kk) {
            const uint32_t d = (c + kk >= NC) ? c + kk - NC : c + kk;
            noc_semaphore_inc(get_noc_addr(nx(d), ny(d), sem_q_addr), 1);
        }
        noc_async_atomic_barrier();
        noc_semaphore_wait(sem_q, NC);
        *sem_q = 0;
        cbin0b.push_back(NQT);
        if constexpr (MT == 2) {
            cbin0b1.push_back(NQT);
        }
    }

    {
        DeviceZoneScopedN("B_OUT");
        uint32_t j = 0, waited = 0;
        for (uint32_t g = 0; g < af::NG2; ++g) {  // write group by group while the matmul runs on
            waited += af::GRP2[g] * MT;
            cbq.wait_front(waited);
            for (uint32_t cc = 0; cc < af::GRP2[g]; ++cc, ++j) {
                const uint32_t qa = af::out_idx(af::GRP2, MT, j, 0) * 512;
                noc.async_write(cbq, q_acc, 256, {.offset_bytes = qa}, {.page_id = qpage(j), .offset_bytes = 0});
                noc.async_write(
                    cbq, q_acc, 256, {.offset_bytes = qa + 256}, {.page_id = qpage(j), .offset_bytes = 512});
                if constexpr (MT == 2) {
                    const uint32_t qb = af::out_idx(af::GRP2, MT, j, 1) * 512;
                    noc.async_write(cbq, q_acc, 256, {.offset_bytes = qb}, {.page_id = qpage(j), .offset_bytes = 256});
                    noc.async_write(
                        cbq, q_acc, 256, {.offset_bytes = qb + 256}, {.page_id = qpage(j), .offset_bytes = 768});
                }
            }
        }
        noc.async_write_barrier();
        cbq.pop_front(PN2 * MT);
    }
}
