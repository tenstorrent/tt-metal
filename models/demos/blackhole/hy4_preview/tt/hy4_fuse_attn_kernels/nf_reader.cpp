// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B22 fused front norms (q_a rms + kv latent rms + k_pe split in ONE program), reader. One core group = one rms
// norm over a W = 32 * NG * TPC wide row (NG cores, TPC 32-wide tiles per core). Core k owns the tiles [k TPC, (k+1)
// TPC) of the input tensor (bf16 TILE, rows = tokens, T <= 8).
//   1. read the TPC x tiles (-> compute: per-row partial sum of squares as diag(x x^T) of the core's tiles) and the TPC
//   gamma pages (RM, 64 B = 32 values per page; built into
//      a tile whose row 0 is the gamma tile) ; KPE groups: cores k < 2 also copy input tile 16 + k to the k_pe output
//   2. exchange (the iHC fused-pre exchange-2 pattern): the core's T partial sums + a flag word (64 B slot) are written
//   to slot k of EVERY core of the group; every core waits for
//      the NG flags and builds the R tile R[t, kk] = partial sum of core kk for token t (face 0, NG <= 16); compute: rs
//      = rsqrt(R @ (1/W) + eps), out = x * rs * gamma
// CT: 0 cb_x 1 cb_g 2 cb_ones 3 cb_p2 4 cb_slot 5 cb_tot 6 cb_kpe 7 T 8 TPC 9 NG 10 MX0 11 MY0 12 MW 13 INVW_BITS 14
// KPE, then TensorAccessorArgs(x) (gamma) (kpe_out) Common: x addr, gamma addr, kpe_out addr. RT: k. cb_slot (one 4096
// B page): slots [0, NG * 64) | gamma staging at +2048 (TPC x 64 B)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_g = get_compile_time_arg_val(1);
    constexpr uint32_t cb_ones = get_compile_time_arg_val(2);
    constexpr uint32_t cb_p2 = get_compile_time_arg_val(3);
    constexpr uint32_t cb_slot = get_compile_time_arg_val(4);
    constexpr uint32_t cb_tot = get_compile_time_arg_val(5);
    constexpr uint32_t cb_kpe = get_compile_time_arg_val(6);
    constexpr uint32_t T = get_compile_time_arg_val(7);
    constexpr uint32_t TPC = get_compile_time_arg_val(8);
    constexpr uint32_t NG = get_compile_time_arg_val(9);
    constexpr uint32_t MX0 = get_compile_time_arg_val(10);
    constexpr uint32_t MY0 = get_compile_time_arg_val(11);
    constexpr uint32_t MW = get_compile_time_arg_val(12);
    constexpr uint32_t INVW = get_compile_time_arg_val(13);
    constexpr uint32_t KPE = get_compile_time_arg_val(14);
    constexpr auto x_args = TensorAccessorArgs<15>();
    constexpr auto g_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr uint32_t SW = 16;  // slot words: T data words, flag at word 15
    constexpr uint32_t FLAG = 0x5EEDF1A9u;
    static_assert(NG <= 16 && T <= 8);

    const uint32_t k = get_arg_val<uint32_t>(0);
    const auto x_acc = TensorAccessor(x_args, get_common_arg_val<uint32_t>(0), 2048);
    const auto g_acc = TensorAccessor(g_args, get_common_arg_val<uint32_t>(1), 64);
    Noc noc;
    CircularBuffer cbx(cb_x), cbg(cb_g), cbo(cb_ones), cbp2(cb_p2), cbs(cb_slot), cbt(cb_tot), cbk(cb_kpe);

    cbs.reserve_back(1);
    const uint32_t slot_base = cbs.get_write_ptr();
    volatile tt_l1_ptr uint32_t* slots = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_base);
    volatile tt_l1_ptr uint32_t* stage = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(slot_base + 2048);

    cbx.reserve_back(TPC);
    for (uint32_t i = 0; i < TPC; ++i) {
        noc.async_read(x_acc, cbx, 2048, {.page_id = k * TPC + i, .offset_bytes = 0}, {.offset_bytes = i * 2048});
        noc.async_read(g_acc, cbs, 64, {.page_id = k * TPC + i, .offset_bytes = 0}, {.offset_bytes = 2048 + 64 * i});
    }
    if constexpr (KPE) {
        if (k < 2) {
            cbk.reserve_back(1);
            noc.async_read(x_acc, cbk, 2048, {.page_id = 16 + k, .offset_bytes = 0}, {.offset_bytes = 0});
        }
    }
    // zero fills (gamma tiles, ones tile, R tile) while the reads are in flight
    cbg.reserve_back(TPC);
    cbo.reserve_back(1);
    cbt.reserve_back(1);
    noc.async_write_zeros(cbg, TPC * 2048, {.offset_bytes = 0});
    noc.async_write_zeros(cbo, 4096, {.offset_bytes = 0});
    noc.async_write_zeros(cbt, 4096, {.offset_bytes = 0});
    noc.async_read_barrier();
    cbx.push_back(TPC);
    if constexpr (KPE) {
        if (k < 2) {
            const auto o_acc = TensorAccessor(o_args, get_common_arg_val<uint32_t>(2), 2048);
            noc.async_write(cbk, o_acc, 2048, {.offset_bytes = 0}, {.page_id = k, .offset_bytes = 0});
        }
    }
    noc.write_zeros_l1_barrier();
    invalidate_l1_cache();
    // gamma tiles: row 0 = the 32 values of the page (face 0 cols 0..15 at +0, face 1 cols 16..31 at +512 B)
    volatile tt_l1_ptr uint32_t* gt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbg.get_write_ptr());
    for (uint32_t i = 0; i < TPC; ++i) {
#pragma GCC unroll 8
        for (uint32_t w = 0; w < 8; ++w) {
            gt[i * 512 + w] = stage[16 * i + w];
            gt[i * 512 + 128 + w] = stage[16 * i + 8 + w];
        }
    }
    cbg.push_back(TPC);
    // ones tile (value 1/W): rows 0..15 of faces 0 and 1
    volatile tt_l1_ptr uint32_t* on = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbo.get_write_ptr());
    for (uint32_t w = 0; w < 512; ++w) {
        on[w] = INVW;
    }
    cbo.push_back(1);

    // exchange
    cbp2.wait_front(1);
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* pp2 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbp2.get_read_ptr());
    for (uint32_t t = 0; t < T; ++t) {
        slots[k * SW + t] = pp2[(t << 4) + t];  // diag(x x^T): element (t, t) of face 0
    }
    slots[k * SW + SW - 1] = FLAG;
    cbp2.pop_front(1);
    for (uint32_t d = 1; d < NG; ++d) {
        const uint32_t kk = (k + d) % NG;
        const uint64_t dst = get_noc_addr(MX0 + (kk % MW), MY0 + kk / MW, slot_base + k * SW * 4);
        noc_async_write(slot_base + k * SW * 4, dst, SW * 4);
    }
    for (uint32_t kk = 0; kk < NG; ++kk) {
        for (uint32_t spin = 0; spin < 20000000u; ++spin) {
            invalidate_l1_cache();
            if (slots[kk * SW + SW - 1] == FLAG) {
                break;
            }
        }
    }
    volatile tt_l1_ptr uint32_t* R = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbt.get_write_ptr());
    for (uint32_t t = 0; t < T; ++t) {
#pragma GCC unroll 16
        for (uint32_t kk = 0; kk < NG; ++kk) {
            R[(t << 4) + kk] = slots[kk * SW + t];
        }
    }
    cbt.push_back(1);
    for (uint32_t kk = 0; kk < NG; ++kk) {
        slots[kk * SW + SW - 1] = 0;
    }
    noc.async_write_barrier();
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
