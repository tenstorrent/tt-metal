// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B40 (HY4_B40_DERIVE=1): the integer / geometry half of SpecDerive.derive as ONE one-core program, bit-identical
// to the stock ~50 tiny elementwise ops. Inputs (per device): p = fp32 TILE [1,1,1,1] (position of row 0, integer
// valued), oc = fp32 TILE [1,1,1,1] (this device's column id). Outputs (64 B pages, per device): wr (int32 x16), kvcnt
// (uint32 x16), colinfo (uint32 x16), ih / il (uint32 x S: position // 1024, % 1024 of row j), upd_j (int32, one page
// each, j < S). CT: 0 CB, 1 S, 2 CH, 3 SL, 4 B2; then TensorAccessorArgs(p), (oc), (wr), (kv), (ci), (ih), (il),
// (upd0..upd3). Common: addresses in the same order.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

static inline int32_t f2i(uint32_t bits) {  // non-negative integer valued fp32 (< 2^24) -> int
    if (bits == 0u) {
        return 0;
    }
    const uint32_t e = (bits >> 23) & 0xFFu;
    const uint32_t m = (bits & 0x7FFFFFu) | 0x800000u;
    if (e >= 150u) {
        return (int32_t)(m << (e - 150u));
    }
    return (int32_t)(m >> (150u - e));
}

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr int32_t CH = get_compile_time_arg_val(2);
    constexpr int32_t SL = get_compile_time_arg_val(3);
    constexpr int32_t B2 = get_compile_time_arg_val(4);
    constexpr auto p_args = TensorAccessorArgs<5>();
    constexpr auto oc_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr auto wr_args = TensorAccessorArgs<oc_args.next_compile_time_args_offset()>();
    constexpr auto kv_args = TensorAccessorArgs<wr_args.next_compile_time_args_offset()>();
    constexpr auto ci_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto ih_args = TensorAccessorArgs<ci_args.next_compile_time_args_offset()>();
    constexpr auto il_args = TensorAccessorArgs<ih_args.next_compile_time_args_offset()>();
    constexpr auto up_args =
        TensorAccessorArgs<il_args.next_compile_time_args_offset()>();  // shared by the S upd tensors (same layout)
    // B47d (HY4_B47D_ROPE): optional rope block appended after the upd accessor args (the ih / il outputs are then
    // unused):
    //   CT: ROPE, FIRST, PMAX, then TensorAccessorArgs(cosT) (sinT) (cos_out) (sin_out) (c128_out) (s128_out)
    constexpr uint32_t ROPE = get_compile_time_arg_val(up_args.next_compile_time_args_offset());
    constexpr uint32_t FIRST = get_compile_time_arg_val(up_args.next_compile_time_args_offset() + 1);
    constexpr int32_t PMAX = get_compile_time_arg_val(up_args.next_compile_time_args_offset() + 2);
    constexpr auto ct_args = TensorAccessorArgs<up_args.next_compile_time_args_offset() + 3>();
    constexpr auto st_args = TensorAccessorArgs<ct_args.next_compile_time_args_offset()>();
    constexpr auto co_args = TensorAccessorArgs<st_args.next_compile_time_args_offset()>();
    constexpr auto so_args = TensorAccessorArgs<co_args.next_compile_time_args_offset()>();
    constexpr auto c1_args = TensorAccessorArgs<so_args.next_compile_time_args_offset()>();
    constexpr auto s1_args = TensorAccessorArgs<c1_args.next_compile_time_args_offset()>();
    Noc noc;
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    // slots of 16 words: 0 p, 1 oc, 2 wr, 3 kv, 4 ci, 5 ih, 6 il, 7..10 upd_j
    for (uint32_t i = 0; i < 16u * 11u; ++i) {
        sc[i] = 0u;
    }
    const auto p_acc = TensorAccessor(p_args, get_common_arg_val<uint32_t>(0), 4096);
    const auto oc_acc = TensorAccessor(oc_args, get_common_arg_val<uint32_t>(1), 4096);
    noc.async_read(p_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read(oc_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 64});
    noc.async_read_barrier();
    invalidate_l1_cache();
    const int32_t p = f2i(sc[0]);
    const int32_t oc = f2i(sc[16]);
    volatile tt_l1_ptr int32_t* wr = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + 2 * 64);
    volatile tt_l1_ptr uint32_t* kv = sc + 3 * 16;
    volatile tt_l1_ptr uint32_t* ci = sc + 4 * 16;
    volatile tt_l1_ptr uint32_t* ih = sc + 5 * 16;
    volatile tt_l1_ptr uint32_t* il = sc + 6 * 16;
    const int32_t n0 = p + 1;
    if constexpr (ROPE) {
        // rows of the (bf16 final) cos / sin tables at positions p + j (clamped), 128 B each: cos rows at +1024, sin
        // rows at +1536
        const auto ct_acc = TensorAccessor(ct_args, get_common_arg_val<uint32_t>(11), 128);
        const auto st_acc = TensorAccessor(st_args, get_common_arg_val<uint32_t>(12), 128);
        for (int32_t j = 0; j < (int32_t)S; ++j) {
            int32_t pv = p + j;
            pv = pv > PMAX ? PMAX : pv;
            noc.async_read(
                ct_acc, cb, 128, {.page_id = (uint32_t)pv, .offset_bytes = 0}, {.offset_bytes = 1024u + j * 128u});
            noc.async_read(
                st_acc, cb, 128, {.page_id = (uint32_t)pv, .offset_bytes = 0}, {.offset_bytes = 1536u + j * 128u});
        }
    }
    for (int32_t j = 0; j < 16; ++j) {
        wr[j] = -1;
        kv[j] = 0u;
        ci[j] = 0u;
    }
    for (int32_t j = 0; j < (int32_t)S; ++j) {
        const int32_t pv = p + j;
        const int32_t h5 = pv / CH, l5 = pv - h5 * CH;
        const int32_t col = l5 / SL;
        const int32_t loc = h5 * SL + (l5 - col * SL);
        const int32_t e = (col == oc) ? loc : -1;
        wr[j] = e;
        volatile tt_l1_ptr int32_t* up = reinterpret_cast<volatile tt_l1_ptr int32_t*>(base + (7 + j) * 64);
        up[0] = e;
        const int32_t n = pv + 1;
        const int32_t fu = n / CH, rem = n - fu * CH;
        int32_t d = rem - oc * SL;
        d = d < 0 ? 0 : (d > SL ? SL : d);
        kv[j] = (uint32_t)(fu * SL + d);
        ih[j] = (uint32_t)(pv / B2);
        il[j] = (uint32_t)(pv - (pv / B2) * B2);
    }
    // colinfo: [0] = own column, [1] = n0, [2 + j] = n0 + j (j < S), rest 0 (verify rows only: S > 1)
    ci[0] = (uint32_t)oc;
    ci[1] = (uint32_t)n0;
    for (int32_t j = 0; S > 1 && j < (int32_t)S && 2 + j < 16; ++j) {
        ci[2 + j] = (uint32_t)(n0 + j);
    }
    const auto wr_acc = TensorAccessor(wr_args, get_common_arg_val<uint32_t>(2), 64);
    const auto kv_acc = TensorAccessor(kv_args, get_common_arg_val<uint32_t>(3), 64);
    const auto ci_acc = TensorAccessor(ci_args, get_common_arg_val<uint32_t>(4), 64);
    const auto ih_acc = TensorAccessor(ih_args, get_common_arg_val<uint32_t>(5), 64);
    const auto il_acc = TensorAccessor(il_args, get_common_arg_val<uint32_t>(6), 64);
    noc.async_write(cb, wr_acc, 64, {.offset_bytes = 2 * 64}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write(cb, kv_acc, 64, {.offset_bytes = 3 * 64}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write(cb, ci_acc, 64, {.offset_bytes = 4 * 64}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write(cb, ih_acc, 64, {.offset_bytes = 5 * 64}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write(cb, il_acc, 64, {.offset_bytes = 6 * 64}, {.page_id = 0, .offset_bytes = 0});
    for (uint32_t j = 0; j < S; ++j) {
        const auto u_acc = TensorAccessor(up_args, get_common_arg_val<uint32_t>(7 + j), 64);
        noc.async_write(cb, u_acc, 64, {.offset_bytes = (7 + j) * 64}, {.page_id = 0, .offset_bytes = 0});
    }
    if constexpr (ROPE) {
        noc.async_read_barrier();
        invalidate_l1_cache();
        // the persistent tabs are zero-initialised and their padding rows are never written: only the 32 B face-row
        // pieces of rows < S are (cos/sin rows are used from L1 in place)
        volatile tt_l1_ptr uint32_t* ones = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + 2048);
        for (uint32_t w = 0; w < 8; ++w) {
            ones[w] = 0x3F803F80u;
        }
        const auto co_acc = TensorAccessor(co_args, get_common_arg_val<uint32_t>(13), 2048);
        const auto so_acc = TensorAccessor(so_args, get_common_arg_val<uint32_t>(14), 2048);
        const auto c1_acc = TensorAccessor(c1_args, get_common_arg_val<uint32_t>(15), 2048);
        const auto s1_acc = TensorAccessor(s1_args, get_common_arg_val<uint32_t>(16), 2048);
        for (uint32_t j = 0; j < S; ++j) {
            for (uint32_t k = 0; k < 4; ++k) {  // 16-column chunks of the 64-wide row: tile k>>1, face k&1, tile row j
                const uint32_t tile = k >> 1, dst = (k & 1) * 512u + j * 32u;
                const uint32_t cs = 1024u + j * 128u + k * 32u, ss = 1536u + j * 128u + k * 32u;
                noc.async_write(cb, co_acc, 32, {.offset_bytes = cs}, {.page_id = tile, .offset_bytes = dst});
                noc.async_write(cb, so_acc, 32, {.offset_bytes = ss}, {.page_id = tile, .offset_bytes = dst});
                noc.async_write(
                    cb, c1_acc, 32, {.offset_bytes = cs}, {.page_id = (FIRST ? 0u : 2u) + tile, .offset_bytes = dst});
                noc.async_write(
                    cb,
                    c1_acc,
                    32,
                    {.offset_bytes = 2048u},
                    {.page_id = (FIRST ? 2u : 0u) + tile, .offset_bytes = dst});
                noc.async_write(
                    cb, s1_acc, 32, {.offset_bytes = ss}, {.page_id = (FIRST ? 0u : 2u) + tile, .offset_bytes = dst});
            }
        }
    }
    noc.async_write_barrier();
}
