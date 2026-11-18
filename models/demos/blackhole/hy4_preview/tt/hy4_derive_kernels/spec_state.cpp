// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B40 (HY4_B40_TAIL=1): next-round state of one user as ONE one-core program, bit-identical to the glue at the end
// of MTPRoundTail.run (~25 tiny ops):
//   parked (park != 0): d_row = -1 x16, tok = [PAD, 0, ...], pos = parkvec (0 without HY4_PFX);  else d_row =
//   [d_1..d_k, -1 ...], tok = [nxt, d_1..d_k], pos = p_new.
// Inputs (per device): d_1..d_k fp32 TILE [1,1,1,1] (k accessors), nxt, p_new, park, parkvec (fp32 TILE; parkvec only
// if HASPV). Outputs: d_row fp32 TILE, tok uint32 RM [1,S], pos fp32 TILE. CT: 0 CB, 1 S, 2 k, 3 PAD, 4 HASPV;
// TensorAccessorArgs(d_1..d_k), (nxt), (p_new), (park), [parkvec], (d_row), (tok), (pos). Common: addresses in the same
// order.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

static inline int32_t f2i(uint32_t bits) {
    if ((bits << 1) == 0u) {
        return 0;
    }
    const uint32_t e = (bits >> 23) & 0xFFu;
    const uint32_t m = (bits & 0x7FFFFFu) | 0x800000u;
    int32_t v = (e >= 150u) ? (int32_t)(m << (e - 150u)) : (int32_t)(m >> (150u - e));
    return (bits >> 31) ? -v : v;
}
static inline uint32_t i2f(int32_t v) {
    if (v == 0) {
        return 0u;
    }
    uint32_t sign = 0u, u = (uint32_t)v;
    if (v < 0) {
        sign = 0x80000000u;
        u = (uint32_t)(-v);
    }
    int msb = 31;
    while (!((u >> msb) & 1u)) {
        --msb;
    }
    const uint32_t mant = (msb <= 23) ? (u << (23 - msb)) : (u >> (msb - 23));
    return sign | ((uint32_t)(127 + msb) << 23) | (mant & 0x7FFFFFu);
}

void kernel_main() {
    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t S = get_compile_time_arg_val(1);
    constexpr uint32_t K = get_compile_time_arg_val(2);
    constexpr int32_t PAD = get_compile_time_arg_val(3);
    constexpr uint32_t HASPV = get_compile_time_arg_val(4);
    constexpr auto d_args = TensorAccessorArgs<5>();  // the k draft tiles share one layout
    constexpr auto nx_args = TensorAccessorArgs<d_args.next_compile_time_args_offset()>();
    constexpr auto pn_args = TensorAccessorArgs<nx_args.next_compile_time_args_offset()>();
    constexpr auto pk_args = TensorAccessorArgs<pn_args.next_compile_time_args_offset()>();
    constexpr auto pv_args = TensorAccessorArgs<pk_args.next_compile_time_args_offset()>();
    constexpr auto dr_args = TensorAccessorArgs<pv_args.next_compile_time_args_offset()>();
    constexpr auto tk_args = TensorAccessorArgs<dr_args.next_compile_time_args_offset()>();
    constexpr auto po_args = TensorAccessorArgs<tk_args.next_compile_time_args_offset()>();
    Noc noc;
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    // slots (64 B): 0..2 d_1..d_k, 3 nxt, 4 p_new, 5 park, 6 parkvec | outputs 8 d_row, 9 tok, 10 pos
    for (uint32_t i = 0; i < 16u * 12u; ++i) {
        sc[i] = 0u;
    }
    for (uint32_t j = 0; j < K; ++j) {
        const auto a = TensorAccessor(d_args, get_common_arg_val<uint32_t>(j), 4096);
        noc.async_read(a, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = j * 64});
    }
    const auto nx_acc = TensorAccessor(nx_args, get_common_arg_val<uint32_t>(K), 4096);
    const auto pn_acc = TensorAccessor(pn_args, get_common_arg_val<uint32_t>(K + 1), 4096);
    const auto pk_acc = TensorAccessor(pk_args, get_common_arg_val<uint32_t>(K + 2), 4096);
    noc.async_read(nx_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 3 * 64});
    noc.async_read(pn_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 4 * 64});
    noc.async_read(pk_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 5 * 64});
    if (HASPV) {
        const auto pv_acc = TensorAccessor(pv_args, get_common_arg_val<uint32_t>(K + 3), 4096);
        noc.async_read(pv_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 6 * 64});
    }
    noc.async_read_barrier();
    invalidate_l1_cache();
    const bool parked = (sc[5 * 16] << 1) != 0u;  // park = 0.0 / 1.0
    volatile tt_l1_ptr uint32_t* dr = sc + 8 * 16;
    volatile tt_l1_ptr uint32_t* tk = sc + 9 * 16;
    const uint32_t NEG1 = 0xBF800000u;
    for (uint32_t j = 0; j < 16u; ++j) {
        dr[j] = NEG1;
    }
    if (!parked) {
        for (uint32_t j = 0; j < K; ++j) {
            dr[j] = sc[j * 16];
            tk[1 + j] = (uint32_t)f2i(sc[j * 16]);
        }
        tk[0] = (uint32_t)f2i(sc[3 * 16]);
        sc[10 * 16] = sc[4 * 16];
    } else {
        tk[0] = (uint32_t)PAD;
        sc[10 * 16] = HASPV ? sc[6 * 16] : 0u;
    }
    const uint32_t np = 0;
    (void)np;
    const auto dr_acc = TensorAccessor(dr_args, get_common_arg_val<uint32_t>(K + 3 + HASPV), 4096);
    const auto tk_acc = TensorAccessor(tk_args, get_common_arg_val<uint32_t>(K + 4 + HASPV), 64);
    const auto po_acc = TensorAccessor(po_args, get_common_arg_val<uint32_t>(K + 5 + HASPV), 4096);
    noc.async_write(cb, dr_acc, 64, {.offset_bytes = 8 * 64}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write(cb, tk_acc, 64, {.offset_bytes = 9 * 64}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write(cb, po_acc, 64, {.offset_bytes = 10 * 64}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write_barrier();
}
