// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Hy4 B40 (HY4_B40_TAIL=1): greedy accept of one user as ONE one-core program, bit-identical to hy4_spec_trace.accept +
// the m / one-hot / next-position glue (~30 tiny ops). Inputs (per device): a_tok uint32 RM [S,1] (verify output
// tokens), d_row fp32 TILE [1,1,1,16] (drafts, -1 at j >= k), cap fp32 TILE [1,1,1,1], pos fp32 TILE [1,1,1,1].
// Outputs: a_out fp32 TILE (a_0..a_{S-1}, zeros to 16), m_out fp32 TILE (element 0 = m), oh fp32? no: oh_col bf16 TILE
// [1,1,S,1] (face 0 column 0: 1.0 at row m), p_new fp32 TILE (P + m + 1), nxt fp32 TILE (a_m). CT: 0 CB, 1 S, 2 k;
// TensorAccessorArgs(a_tok), (d_row), (cap), (pos), (a_out), (m_out), (oh), (p_new), (nxt). Common: the nine addresses
// in the same order.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

static inline int32_t f2i(uint32_t bits) {  // integer valued fp32 (|v| < 2^24) -> int
    if ((bits << 1) == 0u) {
        return 0;
    }
    const uint32_t e = (bits >> 23) & 0xFFu;
    const uint32_t m = (bits & 0x7FFFFFu) | 0x800000u;
    int32_t v = (e >= 150u) ? (int32_t)(m << (e - 150u)) : (int32_t)(m >> (150u - e));
    return (bits >> 31) ? -v : v;
}
static inline uint32_t i2f(int32_t v) {  // int (|v| < 2^24) -> fp32 bits
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
    constexpr auto at_args = TensorAccessorArgs<3>();
    constexpr auto dr_args = TensorAccessorArgs<at_args.next_compile_time_args_offset()>();
    constexpr auto cap_args = TensorAccessorArgs<dr_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<cap_args.next_compile_time_args_offset()>();
    constexpr auto ao_args = TensorAccessorArgs<pos_args.next_compile_time_args_offset()>();
    constexpr auto mo_args = TensorAccessorArgs<ao_args.next_compile_time_args_offset()>();
    constexpr auto oh_args = TensorAccessorArgs<mo_args.next_compile_time_args_offset()>();
    constexpr auto pn_args = TensorAccessorArgs<oh_args.next_compile_time_args_offset()>();
    constexpr auto nx_args = TensorAccessorArgs<pn_args.next_compile_time_args_offset()>();
    Noc noc;
    experimental::CB cb(cb_id);
    cb.reserve_back(1);
    const uint32_t base = cb.get_write_ptr();
    volatile tt_l1_ptr uint32_t* sc = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    // 64 B slots: 0..3 a_tok rows (0..S-1), 4 d_row, 5 cap, 6 pos | outputs: 8 a_out, 9 m_out, 10 p_new, 11 nxt, 16..23
    // oh (512 B)
    for (uint32_t i = 0; i < 16u * 24u; ++i) {
        sc[i] = 0u;
    }
    const auto at_acc = TensorAccessor(at_args, get_common_arg_val<uint32_t>(0), 64);
    const auto dr_acc = TensorAccessor(dr_args, get_common_arg_val<uint32_t>(1), 4096);
    const auto cap_acc = TensorAccessor(cap_args, get_common_arg_val<uint32_t>(2), 4096);
    const auto pos_acc = TensorAccessor(pos_args, get_common_arg_val<uint32_t>(3), 4096);
    for (uint32_t j = 0; j < S; ++j) {
        noc.async_read(at_acc, cb, 64, {.page_id = j, .offset_bytes = 0}, {.offset_bytes = j * 64});
    }
    noc.async_read(dr_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 4 * 64});
    noc.async_read(cap_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 5 * 64});
    noc.async_read(pos_acc, cb, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 6 * 64});
    noc.async_read_barrier();
    invalidate_l1_cache();
    uint32_t a[16];
    for (uint32_t j = 0; j < 16u; ++j) {
        a[j] = (j < S) ? sc[j * 16] : 0u;
    }
    volatile tt_l1_ptr uint32_t* dr = sc + 4 * 16;
    const int32_t cap = f2i(sc[5 * 16]);
    const int32_t pos = f2i(sc[6 * 16]);
    int32_t m = 0;
    for (uint32_t j = 0; j < K; ++j) {
        if ((int32_t)a[j] == f2i(dr[j])) {
            ++m;
        } else {
            break;
        }
    }
    if (m > cap) {
        m = cap;
    }
    volatile tt_l1_ptr uint32_t* ao = sc + 8 * 16;
    for (uint32_t j = 0; j < 16u; ++j) {
        ao[j] = i2f((int32_t)a[j]);
    }
    sc[9 * 16] = i2f(m);
    sc[10 * 16] = i2f(pos + m + 1);
    sc[11 * 16] = i2f((int32_t)a[m]);
    volatile tt_l1_ptr uint16_t* oh = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + 16 * 64);
    oh[m * 16] = 0x3F80;  // bf16 1.0 at (row m, col 0) of face 0
    const auto ao_acc = TensorAccessor(ao_args, get_common_arg_val<uint32_t>(4), 4096);
    const auto mo_acc = TensorAccessor(mo_args, get_common_arg_val<uint32_t>(5), 4096);
    const auto oh_acc = TensorAccessor(oh_args, get_common_arg_val<uint32_t>(6), 2048);
    const auto pn_acc = TensorAccessor(pn_args, get_common_arg_val<uint32_t>(7), 4096);
    const auto nx_acc = TensorAccessor(nx_args, get_common_arg_val<uint32_t>(8), 4096);
    noc.async_write(cb, ao_acc, 64, {.offset_bytes = 8 * 64}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write(cb, mo_acc, 64, {.offset_bytes = 9 * 64}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write(cb, pn_acc, 64, {.offset_bytes = 10 * 64}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write(cb, nx_acc, 64, {.offset_bytes = 11 * 64}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write(cb, oh_acc, 512, {.offset_bytes = 16 * 64}, {.page_id = 0, .offset_bytes = 0});
    noc.async_write_barrier();
}
