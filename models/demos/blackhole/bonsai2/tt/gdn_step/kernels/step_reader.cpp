// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// One-token GatedDeltaNet step, reader (NCRISC). Core j owns value head j (its key head is kh = j / 3).
// Every activation is a [.., 32, N] bf16 TILE tensor whose row 0 is the token; only row 0 is read (64-B reads of
// face rows 0-1 into scratch: Blackhole DRAM reads must be 64-B aligned and sized).
//
// Conv input X_c (12 chunks of 32 channels: 4 of q, 4 of k, 4 of v): rows 0-2 = the conv state rows (oldest first),
// row 3 = the new token, all other rows zero. The compute multiplies it by the taps tile (rows 0-3 = taps 0-3) and
// sums rows 0-3 with a matmul. The new conv state (rows 1-3 of X_c, then a zero row) is written back here: the
// v chunks by every core, the q/k chunks (shared by the 3 cores of a key head) by the first of the 3 once the other
// two have read them (semaphore 0).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t TB = 2048;  // bf16 tile bytes

void zero(uint32_t l1, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t o = 0; o < bytes; o += MEM_ZEROS_SIZE) {
        noc_async_read(z, l1 + o, MEM_ZEROS_SIZE);
    }
}

void copy32(uint32_t dst, uint32_t src) {  // 32 B, word by word (L1 to L1)
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst);
    const volatile tt_l1_ptr uint32_t* s = reinterpret_cast<const volatile tt_l1_ptr uint32_t*>(src);
    for (uint32_t w = 0; w < 8; ++w) {
        d[w] = s[w];
    }
}

// Row 0 of a bf16 tile filled with one value (face 0 and face 1, 16 values each).
void fill_row0(uint32_t tile, uint16_t v) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile);
    const uint32_t two = (uint32_t)v | ((uint32_t)v << 16);
    for (uint32_t w = 0; w < 8; ++w) {
        d[w] = two;
        d[128 + w] = two;  // face 1 starts 512 B in
    }
}
}  // namespace

void kernel_main() {
    const uint32_t qkv_addr = get_arg_val<uint32_t>(0);
    const uint32_t z_addr = get_arg_val<uint32_t>(1);
    const uint32_t ab_addr = get_arg_val<uint32_t>(2);
    const uint32_t conv_addr = get_arg_val<uint32_t>(3);
    const uint32_t st_addr = get_arg_val<uint32_t>(4);
    const uint32_t c_addr = get_arg_val<uint32_t>(5);
    const uint32_t j = get_arg_val<uint32_t>(6);
    const uint32_t leader_x = get_arg_val<uint32_t>(7);
    const uint32_t leader_y = get_arg_val<uint32_t>(8);
    const uint32_t leader = get_arg_val<uint32_t>(9);

    constexpr uint32_t NV = get_compile_time_arg_val(0);     // value heads (48)
    constexpr uint32_t QT = get_compile_time_arg_val(1);     // q (= k) width in tiles (2048 / 32 = 64)
    constexpr uint32_t HT = get_compile_time_arg_val(2);     // head width in tiles (128 / 32 = 4)
    constexpr uint32_t GROUP = get_compile_time_arg_val(3);  // value heads per key head (3)
    constexpr auto qkv_args = TensorAccessorArgs<4>();
    constexpr auto z_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr auto ab_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();
    constexpr auto conv_args = TensorAccessorArgs<ab_args.next_compile_time_args_offset()>();
    constexpr auto st_args = TensorAccessorArgs<conv_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<st_args.next_compile_time_args_offset()>();
    const auto qkv = TensorAccessor(qkv_args, qkv_addr, TB);
    const auto zs = TensorAccessor(z_args, z_addr, TB);
    const auto ab = TensorAccessor(ab_args, ab_addr, TB);
    const auto conv = TensorAccessor(conv_args, conv_addr, TB);
    const auto st = TensorAccessor(st_args, st_addr, TB);
    const auto cs = TensorAccessor(c_args, c_addr, TB);

    constexpr uint32_t cb_x = 0, cb_t = 1, cb_z = 2, cb_a = 3, cb_b = 4, cb_s = 5, cb_w = 6, cb_e = 7, cb_ones = 8;
    constexpr uint32_t cb_sc = 9, cb_n = 10;
    constexpr uint32_t NCH = 3 * HT;  // conv chunks: q, k, v of this head
    const uint32_t kh = j / GROUP;
    auto chunk_tile = [&](uint32_t c) -> uint32_t {
        return c < HT ? kh * HT + c : (c < 2 * HT ? QT + kh * HT + (c - HT) : 2 * QT + j * HT + (c - 2 * HT));
    };

    cb_reserve_back(cb_x, NCH);
    cb_reserve_back(cb_z, HT);
    cb_reserve_back(cb_a, 1);
    cb_reserve_back(cb_b, 1);
    cb_reserve_back(cb_t, NCH);
    cb_reserve_back(cb_s, HT * HT);
    cb_reserve_back(cb_w, HT);
    cb_reserve_back(cb_e, 1);
    cb_reserve_back(cb_ones, 1);
    cb_reserve_back(cb_sc, 2);
    cb_reserve_back(cb_n, 2);
    const uint32_t X = get_write_ptr(cb_x), Z = get_write_ptr(cb_z), A = get_write_ptr(cb_a), B = get_write_ptr(cb_b);
    const uint32_t SC = get_write_ptr(cb_sc), N = get_write_ptr(cb_n);

    zero(X, NCH * TB);
    zero(Z, HT * TB);
    zero(A, TB);
    zero(B, TB);
    noc_async_read_barrier();  // the zero fills land before the row reads into the same pages

    for (uint32_t c = 0; c < NCH; ++c) {
        const uint32_t t = chunk_tile(c);
        const uint64_t s = conv.get_noc_addr(t);
        noc_async_read(s, X + c * TB, 128);              // face 0 rows 0-3 (row 3 is overwritten below)
        noc_async_read(s + 512, X + c * TB + 512, 128);  // face 1 rows 0-3
        const uint64_t x = qkv.get_noc_addr(t);
        noc_async_read(x, SC + c * 128, 64);
        noc_async_read(x + 512, SC + c * 128 + 64, 64);
    }
    for (uint32_t i = 0; i < HT; ++i) {
        const uint64_t s = zs.get_noc_addr(j * HT + i);
        noc_async_read(s, SC + NCH * 128 + i * 128, 64);
        noc_async_read(s + 512, SC + NCH * 128 + i * 128 + 64, 64);
    }
    const uint32_t ca = j, cbb = NV + j;  // a and b columns of the fused [a | b] projection
    noc_async_read(ab.get_noc_addr(ca / 32) + ((ca % 32) / 16) * 512, SC + TB, 64);
    noc_async_read(ab.get_noc_addr(cbb / 32) + ((cbb % 32) / 16) * 512, SC + TB + 64, 64);
    const uint32_t S = get_write_ptr(cb_s);
    for (uint32_t t = 0; t < HT * HT; ++t) {
        noc_async_read(st.get_noc_addr(j * HT * HT + t), S + t * TB, TB);
    }
    // consts: [NV * NCH taps tiles | HT norm-weight tiles | E | ONES]
    const uint32_t T = get_write_ptr(cb_t);
    for (uint32_t c = 0; c < NCH; ++c) {
        noc_async_read(cs.get_noc_addr(j * NCH + c), T + c * TB, TB);
    }
    const uint32_t W = get_write_ptr(cb_w);
    for (uint32_t i = 0; i < HT; ++i) {
        noc_async_read(cs.get_noc_addr(NV * NCH + i), W + i * TB, TB);
    }
    noc_async_read(cs.get_noc_addr(NV * NCH + HT), get_write_ptr(cb_e), TB);
    noc_async_read(cs.get_noc_addr(NV * NCH + HT + 1), get_write_ptr(cb_ones), TB);
    noc_async_read_barrier();

    const uint32_t sem = get_semaphore(0);
    if (!leader) {
        noc_semaphore_inc(get_noc_addr(leader_x, leader_y, sem), 1);  // this core's reads of the shared q/k are done
    }

    for (uint32_t c = 0; c < NCH; ++c) {
        copy32(X + c * TB + 96, SC + c * 128);             // the token into row 3, face 0
        copy32(X + c * TB + 512 + 96, SC + c * 128 + 64);  // face 1
    }
    for (uint32_t i = 0; i < HT; ++i) {
        copy32(Z + i * TB, SC + NCH * 128 + i * 128);
        copy32(Z + i * TB + 512, SC + NCH * 128 + i * 128 + 64);
    }
    {
        const volatile tt_l1_ptr uint16_t* h = reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(SC + TB);
        fill_row0(A, h[ca % 16]);
        fill_row0(B, h[32 + cbb % 16]);
    }
    cb_push_back(cb_a, 1);
    cb_push_back(cb_b, 1);
    cb_push_back(cb_z, HT);
    cb_push_back(cb_x, NCH);
    cb_push_back(cb_t, NCH);
    cb_push_back(cb_e, 1);
    cb_push_back(cb_ones, 1);
    cb_push_back(cb_s, HT * HT);
    cb_push_back(cb_w, HT);

    // New conv state rows 0-3 = old rows 1-2, the token, zero (face 0 then face 1, 128 B each).
    for (uint32_t c = 0; c < NCH; ++c) {
        const uint32_t n = N + c * 256, x = X + c * TB;
        for (uint32_t f = 0; f < 2; ++f) {
            copy32(n + f * 128, x + f * 512 + 32);
            copy32(n + f * 128 + 32, x + f * 512 + 64);
            copy32(n + f * 128 + 64, x + f * 512 + 96);
        }
    }
    {
        volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(N);
        for (uint32_t c = 0; c < NCH; ++c) {
            for (uint32_t f = 0; f < 2; ++f) {
                for (uint32_t w = 0; w < 8; ++w) {
                    d[(c * 256 + f * 128 + 96) / 4 + w] = 0;
                }
            }
        }
    }
    for (uint32_t c = 2 * HT; c < NCH; ++c) {
        const uint64_t s = conv.get_noc_addr(chunk_tile(c));
        noc_async_write(N + c * 256, s, 128);
        noc_async_write(N + c * 256 + 128, s + 512, 128);
    }
    if (leader) {
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), GROUP - 1);
        for (uint32_t c = 0; c < 2 * HT; ++c) {
            const uint64_t s = conv.get_noc_addr(chunk_tile(c));
            noc_async_write(N + c * 256, s, 128);
            noc_async_write(N + c * 256 + 128, s + 512, 128);
        }
    }
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
