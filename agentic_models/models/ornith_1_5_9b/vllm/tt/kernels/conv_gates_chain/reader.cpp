// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Chained multi-token verify for conv_gates (B requests x T candidates in raw rows u*T + t). Derived from
// conv_gates/reader.cpp: the compute kernel is unchanged and sees, for every row t, exactly the operands a plain
// one-token step for candidate t would see: tap j (CB1 + j) row t = sequence element t - 3 + j, where elements -3..-1
// are the request's history row `slot` and elements 0..T-1 are the raw rows. The writer then stores CB2, CB3, CB0 as
// the chain history: row t = the history after candidate t.
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

static inline uint32_t row_offset(uint32_t r) { return (r / 16) * 1024 + (r % 16) * 32; }

static inline void copy_row(uint32_t dst_tile, uint32_t dst_row, uint32_t src_tile, uint32_t src_row) {
    volatile tt_l1_ptr uint32_t* d = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_tile + row_offset(dst_row));
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src_tile + row_offset(src_row));
    for (uint32_t w = 0; w < 8; ++w) {
        d[w] = s[w];              // columns 0..15
        d[w + 128] = s[w + 128];  // columns 16..31 (next face, +512 bytes)
    }
}

void kernel_main() {
    constexpr auto ra = TensorAccessorArgs<0>();
    constexpr auto ha = TensorAccessorArgs<ra.next_compile_time_args_offset()>();
    constexpr auto t0a = TensorAccessorArgs<ha.next_compile_time_args_offset()>();
    constexpr auto t1a = TensorAccessorArgs<t0a.next_compile_time_args_offset()>();
    constexpr auto t2a = TensorAccessorArgs<t1a.next_compile_time_args_offset()>();
    constexpr auto t3a = TensorAccessorArgs<t2a.next_compile_time_args_offset()>();
    constexpr auto da = TensorAccessorArgs<t3a.next_compile_time_args_offset()>();
    constexpr auto ea = TensorAccessorArgs<da.next_compile_time_args_offset()>();
    const uint32_t tile = get_arg_val<uint32_t>(0);
    const auto raw = TensorAccessor(ra, get_arg_val<uint32_t>(1), 2048);
    const auto hist = TensorAccessor(ha, get_arg_val<uint32_t>(2), 2048);
    const auto t0 = TensorAccessor(t0a, get_arg_val<uint32_t>(3), 4096);
    const auto t1 = TensorAccessor(t1a, get_arg_val<uint32_t>(4), 4096);
    const auto t2 = TensorAccessor(t2a, get_arg_val<uint32_t>(5), 4096);
    const auto t3 = TensorAccessor(t3a, get_arg_val<uint32_t>(6), 4096);
    // rows r = u*T + t: candidate t of user u (user u's live history is pool row u)
    const uint32_t T = get_arg_val<uint32_t>(9), R = get_arg_val<uint32_t>(10);
    for (uint32_t cb = 0; cb < 8; ++cb) {
        CircularBuffer buf(cb);
        buf.reserve_back(1);
    }
    // start history taps into scratch CB19 (3 pages), raw into CB0
    CircularBuffer scratch(19);
    scratch.reserve_back(3);
    const uint32_t start = get_write_ptr(19);
    noc_async_read(raw.get_noc_addr(tile), get_write_ptr(0), 2048);
    for (uint32_t j = 0; j < 3; ++j) {
        noc_async_read(hist.get_noc_addr(j * 64 + tile), start + j * 2048, 2048);
    }
    noc_async_read(t0.get_noc_addr(tile), get_write_ptr(4), 4096);
    noc_async_read(t1.get_noc_addr(tile), get_write_ptr(5), 4096);
    noc_async_read(t2.get_noc_addr(tile), get_write_ptr(6), 4096);
    noc_async_read(t3.get_noc_addr(tile), get_write_ptr(7), 4096);
    noc_async_read_barrier();
    const uint32_t raw_tile = get_write_ptr(0);
    for (uint32_t j = 0; j < 3; ++j) {
        const uint32_t dst = get_write_ptr(1 + j);
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t u = r / T, t = r % T;
            const int32_t e = static_cast<int32_t>(t) - 3 + static_cast<int32_t>(j);
            if (e < 0) {
                copy_row(dst, r, start + static_cast<uint32_t>(3 + e) * 2048, u);
            } else {
                copy_row(dst, r, raw_tile, u * T + static_cast<uint32_t>(e));
            }
        }
    }
    for (uint32_t cb = 0; cb < 8; ++cb) {
        CircularBuffer buf(cb);
        buf.push_back(1);
    }
    if (tile == 0) {
        const auto bias = TensorAccessor(da, get_arg_val<uint32_t>(7), 4096);
        const auto decay = TensorAccessor(ea, get_arg_val<uint32_t>(8), 4096);
        for (uint32_t cb = 8; cb < 12; ++cb) {
            CircularBuffer(cb).reserve_back(1);
        }
        noc_async_read(raw.get_noc_addr(96), get_write_ptr(8), 2048);
        noc_async_read(raw.get_noc_addr(97), get_write_ptr(9), 2048);
        noc_async_read(bias.get_noc_addr(0), get_write_ptr(10), 4096);
        noc_async_read(decay.get_noc_addr(0), get_write_ptr(11), 4096);
        noc_async_read_barrier();
        for (uint32_t cb = 8; cb < 12; ++cb) {
            CircularBuffer(cb).push_back(1);
        }
    }
}
