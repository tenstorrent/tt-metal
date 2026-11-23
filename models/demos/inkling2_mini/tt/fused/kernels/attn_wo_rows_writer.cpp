// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Speculative verify: attn_wo_writer.cpp for R rows (tt/fused/attn_wo_rows.py). Row r of the output tiles goes to
// page r of cb_a (same slot layout); the ring lags 1..3 are read once (row 0's), the compute takes row r's lags from
// rows r-1.. of a and the older ones from the ring. Per row: fp32 a -> its pending slot, conv out -> its dense twin.
// Based on the 1-row writer (tt/fused/attn_wo.py): The matmul part is bank_linear_writer.cpp (activation in walk order,
// output tiles per pass). Then row 0 of every output column tile n (32 bf16 values of a, columns 32 n ..) goes to slot
// k (its index on this core) of one local dense page cb_a: element 32 k + i. The conv is elementwise per column, so the
// same slot layout of the ring slots, taps and residual (128 B / 64 B chunks of their dense pages) lets the compute run
// the conv kernel's exact op sequence on this one page. Outputs: fp32 conv input -> ring slot of this phase, conv out
// (+ resid, fp32) -> dense twin of the conv output. a itself is not written.
// Runtime args: x, rot, npass, R pending slots, ring lag 1..3, taps, R resid dense (0: none), R conv out dense,
// then (ncq, first column tile) per pass.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t cb_x = 0, cb_out = 16;
constexpr uint32_t cb_a = 2, cb_ring = 3, cb_taps = 4, cb_res = 5, cb_xf = 17, cb_y = 18;
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");
constexpr uint32_t nblk = get_named_compile_time_arg_val("nblk");
constexpr uint32_t out_tile = get_named_compile_time_arg_val("out_tile");
constexpr uint32_t C = get_named_compile_time_arg_val("cols");         // vector length (columns >= C: no slot)
constexpr uint32_t res_es = get_named_compile_time_arg_val("res_es");  // 0: no resid
constexpr uint32_t R = get_named_compile_time_arg_val("rows");
constexpr uint32_t DST_TILES = 8;
constexpr uint32_t F32 = 4096;             // dense fp32 page
constexpr uint32_t T = (C + 1023) / 1024;  // dense pages per vector (lag k taps at pages k T ..)

FORCE_INLINE uint64_t dense_addr(const InterleavedAddrGen<true>& g, uint32_t col, uint32_t es) {
    return g.get_noc_addr(col >> 10, (col & 1023) * es);
}

void kernel_main() {
    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t rot = get_arg_val<uint32_t>(a++);
    const uint32_t npass = get_arg_val<uint32_t>(a++);
    const uint32_t a_slot = a;
    a += R;
    const uint32_t r1 = get_arg_val<uint32_t>(a++);
    const uint32_t r2 = get_arg_val<uint32_t>(a++);
    const uint32_t r3 = get_arg_val<uint32_t>(a++);
    const uint32_t taps = get_arg_val<uint32_t>(a++);
    const uint32_t a_res = a;
    a += R;
    const uint32_t a_y = a;
    a += R;
    const uint32_t a_pass = a;
    constexpr auto x_args = TensorAccessorArgs<0>();
    const auto xs = TensorAccessor(x_args, x_addr);

    const uint32_t xtb = get_tile_size(cb_x);
    cb_reserve_back(cb_x, kt);
    const uint32_t xl = get_write_ptr(cb_x);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    for (uint32_t p = 0; p < nblk; ++p) {
        const uint32_t j = (p + rot) % nblk;
        noc_async_read_set_trid(p % 4 + 1);
        for (uint32_t r = 0; r < kb; ++r) {
            noc_async_read(xs.get_noc_addr(j * kb + r), xl + (p * kb + r) * xtb, xtb);
        }
        if (p > 0) {
            noc_async_read_barrier_with_trid((p - 1) % 4 + 1);
            cb_push_back(cb_x, kb);
        }
    }
    noc_async_read_barrier_with_trid((nblk - 1) % 4 + 1);
    cb_push_back(cb_x, kb);
    noc_async_read_set_trid(0);

    // conv inputs of this core's columns (land while the weights stream)
    {
        const InterleavedAddrGen<true> g1 = {.bank_base_address = r1, .page_size = F32};
        const InterleavedAddrGen<true> g2 = {.bank_base_address = r2, .page_size = F32};
        const InterleavedAddrGen<true> g3 = {.bank_base_address = r3, .page_size = F32};
        const InterleavedAddrGen<true> gt = {.bank_base_address = taps, .page_size = F32};
        cb_reserve_back(cb_ring, 3);
        cb_reserve_back(cb_taps, 4);
        const uint32_t ring = get_write_ptr(cb_ring), tp = get_write_ptr(cb_taps);
        uint32_t rs = 0;
        if constexpr (res_es) {
            cb_reserve_back(cb_res, R);
            rs = get_write_ptr(cb_res);
        }
        uint32_t k = 0, b = a_pass;
        for (uint32_t q = 0; q < npass; ++q) {
            const uint32_t ncq = get_arg_val<uint32_t>(b++);
            const uint32_t n0 = get_arg_val<uint32_t>(b++);
            for (uint32_t t = 0; t < ncq; ++t) {
                const uint32_t col = (n0 + t) * 32;
                if (col >= C) {
                    continue;
                }
                noc_async_read(dense_addr(g1, col, 4), ring + k * 128, 128);
                noc_async_read(dense_addr(g2, col, 4), ring + F32 + k * 128, 128);
                noc_async_read(dense_addr(g3, col, 4), ring + 2 * F32 + k * 128, 128);
                for (uint32_t l = 0; l < 4; ++l) {
                    noc_async_read(dense_addr(gt, l * T * 1024 + col, 4), tp + l * F32 + k * 128, 128);
                }
                if constexpr (res_es) {
                    for (uint32_t r = 0; r < R; ++r) {
                        const InterleavedAddrGen<true> gr = {
                            .bank_base_address = get_arg_val<uint32_t>(a_res + r), .page_size = 1024 * res_es};
                        noc_async_read(
                            dense_addr(gr, col, res_es), rs + r * 1024 * res_es + k * 32 * res_es, 32 * res_es);
                    }
                }
                ++k;
            }
        }
    }

    // row r of each output tile (bf16: face 0 = columns 0..15, face 1 = 16..31) -> its slot of page r of cb_a
    cb_reserve_back(cb_a, R);
    const uint32_t al = get_write_ptr(cb_a);
    uint32_t k = 0;
    for (uint32_t q = 0, b = a_pass; q < npass; ++q) {
        const uint32_t ncq = get_arg_val<uint32_t>(b++);
        const uint32_t n0 = get_arg_val<uint32_t>(b++);
        cb_wait_front(cb_out, DST_TILES);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < ncq; ++t) {
            if ((n0 + t) * 32 < C) {
                for (uint32_t r = 0; r < R; ++r) {
                    const uint32_t s0 = src + t * out_tile + r * 32, d0 = al + r * 2048 + k * 64;
                    noc_async_read(get_noc_addr(s0), d0, 32);
                    noc_async_read(get_noc_addr(s0 + 512), d0 + 32, 32);
                }
                ++k;
            }
        }
        noc_async_read_barrier();  // also the conv inputs
        cb_pop_front(cb_out, DST_TILES);
    }
    cb_push_back(cb_ring, 3);
    cb_push_back(cb_taps, 4);
    if constexpr (res_es) {
        cb_push_back(cb_res, R);
    }
    cb_push_back(cb_a, R);

    // per row: fp32 a -> its pending slot, conv out -> its dense twin
    for (uint32_t r = 0; r < R; ++r) {
        const InterleavedAddrGen<true> gs = {.bank_base_address = get_arg_val<uint32_t>(a_slot + r), .page_size = F32};
        const InterleavedAddrGen<true> gy = {.bank_base_address = get_arg_val<uint32_t>(a_y + r), .page_size = F32};
        for (uint32_t pass = 0; pass < 2; ++pass) {
            const uint32_t cb = pass ? cb_y : cb_xf;
            const auto& g = pass ? gy : gs;
            cb_wait_front(cb, 1);
            const uint32_t src = get_read_ptr(cb);
            uint32_t i = 0;
            for (uint32_t q = 0, b = a_pass; q < npass; ++q) {
                const uint32_t ncq = get_arg_val<uint32_t>(b++);
                const uint32_t n0 = get_arg_val<uint32_t>(b++);
                for (uint32_t t = 0; t < ncq; ++t) {
                    const uint32_t col = (n0 + t) * 32;
                    if (col < C) {
                        noc_async_write(src + i++ * 128, dense_addr(g, col, 4), 128);
                    }
                }
            }
            noc_async_write_barrier();
            cb_pop_front(cb, 1);
        }
    }
}
