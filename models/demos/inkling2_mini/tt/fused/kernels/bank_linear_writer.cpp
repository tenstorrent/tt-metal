// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Bank-streamed decode linear, writer (BRISC, NOC1) (tt/fused/bank_linear.py): row 0 of each fp32 output tile
// (face 0 row 0 = columns 0..15, face 1 row 0 = 16..31) into the dense fp32 output row at column 32 * n.
// tiles_out: instead the whole tile to page n of a TILE DRAM tensor. Columns >= out_cols are not written.
// First it reads the activation (kt tiles, the token rows) into cb_x on this NOC, block by block in the compute's
// walk order (block j = (p + rot) % nblk to position p, as the reference's lookahead), so compute starts after the
// first block while the weight stream runs on the reader's NOC. Runtime args: x address, rot, out address, number
// of passes, then (ncq, first column tile) per pass.
// x_rows > 0: only the first x_rows rows of each activation tile are in use (the rest zero): x_rows.hpp reads those
// and fills the rest from L1 zeros (same cb_x bytes, less NOC traffic on the activation's shard cores).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "x_rows.hpp"

constexpr uint32_t cb_x = 0, cb_out = 16;
constexpr uint32_t kt = get_named_compile_time_arg_val("kt");
constexpr uint32_t kb = get_named_compile_time_arg_val("kb");
constexpr uint32_t nblk = get_named_compile_time_arg_val("nblk");
constexpr uint32_t out_cols = get_named_compile_time_arg_val("out_cols");
constexpr uint32_t out_page = get_named_compile_time_arg_val("out_page");
constexpr uint32_t tiles_out = get_named_compile_time_arg_val("tiles_out");
constexpr uint32_t raw_rows = get_named_compile_time_arg_val("raw_rows");  // raw rows written (tiles_out 0)
constexpr uint32_t out_tile = get_named_compile_time_arg_val("out_tile");
constexpr uint32_t x_rows = get_named_compile_time_arg_val("x_rows");
// x_half: x and the output are 16 x 32 tiles (bank_linear.py HALF): x tiles get faces 0, 1; out_tile is the half
// tile, written to the start of each 32 x 32 output page (rows 0..15)
constexpr uint32_t x_half = get_named_compile_time_arg_val("x_half");
constexpr uint32_t DST_TILES = 8;
// head (the LM head, tt/fused/head.py): out rows are the logits rows; also the argmax of each row over columns
// < n_arg, as the logits tail: the first NaN wins, else the largest value (-0 == +0), lowest column on ties. Each
// worker scans its tiles as they are written, sends (value, column, valid) per row to worker 0's scratch (cb 17,
// slot = worker), worker 0 merges and writes the tokens. Args after the passes: slot, worker 0's NOC x, y, workers.
// Common args: logits addresses of rows 1.., then the token addresses of rows 0.. Define (head mode only): BL_HEAD =
// n_arg.
#ifdef BL_HEAD
constexpr uint32_t head = 1, n_arg = BL_HEAD;
#else
constexpr uint32_t head = 0, n_arg = 0;
#endif

namespace {
// argmax order key: every NaN the largest (all equal: the lowest column wins, as the tail's "first NaN"), else the
// float order with -0 folded into +0; ties -> the lower column
FORCE_INLINE uint32_t key(uint32_t v) {
    if ((v & 0x7fffffffu) > 0x7f800000u) {
        return 0xffffffffu;
    }
    v = (v == 0x80000000u) ? 0 : v;
    return v ^ (static_cast<uint32_t>(static_cast<int32_t>(v) >> 31) | 0x80000000u);
}
}  // namespace

void kernel_main() {
    uint32_t a = 0;
    const uint32_t x_addr = get_arg_val<uint32_t>(a++);
    const uint32_t rot = get_arg_val<uint32_t>(a++);
    const uint32_t out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t npass = get_arg_val<uint32_t>(a++);
    constexpr auto x_args = TensorAccessorArgs<0>();
    constexpr auto out_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    const auto xs = TensorAccessor(x_args, x_addr);
    const auto out = TensorAccessor(out_args, out_addr, out_page);

    const uint32_t xtb = get_tile_size(cb_x);
    cb_reserve_back(cb_x, kt);
    const uint32_t xl = get_write_ptr(cb_x);
    const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    for (uint32_t p = 0; p < nblk; ++p) {
        const uint32_t j = (p + rot) % nblk;
        noc_async_read_set_trid(p % 4 + 1);
        for (uint32_t r = 0; r < kb; ++r) {
            const uint32_t dst = xl + (p * kb + r) * xtb;
            if constexpr (x_half) {
                read_x_half_tile<x_rows>(xs.get_noc_addr(j * kb + r), dst, zeros);
            } else if constexpr (x_rows > 0) {
                read_x_tile<x_rows>(xs.get_noc_addr(j * kb + r), dst, zeros);
            } else {
                noc_async_read(xs.get_noc_addr(j * kb + r), dst, xtb);
            }
        }
        if (p > 0) {
            noc_async_read_barrier_with_trid((p - 1) % 4 + 1);
            cb_push_back(cb_x, kb);
        }
    }
    noc_async_read_barrier_with_trid((nblk - 1) % 4 + 1);
    cb_push_back(cb_x, kb);
    noc_async_read_set_trid(0);
    uint32_t best_k[raw_rows], best_v[raw_rows], best_i[raw_rows];
    for (uint32_t r = 0; r < raw_rows; ++r) {
        best_k[r] = best_v[r] = 0;
        best_i[r] = 0xffffffffu;  // no column yet (any key >= 0 with a lower column wins)
    }
    for (uint32_t q = 0; q < npass; ++q) {
        const uint32_t ncq = get_arg_val<uint32_t>(a++);
        const uint32_t n0 = get_arg_val<uint32_t>(a++);
        cb_wait_front(cb_out, DST_TILES);
        const uint32_t src = get_read_ptr(cb_out);
        for (uint32_t t = 0; t < ncq; ++t) {
            const uint32_t col = (n0 + t) * 32;
            if (col >= out_cols) {
                continue;
            }
            if constexpr (tiles_out) {
                noc_async_write(src + t * out_tile, out.get_noc_addr(n0 + t), out_tile);
            } else {
#ifdef BL_ROW_TILES
                // out rows are [1, 1, 1, W] fp32 TILE tensors: row 0 of tile page n (face 0 / 1 row 0 = +0 / +1024),
                // pages spread over the banks (a ROW_MAJOR row is one page in one bank, which every worker would
                // write while that bank's workers stream from it)
                const uint64_t dst = out.get_noc_addr(n0 + t);
                constexpr uint32_t f1 = 1024;
#else
                const uint64_t dst = out.get_noc_addr(0, col * 4);
                constexpr uint32_t f1 = 64;
#endif
                noc_async_write(src + t * 4096, dst, 64);
                noc_async_write(src + t * 4096 + 1024, dst + f1, 64);
                // R verify rows: row r of the tile into raw row r (common args: the rows' row addresses)
                for (uint32_t r = 1; r < raw_rows; ++r) {
                    const auto o = TensorAccessor(out_args, get_common_arg_val<uint32_t>(r - 1), out_page);
#ifdef BL_ROW_TILES
                    const uint64_t d = o.get_noc_addr(n0 + t);
#else
                    const uint64_t d = o.get_noc_addr(0, col * 4);
#endif
                    noc_async_write(src + t * 4096 + r * 64, d, 64);
                    noc_async_write(src + t * 4096 + 1024 + r * 64, d + f1, 64);
                }
                if constexpr (head) {  // scan while the writes are in flight (cb_out is final after the wait)
                    const uint32_t nc = col + 32 <= n_arg ? 32 : (col < n_arg ? n_arg - col : 0);
                    for (uint32_t r = 0; r < raw_rows; ++r) {
                        uint32_t bk = best_k[r], bi = best_i[r], bv = best_v[r];
                        for (uint32_t h = 0; h < 2; ++h) {  // face h: columns 16 h .. 16 h + 15
                            const uint32_t c0 = 16 * h;
                            const uint32_t n = nc > c0 ? (nc - c0 < 16 ? nc - c0 : 16) : 0;
                            if (n == 0) {
                                continue;
                            }
                            const uint32_t* f = reinterpret_cast<const uint32_t*>(src + t * 4096 + h * 1024 + r * 64);
                            // fast filter: max of the sortable signed keys (float order, -0 just below +0) and a
                            // NaN flag; the exact pass only if a value may beat or tie the best so far
                            int32_t m = INT32_MIN;
                            uint32_t nan = 0;
#pragma GCC unroll 16
                            for (uint32_t c = 0; c < 16; ++c) {
                                const uint32_t v = f[c];
                                const int32_t k = static_cast<int32_t>(
                                    v ^ (static_cast<uint32_t>(static_cast<int32_t>(v) >> 31) & 0x7fffffffu));
                                m = (c < n && k > m) ? k : m;
                                nan |= c < n ? (v & 0x7fffffffu) + 0x007fffffu : 0;
                            }
                            // sortable m -> the ordered key domain (bias by 2^31), one below for -0 / +0
                            const uint32_t mk = static_cast<uint32_t>(m) ^ 0x80000000u;
                            if (!(nan & 0x80000000u) && bi != 0xffffffffu && mk + 1 < bk) {
                                continue;
                            }
                            for (uint32_t c = 0; c < n; ++c) {
                                const uint32_t v = f[c];
                                const uint32_t k = key(v);
                                if (k > bk || (k == bk && col + c0 + c < bi)) {
                                    bk = k;
                                    bi = col + c0 + c;
                                    bv = v;
                                }
                            }
                        }
                        best_k[r] = bk;
                        best_i[r] = bi;
                        best_v[r] = bv;
                    }
                }
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, DST_TILES);
    }
    if constexpr (head) {
        const uint32_t slot = get_arg_val<uint32_t>(a++);
        const uint32_t mx = get_arg_val<uint32_t>(a++), my = get_arg_val<uint32_t>(a++);
        const uint32_t n_w = get_arg_val<uint32_t>(a++);
        constexpr uint32_t cb_scr = 17;
        const uint32_t scr = get_write_ptr(cb_scr);  // same address on every worker; row r's slots at 16 r n_w
        for (uint32_t r = 0; r < raw_rows; ++r) {
            const uint32_t own = scr + 16 * (raw_rows * n_w + r);
            volatile tt_l1_ptr uint32_t* e = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(own);
            e[0] = best_k[r];
            e[1] = best_i[r];
            e[2] = best_i[r] != 0xffffffffu;
            e[3] = best_v[r];
            noc_async_write(own, get_noc_addr(mx, my, scr + 16 * (r * n_w + slot)), 16);
        }
        noc_async_write_barrier();
        const uint32_t sem = get_semaphore(0);
        if (slot != 0) {
            noc_semaphore_inc(get_noc_addr(mx, my, sem), 1);
            noc_async_atomic_barrier();
            return;
        }
        noc_semaphore_wait(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem), n_w - 1);
        for (uint32_t r = 0; r < raw_rows; ++r) {
            uint32_t bk = 0, bi = 0xffffffffu;
            for (uint32_t k = 0; k < n_w; ++k) {
                volatile tt_l1_ptr uint32_t* e =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + 16 * (r * n_w + k));
                if (e[2] && (e[0] > bk || (e[0] == bk && e[1] < bi))) {
                    bk = e[0];
                    bi = e[1];
                }
            }
            const uint32_t own = scr + 16 * (raw_rows * n_w + r);
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(own) = bi;
            const InterleavedAddrGen<true> tk = {
                .bank_base_address = get_common_arg_val<uint32_t>(raw_rows - 1 + r), .page_size = 4};
            noc_async_write(own, tk.get_noc_addr(0), 4);
        }
        noc_async_write_barrier();
    }
}
