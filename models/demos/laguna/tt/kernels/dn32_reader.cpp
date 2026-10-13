// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Batched decode down + expert-sum reader over column-page weights (Laguna). Core c owns output columns
// c*CPC .. c*CPC+CPC-1. Per active expert pushes its Kt activation tiles (the [1, E, 32, I] SwiGLU output row) and
// its CPC down column pages (CPC*Kt tiles, one contiguous read per column). Mt > 1 (more than 32 tokens): a unit pushes
// its Kq activation tiles for every token tile m (cb_x tile m * Kq + k; activation tile (e * Mt + m) * Kt + k).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Nh = get_compile_time_arg_val(1);
    constexpr uint32_t E = get_compile_time_arg_val(2);
    constexpr uint32_t x_page = get_compile_time_arg_val(3);
    constexpr uint32_t w_tile = get_compile_time_arg_val(4);
    constexpr uint32_t sp_page = get_compile_time_arg_val(5);
    constexpr uint32_t grid_x = get_compile_time_arg_val(6);
    constexpr uint32_t CPC = get_compile_time_arg_val(7);
    constexpr uint32_t EG = get_compile_time_arg_val(8);  // expert groups: active expert a goes to group a % EG
    // shared expert (Kt_sh > 0): one more unit, on the group after the last routed one (active count % EG): its
    // Kt_sh activation tiles and its CPC down columns (column pages, own format in cb_w_sh)
    constexpr uint32_t Kt_sh = get_compile_time_arg_val(9);
    constexpr uint32_t w_sh_tile = get_compile_time_arg_val(10);
    constexpr uint32_t cb_x = 0, cb_w = 1, cb_meta = 2, cb_sp = 3, cb_w_sh = 6;
    // KS > 1: each active expert is KS units of Kt / KS K tiles (partial sums add in the same DST tiles), dealt over
    // the expert groups in order -- finer than whole experts, so the groups stay balanced
    constexpr uint32_t KS = get_compile_time_arg_val(11);
    constexpr uint32_t Kq = Kt / KS;
    constexpr uint32_t Mt = get_compile_time_arg_val(12);
    // sparse_x (Mt > 1): a routed expert's activation rows are zero for every token not routed to it (gate_up32
    // applies the routing weight), so a unit zero-fills its Mt x Kq activation tiles and copies in only the routed
    // tokens' rows (two 32-byte face rows per tile) instead of reading every tile whole
    constexpr bool sparse_x = get_compile_time_arg_val(13) == 1 && Mt > 1;
    constexpr auto x_args = TensorAccessorArgs<14>();
    constexpr auto w_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto sp_args = TensorAccessorArgs<w_args.next_compile_time_args_offset()>();
    constexpr auto xs_args = TensorAccessorArgs<sp_args.next_compile_time_args_offset()>();
    constexpr auto ws_args = TensorAccessorArgs<xs_args.next_compile_time_args_offset()>();
    const uint32_t x_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t sp_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t x_rows = get_common_arg_val<uint32_t>(4);  // rows of the activation tiles that are read
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();
    const uint32_t c0 = (core / EG) * CPC, eg = core % EG;
    const auto x = TensorAccessor(x_args, x_addr, x_page);
    const auto w = TensorAccessor(w_args, w_addr, w_tile);
    const auto sp = TensorAccessor(sp_args, sp_addr, sp_page);
    const uint32_t sp_l1 = get_write_ptr(cb_sp);
    // sp_rows > 1: per-token routing rows (router32.route_local_rows); an expert is active if any row routes to it
    const uint32_t sp_rows = get_common_arg_val<uint32_t>(3);
    for (uint32_t r = 0; r < sp_rows; ++r) {
        noc_async_read(sp.get_noc_addr(r), sp_l1 + r * sp_page, sp_page);
    }
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint16_t* rowsv = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(sp_l1);
    volatile tt_l1_ptr uint16_t* spv = rowsv;
    if constexpr (sparse_x) {
        // the union in the page after the rows (the rows stay intact: they tell which tokens an expert has)
        spv = rowsv + sp_rows * (sp_page / 2);
        for (uint32_t e = 0; e < E; ++e) {
            uint16_t any = 0;
            for (uint32_t r = 0; r < sp_rows; ++r) {
                any |= rowsv[r * (sp_page / 2) + e];
            }
            spv[e] = any;
        }
    } else {
        for (uint32_t r = 1; r < sp_rows; ++r) {
            for (uint32_t e = 0; e < E; ++e) {
                spv[e] |= spv[r * (sp_page / 2) + e];
            }
        }
    }
    uint32_t na = 0, seen = 0;  // seen counts units (active experts x KS)
    for (uint32_t e = 0; e < E; ++e) {
        if (spv[e] != 0) {
            for (uint32_t q = 0; q < KS; ++q) {
                na += (seen % EG == eg) ? 1 : 0;
                ++seen;
            }
        }
    }
    const bool has_sh = Kt_sh > 0 && seen % EG == eg;
    cb_reserve_back(cb_meta, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[0] = na;
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_meta))[1] = has_sh ? 1 : 0;
    cb_push_back(cb_meta, 1);
    seen = 0;
    for (uint32_t e = 0; e < E; ++e) {
        if (spv[e] == 0) {
            continue;
        }
        for (uint32_t q = 0; q < KS; ++q) {
            const bool mine = (seen % EG == eg);
            ++seen;
            if (!mine) {
                continue;
            }
            const uint32_t kb = q * Kq;  // this unit's first K tile
            cb_reserve_back(cb_w, CPC * Kq);
            const uint32_t w_l1 = get_write_ptr(cb_w);
            for (uint32_t j = 0; j < CPC; ++j) {
                noc_async_read(w.get_noc_addr(e * Kt * Nh + c0 + j) + kb * w_tile, w_l1 + j * Kq * w_tile, Kq * w_tile);
            }
            cb_reserve_back(cb_x, Mt * Kq);
            const uint32_t x_l1 = get_write_ptr(cb_x);
            if constexpr (sparse_x) {
                const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
                for (uint32_t off = 0; off < Mt * Kq * x_page; off += MEM_ZEROS_SIZE) {
                    noc_async_read(zeros, x_l1 + off, MEM_ZEROS_SIZE);
                }
                noc_async_read_barrier();  // the zeros land before the routed rows
                for (uint32_t t = 0; t < sp_rows; ++t) {
                    if (rowsv[t * (sp_page / 2) + e] == 0) {
                        continue;
                    }
                    const uint32_t m = t / 32, r = t % 32;
                    const uint32_t ro = (r / 16) * 1024 + (r % 16) * 32;
                    for (uint32_t k = 0; k < Kq; ++k) {
                        const uint64_t src = x.get_noc_addr((e * Mt + m) * Kt + kb + k) + ro;
                        const uint32_t dst = x_l1 + (m * Kq + k) * x_page + ro;
                        noc_async_read(src, dst, 32);
                        noc_async_read(src + 512, dst + 512, 32);
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_x, Mt * Kq);
                cb_push_back(cb_w, CPC * Kq);
                continue;
            }
            for (uint32_t m = 0; m < Mt; ++m) {
                for (uint32_t k = 0; k < Kq; ++k) {
                    const uint32_t dst = x_l1 + (m * Kq + k) * x_page;
                    if (x_rows < 16) {
                        // only rows 0..x_rows-1 are real: their 32-byte rows of face 0 and face 1 (other rows keep
                        // stale data; matmul rows are independent, so they only reach output rows never read)
                        const uint64_t src = x.get_noc_addr(e * Kt + kb + k);
                        noc_async_read(src, dst, x_rows * 32);
                        noc_async_read(src + 512, dst + 512, x_rows * 32);
                    } else {
                        noc_async_read_tile((e * Mt + m) * Kt + kb + k, x, dst);
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, Mt * Kq);
            cb_push_back(cb_w, CPC * Kq);
        }
    }
    if constexpr (Kt_sh > 0) {
        if (has_sh) {
            const auto xs = TensorAccessor(xs_args, get_common_arg_val<uint32_t>(5), x_page);
            const auto ws = TensorAccessor(ws_args, get_common_arg_val<uint32_t>(6), w_sh_tile);
            cb_reserve_back(cb_w_sh, CPC * Kt_sh);
            const uint32_t w_l1 = get_write_ptr(cb_w_sh);
            for (uint32_t j = 0; j < CPC; ++j) {
                noc_async_read(ws.get_noc_addr(c0 + j), w_l1 + j * Kt_sh * w_sh_tile, Kt_sh * w_sh_tile);
            }
            cb_reserve_back(cb_x, Mt * Kt_sh);
            const uint32_t x_l1 = get_write_ptr(cb_x);
            for (uint32_t k = 0; k < Mt * Kt_sh; ++k) {
                if (x_rows < 16) {
                    const uint64_t src = xs.get_noc_addr(k);
                    noc_async_read(src, x_l1 + k * x_page, x_rows * 32);
                    noc_async_read(src + 512, x_l1 + k * x_page + 512, x_rows * 32);
                } else {
                    noc_async_read_tile(k, xs, x_l1 + k * x_page);  // tile m * Kt_sh + k of [1, 1, 32 Mt, I_sh]
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_x, Mt * Kt_sh);
            cb_push_back(cb_w_sh, CPC * Kt_sh);
        }
    }
}
