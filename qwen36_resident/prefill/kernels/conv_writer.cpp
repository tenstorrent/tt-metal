// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Causal conv1d + SiLU over one prefill chunk, writer (BRISC). Reads the chunk control (valid rows v,
// update flag) and builds the 0/1 matrices (bf16 32x32, entry (t, s) = 1 when output row t takes input
// row s):
//   0 I      s == t              1..3 L_d  s == t - d (t >= d)      4..6 U_d  s == 32 + t - d (t < d)
//   7 D29    s == t - 29         8 A      s == o - 2 + t (t < 3)    9 B      s == 32 + o - 2 + t (t < 3)
// with o = (v - 1) % 32 the last valid row's offset in its tile: A @ x_r + B @ x_{r-1} are the 3 inputs
// ending at the last valid row. Stores y split into its q, k and v column ranges (the delta-rule op takes
// them as separate tensors) and, when the flag is set, the candidate of the last valid tile as the new
// carry (in place: the reader of this column has already read the old one).
//
// Compile-time args: 0 Rt, 1 Ct, 2 q tiles per row (= k tiles; v takes the rest). Runtime args: 0 core,
// 1 cores, 2 q, 3 k, 4 v, 5 carry, 6 control (DRAM uint32 page: [0] valid rows, [1] update flag).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {

constexpr uint32_t kConsts = 10;
constexpr uint16_t kOneBf16 = 0x3f80;

inline uint32_t elem_offset(uint32_t t, uint32_t s) {
    return ((t / 16) * 2 + (s / 16)) * 256 + (t % 16) * 16 + (s % 16);
}

inline void set_one(volatile tt_l1_ptr uint16_t* tile, uint32_t t, int32_t s) {
    if (s >= 0 && s < 32) {
        tile[elem_offset(t, static_cast<uint32_t>(s))] = kOneBf16;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Rt = get_compile_time_arg_val(0);
    constexpr uint32_t Ct = get_compile_time_arg_val(1);
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t cores = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_consts = 3, cb_out = 6, cb_cand = 7;
    constexpr uint32_t tile = get_tile_size(cb_out);
    constexpr uint32_t Qt = get_compile_time_arg_val(2);
    auto out = [](uint32_t i) {
        return InterleavedAddrGenFast<true>{
            .bank_base_address = get_arg_val<uint32_t>(2 + i), .page_size = tile, .data_format = DataFormat::Float16_b};
    };
    const InterleavedAddrGenFast<true> q_out = out(0), k_out = out(1), v_out = out(2);
    const InterleavedAddrGenFast<true> carry{
        .bank_base_address = get_arg_val<uint32_t>(5), .page_size = tile, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGen<true> ctrl{.bank_base_address = get_arg_val<uint32_t>(6), .page_size = 32};

    cb_reserve_back(cb_consts, kConsts);
    const uint32_t base = get_write_ptr(cb_consts);
    noc_async_read(ctrl.get_noc_addr(0), base, 32);
    noc_async_read_barrier();
    const uint32_t v = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base)[0];
    const uint32_t update = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base)[1];
    const uint32_t r_last = (v - 1) / 32;
    const int32_t o = static_cast<int32_t>((v - 1) % 32);
    {
        constexpr uint32_t total = kConsts * tile;
        volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
        for (uint32_t i = 0; i < 16; i++) {
            w[i] = 0;
        }
        for (uint32_t done = 64; done < total;) {
            const uint32_t n = done < total - done ? done : total - done;
            noc_async_read(get_noc_addr(base), base + done, n);
            noc_async_read_barrier();
            done += n;
        }
        auto m = [&](uint32_t i) { return reinterpret_cast<volatile tt_l1_ptr uint16_t*>(base + i * tile); };
        for (uint32_t t = 0; t < 32; t++) {
            const int32_t ti = static_cast<int32_t>(t);
            set_one(m(0), t, ti);
            for (int32_t d = 1; d <= 3; d++) {
                if (ti >= d) {
                    set_one(m(d), t, ti - d);
                } else {
                    set_one(m(3 + d), t, 32 + ti - d);
                }
            }
            set_one(m(7), t, ti - 29);
            if (t < 3) {
                set_one(m(8), t, o - 2 + ti);
                set_one(m(9), t, 32 + o - 2 + ti);
            }
        }
    }
    cb_push_back(cb_consts, kConsts);

    for (uint32_t col = core; col < Ct; col += cores) {
        for (uint32_t r = 0; r < Rt; r++) {
            cb_wait_front(cb_out, 1);
            if (col < Qt) {
                noc_async_write_tile(r * Qt + col, q_out, get_read_ptr(cb_out));
            } else if (col < 2 * Qt) {
                noc_async_write_tile(r * Qt + col - Qt, k_out, get_read_ptr(cb_out));
            } else {
                noc_async_write_tile(r * (Ct - 2 * Qt) + col - 2 * Qt, v_out, get_read_ptr(cb_out));
            }
            cb_wait_front(cb_cand, 1);
            if (r == r_last && update) {
                noc_async_write_tile(col, carry, get_read_ptr(cb_cand));
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
            cb_pop_front(cb_cand, 1);
        }
    }
}
