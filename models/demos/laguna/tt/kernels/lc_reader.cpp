// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Local-only token-dispatch combine (Laguna), reader. The routed-expert output is a TILE bf8 [R, H] buffer whose
// local expert l owns the tile-aligned row region starting at start[idx[l]] with counts[idx[l]] real rows; the
// dispatch metadata row r holds (chip, token, top-k slot, expert, weight). Work unit = one 32-row tile row of one
// expert; units are numbered expert by expert and core c takes units c, c + cores, ...
// Pushes: cb_n <- this core's unit count (for compute); per unit cb_wm <- {valid rows, metadata of the 32 rows}
// (for the writer) and cb_in <- the Ht tiles of the tile row, BLK at a time; finally a cb_wm sentinel.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t E = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t BLK = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t meta_page = get_compile_time_arg_val(4);  // L1 slot stride of one metadata row
    constexpr uint32_t meta_bytes = 5 * 4;                         // (chip, token, slot, expert, weight) int32
    constexpr uint32_t grid_x = get_compile_time_arg_val(5);
    constexpr uint32_t num_cores = get_compile_time_arg_val(6);
    constexpr uint32_t vec_page = get_compile_time_arg_val(7);  // L1 slot per counts / idx / start vector
    constexpr uint32_t cnt_bytes = get_compile_time_arg_val(8);  // each vector is one page of this many bytes
    constexpr uint32_t idx_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t st_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t cb_in = 0, cb_n = 1, cb_wm = 2, cb_vec = 3;
    constexpr auto eo_args = TensorAccessorArgs<11>();
    constexpr auto meta_args = TensorAccessorArgs<eo_args.next_compile_time_args_offset()>();
    constexpr auto cnt_args = TensorAccessorArgs<meta_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<cnt_args.next_compile_time_args_offset()>();
    constexpr auto st_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();

    const auto eo = TensorAccessor(eo_args, get_common_arg_val<uint32_t>(0), tile_bytes);
    const auto meta = TensorAccessor(meta_args, get_common_arg_val<uint32_t>(1));  // aligned page size from the args
    const auto cnt_acc = TensorAccessor(cnt_args, get_common_arg_val<uint32_t>(2), cnt_bytes);
    const auto idx_acc = TensorAccessor(idx_args, get_common_arg_val<uint32_t>(3), idx_bytes);
    const auto st_acc = TensorAccessor(st_args, get_common_arg_val<uint32_t>(4), st_bytes);
    const uint32_t core = get_absolute_logical_y() * grid_x + get_absolute_logical_x();

    const uint32_t vec = get_write_ptr(cb_vec);
    noc_async_read(cnt_acc.get_noc_addr(0), vec, cnt_bytes);
    noc_async_read(idx_acc.get_noc_addr(0), vec + vec_page, idx_bytes);
    noc_async_read(st_acc.get_noc_addr(0), vec + 2 * vec_page, st_bytes);
    noc_async_read_barrier();
    invalidate_l1_cache();
    volatile tt_l1_ptr uint32_t* counts = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vec);
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vec + vec_page);
    volatile tt_l1_ptr uint32_t* start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(vec + 2 * vec_page);

    uint32_t mine = 0, u = 0;
    for (uint32_t l = 0; l < E; ++l) {
        const uint32_t nt = (counts[idx[l]] + 31) / 32;
        for (uint32_t j = 0; j < nt; ++j, ++u) {
            mine += (u % num_cores == core) ? 1 : 0;
        }
    }
    cb_reserve_back(cb_n, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_n))[0] = mine;
    cb_push_back(cb_n, 1);

    u = 0;
    for (uint32_t l = 0; l < E; ++l) {
        const uint32_t g = idx[l];
        const uint32_t cnt = counts[g], row0 = start[g];
        const uint32_t nt = (cnt + 31) / 32;
        for (uint32_t j = 0; j < nt; ++j, ++u) {
            if (u % num_cores != core) {
                continue;
            }
            const uint32_t r0 = row0 + j * 32;
            const uint32_t valid = (cnt - j * 32) < 32 ? (cnt - j * 32) : 32;
            cb_reserve_back(cb_wm, 1);
            const uint32_t wm = get_write_ptr(cb_wm);
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wm)[0] = valid;
            // metadata rows at 64-byte-aligned L1 slots (the DRAM pages are 64-byte aligned)
            const uint32_t rows_l1 = (wm + 32 + 63) & ~63u;
            for (uint32_t r = 0; r < valid; ++r) {
                noc_async_read(meta.get_noc_addr(r0 + r), rows_l1 + r * meta_page, meta_bytes);
            }
            const uint32_t tr = r0 / 32;
            for (uint32_t b = 0; b < Ht; b += BLK) {
                cb_reserve_back(cb_in, BLK);
                const uint32_t dst = get_write_ptr(cb_in);
                for (uint32_t i = 0; i < BLK; ++i) {
                    noc_async_read_tile(tr * Ht + b + i, eo, dst + i * tile_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_in, BLK);
            }
            cb_push_back(cb_wm, 1);
        }
    }
    cb_reserve_back(cb_wm, 1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_wm))[0] = 0xFFFFFFFFu;
    cb_push_back(cb_wm, 1);
}
