// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// otherlayers: candidate stage of the column-sharded decode indexer (same function as tt/idx_cs_kernels/cs_cand.cpp:
// vals[r][i] = sc[r][ids[r][i]] (bf16), gids[r][i] = ids[r][i] * COLS + column, ids >= N give -inf / 0xFFFFFFFF), but
// (1) the whole score row of n * 2 bytes is read with ONE NOC read (the stock kernel issued K = 512 random 64 B reads
// per row, 30 us), and (2) the K ids of a row are split over SPLIT cores (one core per (row, part)). ids / vals / gids
// pages are K * 4 / K * 2 / K * 4 bytes; part p handles ids [p * KP, (p + 1) * KP).
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ol_zones.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_ids = get_compile_time_arg_val(0);
    constexpr uint32_t cb_row = get_compile_time_arg_val(1);
    constexpr uint32_t cb_val = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gid = get_compile_time_arg_val(3);
    constexpr uint32_t cb_col = get_compile_time_arg_val(4);
    constexpr uint32_t K = get_compile_time_arg_val(5);
    constexpr uint32_t COLS = get_compile_time_arg_val(6);
    constexpr uint32_t N = get_compile_time_arg_val(7);
    constexpr uint32_t SPLIT = get_compile_time_arg_val(8);
    constexpr uint32_t KP = K / SPLIT;
    constexpr auto sc_args = TensorAccessorArgs<9>();
    constexpr auto ids_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr auto col_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto val_args = TensorAccessorArgs<col_args.next_compile_time_args_offset()>();
    constexpr auto gid_args = TensorAccessorArgs<val_args.next_compile_time_args_offset()>();

    const uint32_t t = ol_core_idx(get_common_arg_val<uint32_t>(5));  // core index: row = t / SPLIT, part = t % SPLIT
    const uint32_t row = t / SPLIT, part = t % SPLIT;
    const uint32_t sc_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t ids_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t col_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t val_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t gid_addr = get_common_arg_val<uint32_t>(4);
    OLZ("Z_CALL");

    Noc noc;
    const auto sc_acc = TensorAccessor(sc_args, sc_addr, N * 2);
    const auto ids_acc = TensorAccessor(ids_args, ids_addr, K * 4);
    const auto col_acc = TensorAccessor(col_args, col_addr, 64);
    const auto val_acc = TensorAccessor(val_args, val_addr, K * 2);
    const auto gid_acc = TensorAccessor(gid_args, gid_addr, K * 4);
    experimental::CB cbi(cb_ids), cbr(cb_row), cbv(cb_val), cbg(cb_gid), cbc(cb_col);
    cbi.reserve_back(1);
    cbr.reserve_back(1);
    cbv.reserve_back(1);
    cbg.reserve_back(1);
    cbc.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbi.get_write_ptr());
    volatile tt_l1_ptr uint16_t* rowp = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbr.get_write_ptr());
    volatile tt_l1_ptr uint16_t* vals = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbv.get_write_ptr());
    volatile tt_l1_ptr uint32_t* gids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbg.get_write_ptr());
    volatile tt_l1_ptr uint32_t* colp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbc.get_write_ptr());

    {
        OLZ("Z_READ");
        noc.async_read(col_acc, cbc, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read(ids_acc, cbi, KP * 4, {.page_id = row, .offset_bytes = part * KP * 4}, {.offset_bytes = 0});
        noc.async_read(sc_acc, cbr, N * 2, {.page_id = row, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
    }
    const uint32_t column = colp[0];
    {
        OLZ("Z_LOOP");
        for (uint32_t i = 0; i < KP; ++i) {
            const uint32_t id = ids[i];
            if (id < N) {
                vals[i] = rowp[id];
                gids[i] = id * COLS + column;
            } else {
                vals[i] = 0xFF80;
                gids[i] = 0xFFFFFFFFu;
            }
        }
    }
    noc.async_write(cbv, val_acc, KP * 2, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = part * KP * 2});
    noc.async_write(cbg, gid_acc, KP * 4, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = part * KP * 4});
    noc.async_write_barrier();
}
