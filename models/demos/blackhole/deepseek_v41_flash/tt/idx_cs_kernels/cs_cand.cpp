// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Column-sharded decode indexer, candidate stage: for every row r the K local top-k slots ``ids[r]`` (uint32, ROW_MAJOR
// [R,1,1,K]) of this column's score row ``sc[r]`` (bf16 ROW_MAJOR [R,1,1,n], page = one row of n * 2 bytes, n a
// multiple of 32) -> vals[r][i] = sc[r][ids[r][i]] (bf16) and gids[r][i] = ids[r][i] * COLS + column (uint32, the
// GLOBAL entry id of the entry interleave: entry e on column e % COLS at slot e / COLS). One core per row. The score is
// fetched with a 64-byte aligned NOC read of the beat that holds it (BH DRAM reads are 64 B aligned). ids >= n (a
// sentinel) give val = -inf (bf16 0xFF80) and gid = 0xFFFFFFFF.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>

void kernel_main() {
    constexpr uint32_t cb_ids = get_compile_time_arg_val(0);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(1);
    constexpr uint32_t cb_val = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gid = get_compile_time_arg_val(3);
    constexpr uint32_t cb_col = get_compile_time_arg_val(4);
    constexpr uint32_t K = get_compile_time_arg_val(5);
    constexpr uint32_t COLS = get_compile_time_arg_val(6);
    constexpr uint32_t N = get_compile_time_arg_val(7);  // entries per score row
    constexpr auto sc_args = TensorAccessorArgs<8>();
    constexpr auto ids_args = TensorAccessorArgs<sc_args.next_compile_time_args_offset()>();
    constexpr auto col_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();
    constexpr auto val_args = TensorAccessorArgs<col_args.next_compile_time_args_offset()>();
    constexpr auto gid_args = TensorAccessorArgs<val_args.next_compile_time_args_offset()>();

    const uint32_t t_first = get_arg_val<uint32_t>(0);
    const uint32_t t_stride = get_arg_val<uint32_t>(1);
    const uint32_t count = get_arg_val<uint32_t>(2);
    const uint32_t sc_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t ids_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t col_addr = get_common_arg_val<uint32_t>(2);
    const uint32_t val_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t gid_addr = get_common_arg_val<uint32_t>(4);

    Noc noc;
    const auto sc_acc = TensorAccessor(sc_args, sc_addr, N * 2);
    const auto ids_acc = TensorAccessor(ids_args, ids_addr, K * 4);
    const auto col_acc = TensorAccessor(col_args, col_addr, 64);
    const auto val_acc = TensorAccessor(val_args, val_addr, K * 2);
    const auto gid_acc = TensorAccessor(gid_args, gid_addr, K * 4);
    experimental::CB cbi(cb_ids), cbs(cb_stage), cbv(cb_val), cbg(cb_gid), cbc(cb_col);
    cbi.reserve_back(1);
    cbs.reserve_back(1);
    cbv.reserve_back(1);
    cbg.reserve_back(1);
    cbc.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbi.get_write_ptr());
    volatile tt_l1_ptr uint8_t* stage = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(cbs.get_write_ptr());
    volatile tt_l1_ptr uint16_t* vals = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(cbv.get_write_ptr());
    volatile tt_l1_ptr uint32_t* gids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbg.get_write_ptr());
    volatile tt_l1_ptr uint32_t* colp = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cbc.get_write_ptr());

    noc.async_read(col_acc, cbc, 64, {.page_id = 0, .offset_bytes = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    const uint32_t column = colp[0];

    for (uint32_t it = 0; it < count; ++it) {
        const uint32_t t = t_first + it * t_stride;
        noc.async_read(ids_acc, cbi, K * 4, {.page_id = t, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        for (uint32_t i = 0; i < K; ++i) {
            const uint32_t id = ids[i];
            const uint32_t off = (id < N) ? ((id * 2u) & ~63u) : 0u;
            noc.async_read(sc_acc, cbs, 64, {.page_id = t, .offset_bytes = off}, {.offset_bytes = i * 64});
        }
        noc.async_read_barrier();
        for (uint32_t i = 0; i < K; ++i) {
            const uint32_t id = ids[i];
            if (id < N) {
                vals[i] = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(stage + i * 64 + ((id * 2u) & 63u));
                gids[i] = id * COLS + column;
            } else {
                vals[i] = 0xFF80;
                gids[i] = 0xFFFFFFFFu;
            }
        }
        noc.async_write(cbv, val_acc, K * 2, {.offset_bytes = 0}, {.page_id = t, .offset_bytes = 0});
        noc.async_write(cbg, gid_acc, K * 4, {.offset_bytes = 0}, {.page_id = t, .offset_bytes = 0});
        noc.async_write_barrier();
    }
}
