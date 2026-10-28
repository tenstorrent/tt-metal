// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Probe: gather `n_rows` row records into one L1 block per pass, the data-movement core only issuing NoC reads.
// BFP4 mode: DRAM row = [32 exponents | the same 32 | 256 nibble bytes] (320 B); per row two reads: the exponent copy
// whose address matches the destination mod 64 -> exps + 32 p, the nibbles -> data + 256 p. The block is then a
// row-major bfloat4_b tensor [n_rows x 512] (exponent section first). BF16 mode (the baseline): one 1024 B read a row.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t table_addr = get_arg_val<uint32_t>(0);
    const uint32_t ids_addr = get_arg_val<uint32_t>(1);
    constexpr uint32_t cb_rows = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ids = get_compile_time_arg_val(1);
    constexpr uint32_t n_rows = get_compile_time_arg_val(2);
    constexpr uint32_t passes = get_compile_time_arg_val(3);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(4);  // DRAM page of a row
    constexpr bool bfp4 = get_compile_time_arg_val(5) != 0;
    constexpr uint32_t ppp = get_compile_time_arg_val(6);  // CB pages a pass
    constexpr auto table_args = TensorAccessorArgs<7>();
    constexpr auto ids_args = TensorAccessorArgs<table_args.next_compile_time_args_offset()>();
    const auto table = TensorAccessor(table_args, table_addr, page_bytes);
    const auto ids_acc = TensorAccessor(ids_args, ids_addr, n_rows * 4);

    cb_reserve_back(cb_ids, 1);
    const uint32_t ids_l1 = get_write_ptr(cb_ids);
    noc_async_read(ids_acc.get_noc_addr(0), ids_l1, n_rows * 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_l1);

    for (uint32_t pass = 0; pass < passes; ++pass) {
        cb_reserve_back(cb_rows, ppp);
        const uint32_t blk = get_write_ptr(cb_rows);
#ifndef PROBE_NO_GATHER
        for (uint32_t p = 0; p < n_rows; ++p) {
            const uint32_t id = ids[p];
            if constexpr (bfp4) {
                const uint32_t exps = blk + p * 32;
                const uint32_t data = blk + n_rows * 32 + p * 256;
                noc_async_read(table.get_noc_addr(id, (p & 1) * 32), exps, 32);
                noc_async_read(table.get_noc_addr(id, 64), data, 256);
            } else {
                noc_async_read(table.get_noc_addr(id), blk + p * 1024, 1024);
            }
        }
#endif
        noc_async_read_barrier();
        cb_push_back(cb_rows, ppp);
    }
}
