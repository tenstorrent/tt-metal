// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Reads a row-major block of 32 rows x (W*32) cols from DRAM into cb_in as a
// single contiguous row-major stripe. This mirrors the standard tilize reader
// contract (reader_unary_stick_layout_split_rows_singlecore): 32 sticks of the
// full row width laid down back-to-back. The unpacker's tilize address
// generator interprets this stripe as W column-tiles.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(2);

    // page_id == row index; page_size == full row width in bytes.
    const InterleavedAddrGen<true> s = {.bank_base_address = src_addr, .page_size = row_bytes};

    cb_reserve_back(cb_in, W);
    uint32_t l1_write_addr = get_write_ptr(cb_in);
    for (uint32_t r = 0; r < 32; ++r) {
        uint64_t noc_addr = s.get_noc_addr(r);
        noc_async_read(noc_addr, l1_write_addr, row_bytes);
        l1_write_addr += row_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_in, W);
}
