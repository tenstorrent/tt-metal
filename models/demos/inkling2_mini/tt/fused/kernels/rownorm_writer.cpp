// SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
// SPDX-License-Identifier: Apache-2.0
//
// Row norm / split, writer (one core per row): weighted rms(row r) (dense, out dtype) -> row out_row (0; verify row
// r: r) of its own row-form output tensor.
#include "rowio.hpp"

void kernel_main() {
    constexpr uint32_t ntc = get_named_compile_time_arg_val("ntc");
    constexpr uint32_t T = get_named_compile_time_arg_val("pages");
    constexpr uint32_t cb_np = 16;

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_tile = get_arg_val<uint32_t>(1);
    const uint32_t out_es = get_arg_val<uint32_t>(2);
    const uint32_t out_row = get_arg_val<uint32_t>(3);  // tile row of the output (verify row r: r)

    cb_wait_front(cb_np, T);
    // the reader scatters tiles 0 .. ntc/2 (E10d: both RISCs)
    constexpr bool dram = get_named_compile_time_arg_val("split_out") != 2;  // 2: L1-interleaved output
    rowio::scatter_row<dram>(out_addr, out_tile, out_es, ntc, out_row, get_read_ptr(cb_np), ntc / 2, ntc);
    noc_async_write_barrier();
}
