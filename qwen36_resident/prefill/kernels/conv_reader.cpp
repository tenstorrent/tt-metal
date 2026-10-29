// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Causal conv1d (4 taps) + SiLU over one prefill chunk, reader (NCRISC). The chunk's C = 32 * Rt rows
// are the leading Ct tile columns of x (the in-projection output, Xw tiles wide); a core handles the
// channel columns col = core, core + cores, ... . Per column: the 4 tap tiles (row 0 = tap j of the
// column's channels; tap j multiplies x[t - 3 + j]), the carry tile (rows 0..2 = the 3 inputs before
// the chunk, oldest first), then the Rt input tiles top to bottom.
//
// Compile-time args: 0 Rt, 1 Xw, 2 Ct. Runtime args: 0 core, 1 cores, 2 x, 3 carry, 4 taps addresses
// (DRAM interleaved bf16 tiles; taps is 4 tile rows x Ct).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Rt = get_compile_time_arg_val(0);
    constexpr uint32_t Xw = get_compile_time_arg_val(1);
    constexpr uint32_t Ct = get_compile_time_arg_val(2);
    const uint32_t core = get_arg_val<uint32_t>(0);
    const uint32_t cores = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_x = 0, cb_taps = 1, cb_carry = 2;
    constexpr uint32_t tile = get_tile_size(cb_x);
    const InterleavedAddrGenFast<true> x{
        .bank_base_address = get_arg_val<uint32_t>(2), .page_size = tile, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<true> carry{
        .bank_base_address = get_arg_val<uint32_t>(3), .page_size = tile, .data_format = DataFormat::Float16_b};
    const InterleavedAddrGenFast<true> taps{
        .bank_base_address = get_arg_val<uint32_t>(4), .page_size = tile, .data_format = DataFormat::Float16_b};

    for (uint32_t col = core; col < Ct; col += cores) {
        cb_reserve_back(cb_taps, 4);
        for (uint32_t j = 0; j < 4; j++) {
            noc_async_read_tile(j * Ct + col, taps, get_write_ptr(cb_taps) + j * tile);
        }
        cb_reserve_back(cb_carry, 1);
        noc_async_read_tile(col, carry, get_write_ptr(cb_carry));
        noc_async_read_barrier();
        cb_push_back(cb_taps, 4);
        cb_push_back(cb_carry, 1);
        for (uint32_t r = 0; r < Rt; r++) {
            cb_reserve_back(cb_x, 1);
            noc_async_read_tile(r * Xw + col, x, get_write_ptr(cb_x));
            noc_async_read_barrier();
            cb_push_back(cb_x, 1);
        }
    }
}
