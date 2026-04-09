// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Writer kernel for fused GDN recurrence.
// Uses InterleavedAddrGenFast for DRAM (output) and optionally L1 (state).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t state_addr = get_arg_val<uint32_t>(1);
    uint32_t pair_start = get_arg_val<uint32_t>(2);
    uint32_t num_pairs = get_arg_val<uint32_t>(3);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t STATE_IN_L1 = get_compile_time_arg_val(3);  // 1 = state in L1, 0 = DRAM
    constexpr uint32_t state_tiles = Kt * Vt;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_state_out = tt::CBIndex::c_8;

    // Output always to DRAM
    constexpr bool is_dram = true;
    const InterleavedAddrGenFast<is_dram> out_wr = {
        .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    // State can be in L1 or DRAM (compile-time switch)
    constexpr bool state_is_dram = (STATE_IN_L1 == 0);
    const InterleavedAddrGenFast<state_is_dram> state_wr = {
        .bank_base_address = state_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    for (uint32_t pair = 0; pair < num_pairs; pair++) {
        uint32_t p = pair_start + pair;

        // Wait for output from compute
        cb_wait_front(cb_out, Vt);
        uint32_t rp = get_read_ptr(cb_out);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            noc_async_write_tile(p * Vt + vt, out_wr, rp);
            rp += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Vt);

        // Write updated state back to DRAM
        cb_wait_front(cb_state_out, state_tiles);
        uint32_t sp = get_read_ptr(cb_state_out);
        for (uint32_t s = 0; s < state_tiles; s++) {
            noc_async_write_tile(p * state_tiles + s, state_wr, sp);
            sp += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_state_out, state_tiles);
    }
}
