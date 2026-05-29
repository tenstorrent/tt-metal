// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Writer kernel for fused GDN recurrence.
// Writes output tiles and updated state tiles back to DRAM.
// Uses TensorAccessor API for Blackhole compatibility.
//
// Sync: waits for cb_out (output from compute). Once output is available,
// state modification in cb_state is guaranteed complete (compute produces
// output AFTER all state updates).

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);
    uint32_t state_addr = get_arg_val<uint32_t>(1);
    uint32_t pair_start = get_arg_val<uint32_t>(2);
    uint32_t num_pairs = get_arg_val<uint32_t>(3);

    // Compile-time args: Kt, Vt, tile_bytes, then 2 TensorAccessorArgs
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t state_tiles = Kt * Vt;

    constexpr auto out_args = TensorAccessorArgs<3>();
    constexpr auto state_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_state_out = tt::CBIndex::c_8;  // updated state from compute

    const auto out_wr = TensorAccessor(out_args, out_addr, tile_bytes);
    const auto state_wr = TensorAccessor(state_args, state_addr, tile_bytes);

    for (uint32_t pair = 0; pair < num_pairs; pair++) {
        uint32_t p = pair_start + pair;

        // Wait for output from compute (guarantees state update is complete)
        cb_wait_front(cb_out, Vt);
        uint32_t rp = get_read_ptr(cb_out);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            noc_async_write_page(p * Vt + vt, out_wr, rp);
            rp += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_out, Vt);

        // Write updated state back to DRAM
        cb_wait_front(cb_state_out, state_tiles);
        uint32_t sp = get_read_ptr(cb_state_out);
        for (uint32_t s = 0; s < state_tiles; s++) {
            noc_async_write_page(p * state_tiles + s, state_wr, sp);
            sp += tile_bytes;
        }
        noc_async_write_barrier();
        cb_pop_front(cb_state_out, state_tiles);
    }
}
