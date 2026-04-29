// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Prefill GDN writer kernel — writes N output tokens + final state per pair-shard.
//
// Output layout (global): [num_pairs * N, 1, Dv] flat
//   tile_id(p, t, vt_global) = (p * N + t) * Vt_global + vt_global
// Each shard writes its Vt local tiles at offset vt_start = v_shard_idx * Vt.
//
// State layout (global): [num_pairs, Kt, Vt_global] tiles.
//   tile_id(p, kt, vt_global) = p * (Kt * Vt_global) + kt * Vt_global + vt_global
// Each shard writes Kt * Vt local tiles at the matching kt-major slice.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);    // [num_pairs_total * N, 1, Dv]
    uint32_t state_addr = get_arg_val<uint32_t>(1);  // [num_pairs_total, Dk, Dv]
    uint32_t pair_start = get_arg_val<uint32_t>(2);
    uint32_t num_pairs = get_arg_val<uint32_t>(3);
    uint32_t v_shard_idx = get_arg_val<uint32_t>(4);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);  // per-shard V-tile count
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t STATE_IN_L1 = get_compile_time_arg_val(3);
    constexpr uint32_t num_tokens = get_compile_time_arg_val(4);
    constexpr uint32_t Vt_global = get_compile_time_arg_val(5);  // global V-tile count per pair
    constexpr uint32_t state_tiles = Kt * Vt;                    // per-shard state tiles
    constexpr uint32_t state_tiles_global = Kt * Vt_global;
    const uint32_t vt_start = v_shard_idx * Vt;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_state_out = tt::CBIndex::c_8;

    constexpr bool is_dram = true;
    const InterleavedAddrGenFast<is_dram> out_wr = {
        .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    constexpr bool state_is_dram = (STATE_IN_L1 == 0);
    const InterleavedAddrGenFast<state_is_dram> state_wr = {
        .bank_base_address = state_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    for (uint32_t pair = 0; pair < num_pairs; pair++) {
        uint32_t p = pair_start + pair;

        // Write N output tokens — each token has Vt local tiles, mapped to global V-slice.
        for (uint32_t tok = 0; tok < num_tokens; tok++) {
            cb_wait_front(cb_out, Vt);
            uint32_t rp = get_read_ptr(cb_out);

            uint32_t out_tile_base_global = (p * num_tokens + tok) * Vt_global;
            for (uint32_t vt_local = 0; vt_local < Vt; vt_local++) {
                noc_async_write_tile(out_tile_base_global + vt_start + vt_local, out_wr, rp);
                rp += tile_bytes;
            }

            noc_async_write_barrier();
            cb_pop_front(cb_out, Vt);
        }

        // Write final state for this shard's V-slice (once, after all tokens).
        cb_wait_front(cb_state_out, state_tiles);
        uint32_t sp = get_read_ptr(cb_state_out);
        for (uint32_t kt = 0; kt < Kt; kt++) {
            for (uint32_t vt_local = 0; vt_local < Vt; vt_local++) {
                uint32_t s_global = kt * Vt_global + (vt_start + vt_local);
                uint32_t s_local = kt * Vt + vt_local;
                noc_async_write_tile(p * state_tiles_global + s_global, state_wr, sp + s_local * tile_bytes);
            }
        }
        noc_async_write_barrier();
        cb_pop_front(cb_state_out, state_tiles);
    }
}
