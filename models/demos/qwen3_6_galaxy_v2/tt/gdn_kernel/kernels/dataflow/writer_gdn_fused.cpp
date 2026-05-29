// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Fused GDN writer kernel.
//
// Writes output tiles to [1, B, value_dim_tp] layout (not [num_pairs, 1, Dv]),
// mapping pair → (batch_idx, v_head) to place tiles at correct positions.
// Also writes updated recurrence state back to DRAM/L1.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t out_addr = get_arg_val<uint32_t>(0);    // [1, B, value_dim_tp]
    uint32_t state_addr = get_arg_val<uint32_t>(1);  // [num_pairs_total, Dk, Dv]
    uint32_t pair_start = get_arg_val<uint32_t>(2);
    uint32_t num_pairs = get_arg_val<uint32_t>(3);

    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t STATE_IN_L1 = get_compile_time_arg_val(3);
    // compile_time_arg_val(4) = Nv_TP (unused, kept for compat)
    // compile_time_arg_val(5) = out_tiles_per_batch (unused, kept for compat)
    constexpr uint32_t STATE_IS_SHARDED = get_compile_time_arg_val(6);
    constexpr uint32_t state_tiles = Kt * Vt;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_state_out = tt::CBIndex::c_8;

    // Output always to DRAM
    constexpr bool is_dram = true;
    const InterleavedAddrGenFast<is_dram> out_wr = {
        .bank_base_address = out_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    // State can be L1 or DRAM
    constexpr bool state_is_dram = (STATE_IN_L1 == 0) && (STATE_IS_SHARDED == 0);
    const InterleavedAddrGenFast<state_is_dram> state_wr = {
        .bank_base_address = state_addr, .page_size = tile_bytes, .data_format = DataFormat::Float16_b};

    for (uint32_t pair = 0; pair < num_pairs; pair++) {
        uint32_t p = pair_start + pair;

        // Output is [num_pairs, 1, Dv] — sequential per-pair layout
        uint32_t out_tile_base = p * Vt;

        // Write output [Vt tiles] + state [state_tiles] with single barrier
        cb_wait_front(cb_out, Vt);
        cb_wait_front(cb_state_out, state_tiles);

        uint32_t rp = get_read_ptr(cb_out);
        for (uint32_t vt = 0; vt < Vt; vt++) {
            noc_async_write_tile(out_tile_base + vt, out_wr, rp);
            rp += tile_bytes;
        }

        uint32_t sp = get_read_ptr(cb_state_out);
        if constexpr (STATE_IS_SHARDED) {
            // HEIGHT_SHARDED: write to local L1 shard — direct copy (no NOC)
            uint32_t shard_byte_offset = pair * state_tiles * tile_bytes;
            uint32_t dst_addr = state_addr + shard_byte_offset;
            uint32_t num_words = (state_tiles * tile_bytes) >> 2;
            volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sp);
            volatile tt_l1_ptr uint32_t* dst = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst_addr);
            for (uint32_t w = 0; w < num_words; w++) {
                dst[w] = src[w];
            }
        } else {
            for (uint32_t s = 0; s < state_tiles; s++) {
                noc_async_write_tile(p * state_tiles + s, state_wr, sp);
                sp += tile_bytes;
            }
        }

        noc_async_write_barrier();
        cb_pop_front(cb_out, Vt);
        cb_pop_front(cb_state_out, state_tiles);
    }
}
