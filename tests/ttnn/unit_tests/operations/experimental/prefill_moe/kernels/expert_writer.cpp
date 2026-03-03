// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Prefill MoE compute: Two-phase weight reader + output writer + barrier signal (dm1/RISCV_1)
//
// Phase A: Read gate_up weight tiles [K, N_weight] from DRAM → CB_WEIGHTS,
//          then write SwiGLU output from CB_OUT → DRAM inter_buf,
//          then signal cross-core barrier (SEM_BARRIER on leader).
// Phase B: Read down weight tiles [K, N_down] from DRAM → CB_WEIGHTS (starts immediately),
//          then write down output from CB_OUT → DRAM output.
//
// Semaphores:
//   SEM_BARRIER (id=0): Incremented on leader core after SwiGLU DRAM write completes.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Runtime args
    const uint32_t gate_up_w_addr = get_arg_val<uint32_t>(0);
    const uint32_t inter_write_addr = get_arg_val<uint32_t>(1);
    const uint32_t down_w_addr = get_arg_val<uint32_t>(2);
    const uint32_t output_addr = get_arg_val<uint32_t>(3);
    const uint32_t k_tiles = get_arg_val<uint32_t>(4);
    const uint32_t n_weight_per_core_gu = get_arg_val<uint32_t>(5);   // 12
    const uint32_t n_weight_tiles_gu = get_arg_val<uint32_t>(6);      // 180 (total N tiles in gate_up weight)
    const uint32_t core_weight_offset_gu = get_arg_val<uint32_t>(7);  // This core's N offset in gate_up
    const uint32_t core_out_offset_gu = get_arg_val<uint32_t>(8);     // This core's output start tile (inter)
    const uint32_t n_out_per_core = get_arg_val<uint32_t>(9);         // 6 (SwiGLU output tiles)
    const uint32_t n_per_core_dn = get_arg_val<uint32_t>(10);         // 6
    const uint32_t n_tiles_dn = get_arg_val<uint32_t>(11);            // 90 (total N tiles in down weight)
    const uint32_t core_dn_offset = get_arg_val<uint32_t>(12);        // This core's N offset in down
    const uint32_t leader_phys_x = get_arg_val<uint32_t>(13);
    const uint32_t leader_phys_y = get_arg_val<uint32_t>(14);

    // Compile-time args: [TensorAccessorArgs(gate_up_w), TensorAccessorArgs(inter),
    //                      TensorAccessorArgs(down_w), TensorAccessorArgs(output)]
    constexpr auto gate_up_w_args = TensorAccessorArgs<0>();
    constexpr auto inter_out_args = TensorAccessorArgs<1>();
    constexpr auto down_w_args = TensorAccessorArgs<2>();
    constexpr auto output_out_args = TensorAccessorArgs<3>();

    constexpr uint32_t cb_weights = 1;
    constexpr uint32_t cb_out = 2;
    constexpr uint32_t SEM_BARRIER = 0;

    const uint32_t weight_page_bytes = get_local_cb_interface(cb_weights).fifo_page_size;
    const uint32_t output_page_bytes = get_local_cb_interface(cb_out).fifo_page_size;

    const auto gu_w_accessor = TensorAccessor(gate_up_w_args, gate_up_w_addr, weight_page_bytes);
    const auto inter_accessor = TensorAccessor(inter_out_args, inter_write_addr, output_page_bytes);
    const auto dn_w_accessor = TensorAccessor(down_w_args, down_w_addr, weight_page_bytes);
    const auto out_accessor = TensorAccessor(output_out_args, output_addr, output_page_bytes);

    // ========== Phase A: Read gate_up weights, write SwiGLU output ==========

    // Read gate_up weight tiles, one K-row at a time
    for (uint32_t k = 0; k < k_tiles; ++k) {
        cb_reserve_back(cb_weights, n_weight_per_core_gu);
        uint32_t l1_write_addr = get_write_ptr(cb_weights);

        uint32_t base_tile_id = k * n_weight_tiles_gu + core_weight_offset_gu;
        for (uint32_t n = 0; n < n_weight_per_core_gu; ++n) {
            noc_async_read_page(base_tile_id + n, gu_w_accessor, l1_write_addr);
            l1_write_addr += weight_page_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_weights, n_weight_per_core_gu);
    }

    // Write SwiGLU output tiles to DRAM intermediate
    cb_wait_front(cb_out, n_out_per_core);
    uint32_t l1_read_addr = get_read_ptr(cb_out);
    uint32_t out_tile_id = core_out_offset_gu;
    for (uint32_t n = 0; n < n_out_per_core; ++n) {
        noc_async_write_page(out_tile_id, inter_accessor, l1_read_addr);
        l1_read_addr += output_page_bytes;
        out_tile_id++;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, n_out_per_core);

    // Signal barrier: increment leader's SEM_BARRIER
    uint64_t leader_barrier_noc = get_noc_addr(leader_phys_x, leader_phys_y, get_semaphore(SEM_BARRIER));
    noc_semaphore_inc(leader_barrier_noc, 1);

    // ========== Phase B: Read down weights, write final output ==========
    // Starts immediately — no barrier wait needed (weights are static in DRAM).
    // dm0 (expert_reader) handles the barrier wait before reading intermediate.

    // Read down weight tiles, one K-row at a time
    for (uint32_t k = 0; k < k_tiles; ++k) {
        cb_reserve_back(cb_weights, n_per_core_dn);
        uint32_t l1_write_addr = get_write_ptr(cb_weights);

        uint32_t base_tile_id = k * n_tiles_dn + core_dn_offset;
        for (uint32_t n = 0; n < n_per_core_dn; ++n) {
            noc_async_read_page(base_tile_id + n, dn_w_accessor, l1_write_addr);
            l1_write_addr += weight_page_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_weights, n_per_core_dn);
    }

    // Write down output tiles to DRAM
    cb_wait_front(cb_out, n_per_core_dn);
    l1_read_addr = get_read_ptr(cb_out);
    out_tile_id = core_dn_offset;  // Output N-offset matches down weight N-offset
    for (uint32_t n = 0; n < n_per_core_dn; ++n) {
        noc_async_write_page(out_tile_id, out_accessor, l1_read_addr);
        l1_read_addr += output_page_bytes;
        out_tile_id++;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, n_per_core_dn);
}
