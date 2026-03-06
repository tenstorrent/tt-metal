// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Prefill MoE multi-expert writer (dm1/RISCV_1)
//
// For each expert e = 0..num_experts-1:
//   For each tile row m = 0..M_tiles-1:
//     Phase A: Read gate_up weights[e] -> CB1, write SwiGLU output -> inter, signal barrier A
//     Phase B: Read down weights[e] -> CB1, write output -> out_tensor[e], signal barrier B
//
// For M_tiles > 1, weights are re-read for each tile row (same weight data,
// different activation rows). Output writes use 2D page indexing:
//   inter page = m * n_inter_row_tiles + core_out_offset_gu + n
//   output page = m * n_tiles_dn + core_dn_offset + n
//
// Barrier A and B fire once per expert (after all M_tiles are processed).
//
// inter_addr is shared across experts (overwritten each iteration).
// gate_up_w_addr, down_w_addr, output_addr are per-expert.
//
// Runtime args layout:
//   [0]  inter_write_addr (shared)
//   [1]  k_tiles_gu
//   [2]  k_tiles_dn
//   [3]  n_weight_per_core_gu
//   [4]  n_weight_tiles_gu
//   [5]  core_weight_offset_gu
//   [6]  core_out_offset_gu
//   [7]  n_out_per_core
//   [8]  n_per_core_dn
//   [9]  n_tiles_dn
//   [10] core_dn_offset
//   [11] leader_phys_x
//   [12] leader_phys_y
//   [13] num_experts
//   [14] M_tiles
//   Per expert e (repeated num_experts times):
//   [15+e*3+0] gate_up_w_addr[e]
//   [15+e*3+1] down_w_addr[e]
//   [15+e*3+2] output_addr[e]
//
// Semaphores:
//   SEM_BARRIER (id=0): Incremented on leader core after Phase A and Phase B writes.

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Fixed args (same for all experts)
    const uint32_t inter_write_addr = get_arg_val<uint32_t>(0);
    const uint32_t k_tiles_gu = get_arg_val<uint32_t>(1);
    const uint32_t k_tiles_dn = get_arg_val<uint32_t>(2);
    const uint32_t n_weight_per_core_gu = get_arg_val<uint32_t>(3);
    const uint32_t n_weight_tiles_gu = get_arg_val<uint32_t>(4);
    const uint32_t core_weight_offset_gu = get_arg_val<uint32_t>(5);
    const uint32_t core_out_offset_gu = get_arg_val<uint32_t>(6);
    const uint32_t n_out_per_core = get_arg_val<uint32_t>(7);
    const uint32_t n_per_core_dn = get_arg_val<uint32_t>(8);
    const uint32_t n_tiles_dn = get_arg_val<uint32_t>(9);
    const uint32_t core_dn_offset = get_arg_val<uint32_t>(10);
    const uint32_t leader_phys_x = get_arg_val<uint32_t>(11);
    const uint32_t leader_phys_y = get_arg_val<uint32_t>(12);
    const uint32_t num_experts = get_arg_val<uint32_t>(13);
    const uint32_t M_tiles = get_arg_val<uint32_t>(14);

    constexpr auto gate_up_w_args = TensorAccessorArgs<0>();
    constexpr auto inter_out_args = TensorAccessorArgs<1>();
    constexpr auto down_w_args = TensorAccessorArgs<2>();
    constexpr auto output_out_args = TensorAccessorArgs<3>();

    constexpr uint32_t cb_weights = 1;
    constexpr uint32_t cb_out = 2;
    constexpr uint32_t SEM_BARRIER = 0;

    const uint32_t weight_page_bytes = get_local_cb_interface(cb_weights).fifo_page_size;
    const uint32_t output_page_bytes = get_local_cb_interface(cb_out).fifo_page_size;

    const auto inter_accessor = TensorAccessor(inter_out_args, inter_write_addr, output_page_bytes);

    // Row stride for inter_buf: total N-tiles per tile row = n_weight_tiles_gu / 2
    const uint32_t n_inter_row_tiles = n_weight_tiles_gu / 2;

    uint64_t leader_barrier_noc = get_noc_addr(leader_phys_x, leader_phys_y, get_semaphore(SEM_BARRIER));

    for (uint32_t expert = 0; expert < num_experts; ++expert) {
        // Per-expert addresses
        uint32_t gate_up_w_addr = get_arg_val<uint32_t>(15 + expert * 3 + 0);
        uint32_t down_w_addr = get_arg_val<uint32_t>(15 + expert * 3 + 1);
        uint32_t output_addr = get_arg_val<uint32_t>(15 + expert * 3 + 2);

        const auto gu_w_accessor = TensorAccessor(gate_up_w_args, gate_up_w_addr, weight_page_bytes);
        const auto dn_w_accessor = TensorAccessor(down_w_args, down_w_addr, weight_page_bytes);
        const auto out_accessor = TensorAccessor(output_out_args, output_addr, output_page_bytes);

        // ========== Phase A: Read gate_up weights, write SwiGLU output (M_tiles rows) ==========
        for (uint32_t m = 0; m < M_tiles; ++m) {
            // Read gate_up weights (same weight data for each M_tile)
            for (uint32_t k = 0; k < k_tiles_gu; ++k) {
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

            // Write SwiGLU output tiles to DRAM intermediate (2D page indexing)
            cb_wait_front(cb_out, n_out_per_core);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            uint32_t out_tile_id = m * n_inter_row_tiles + core_out_offset_gu;
            for (uint32_t n = 0; n < n_out_per_core; ++n) {
                noc_async_write_page(out_tile_id, inter_accessor, l1_read_addr);
                l1_read_addr += output_page_bytes;
                out_tile_id++;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, n_out_per_core);
        }

        // Signal barrier A (once after all M_tiles)
        noc_semaphore_inc(leader_barrier_noc, 1);

        // ========== Phase B: Read down weights, write final output (M_tiles rows) ==========
        for (uint32_t m = 0; m < M_tiles; ++m) {
            // Read down weights (same weight data for each M_tile)
            for (uint32_t k = 0; k < k_tiles_dn; ++k) {
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

            // Write down output tiles to DRAM (2D page indexing)
            cb_wait_front(cb_out, n_per_core_dn);
            uint32_t l1_read_addr = get_read_ptr(cb_out);
            uint32_t out_tile_id = m * n_tiles_dn + core_dn_offset;
            for (uint32_t n = 0; n < n_per_core_dn; ++n) {
                noc_async_write_page(out_tile_id, out_accessor, l1_read_addr);
                l1_read_addr += output_page_bytes;
                out_tile_id++;
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, n_per_core_dn);
        }

        // Signal barrier B (once after all M_tiles)
        noc_semaphore_inc(leader_barrier_noc, 1);
    }
}
