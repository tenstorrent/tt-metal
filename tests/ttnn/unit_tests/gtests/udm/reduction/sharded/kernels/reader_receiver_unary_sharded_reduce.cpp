// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/hw/inc/experimental/udm/udm_api.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

/**
 * @brief Receiver (non-coordinator) kernel for distributed SUM reduction
 *
 * @details Synchronizes with sender coordinator for reduction:
 * 1. Wait for local partial reduction results
 * 2. Notify sender that partials are ready
 * 3. Wait for sender's signal to read remote partials
 * 4. Read remote partial results for global combine
 * 5. Notify sender when global combine is done
 * 6. Receive (unicast) final global results from sender
 *
 * @note Based on LayerNorm mcast_receiver style but simplified for SUM reduction only
 */
void kernel_main() {
    // ============================================================================
    // Compile-time arguments
    // ============================================================================
    // Note: With GlobalSemaphore, these are L1 addresses (same on all devices), not semaphore IDs
    constexpr uint32_t receiver_semaphore_addr = get_compile_time_arg_val(0);
    constexpr uint32_t sender_semaphore_addr = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);  // num_cores_x
    constexpr uint32_t block_ht = get_compile_time_arg_val(3);    // Height per core
    constexpr uint32_t num_rows_per_worker = get_compile_time_arg_val(4);
    constexpr uint32_t num_rows_per_worker_last = get_compile_time_arg_val(5);
    constexpr uint32_t winv_packed = get_compile_time_arg_val(6);  // 1/W scaler
    constexpr uint32_t coord_dims = get_compile_time_arg_val(7);   // Coordinate dimensions

    static_assert(num_blocks > 1, "Need at least 2 cores for reduction");

    // ============================================================================
    // Runtime arguments - core index and all cores' coordinates
    // ============================================================================
    uint32_t arg_idx = 0;
    const uint32_t core_idx = get_arg_val<uint32_t>(arg_idx++);  // This core's index in the row (1, 2, 3, ...)

    // Store as simple arrays - the UDM API accepts array-like types
    // Index 0 = sender, indices 1..N = receivers
    // All coordinates have the same dimensionality (coord_dims)
    uint32_t all_coords[num_blocks][coord_dims];

    for (uint32_t i = 0; i < num_blocks; ++i) {
        for (uint32_t d = 0; d < coord_dims; ++d) {
            all_coords[i][d] = get_arg_val<uint32_t>(arg_idx++);
        }
    }

    // ============================================================================
    // CB definitions
    // ============================================================================
    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;       // Input (globally allocated)
    constexpr uint32_t cb_scaler = tt::CBIndex::c_1;    // Scaler for reduction
    constexpr uint32_t cb_partial = tt::CBIndex::c_2;   // Local partial results
    constexpr uint32_t cb_reduced = tt::CBIndex::c_3;   // Global reduced results
    constexpr uint32_t cb_external = tt::CBIndex::c_4;  // External (remote) data
    constexpr uint32_t cb_out = tt::CBIndex::c_5;       // Output (received results)

    // ============================================================================
    // Setup
    // ============================================================================
    const uint32_t single_tile_size_bytes = get_tile_size(cb_partial);

    // ============================================================================
    // Phase 1: Generate scaler tile for reduction
    // ============================================================================
    cb_reserve_back(cb_scaler, 1);
    generate_reduce_scaler(cb_scaler, winv_packed);
    cb_push_back(cb_scaler, 1);

    // ============================================================================
    // Determine number of rows this core is responsible for
    // ============================================================================
    const uint32_t my_num_rows = (core_idx == num_blocks - 1) ? num_rows_per_worker_last : num_rows_per_worker;

    // ============================================================================
    // Phase 2: Wait for local partial results
    // ============================================================================
    cb_wait_front(cb_partial, block_ht);

    // ============================================================================
    // Phase 3: Notify sender that partials are ready and wait for signal
    // ============================================================================
    tt::tt_fabric::experimental::udm::semaphore_set(sender_semaphore_addr, 0);
    tt::tt_fabric::experimental::udm::semaphore_inc(all_coords[0], 1, receiver_semaphore_addr);
    tt::tt_fabric::experimental::udm::atomic_barrier();
    tt::tt_fabric::experimental::udm::semaphore_wait(sender_semaphore_addr, 1);

    // ============================================================================
    // Phase 4: Read remote partial results for THIS CORE's assigned rows
    // ============================================================================
    // This receiver (core_idx) is responsible for my_num_rows rows
    // Row offset: core_idx * num_rows_per_worker (last core may have fewer rows)
    uint32_t l1_read_addr_partial_base = get_read_ptr(cb_partial);
    uint32_t row_offset = core_idx * num_rows_per_worker;

    for (uint32_t row = 0; row < my_num_rows; ++row) {
        // Read this row's partial from all cores
        cb_reserve_back(cb_external, num_blocks);
        uint32_t l1_write_addr_external = get_write_ptr(cb_external);
        uint32_t global_row_idx = row_offset + row;
        uint32_t remote_partial_offset = l1_read_addr_partial_base + global_row_idx * single_tile_size_bytes;

        for (uint32_t core = 0; core < num_blocks; ++core) {
            // async_read(coord, local_dst_addr, size, remote_offset)
            // Read from remote core at remote_partial_offset, store to local l1_write_addr_external
            tt::tt_fabric::experimental::udm::async_read(
                all_coords[core], l1_write_addr_external, single_tile_size_bytes, remote_partial_offset);
            l1_write_addr_external += single_tile_size_bytes;
        }

        tt::tt_fabric::experimental::udm::async_read_barrier();
        cb_push_back(cb_external, num_blocks);
    }

    // ============================================================================
    // Phase 5: Wait for global combine and notify sender
    // ============================================================================
    cb_wait_front(cb_reduced, my_num_rows);
    // Notify sender that our global combine is done
    tt::tt_fabric::experimental::udm::semaphore_inc(all_coords[0], 1, receiver_semaphore_addr);
    tt::tt_fabric::experimental::udm::atomic_barrier();

    // ============================================================================
    // Phase 6: Receive final global results from sender directly to output
    // ============================================================================
    // Reserve space for all block_ht tiles
    cb_reserve_back(cb_out, block_ht);

    // Wait for sender to finish writing all data and signal
    // Semaphore: 0 → 1 (Phase 3) → 2 (Phase 6)
    tt::tt_fabric::experimental::udm::semaphore_wait(sender_semaphore_addr, 2);

    // All data received, push to output
    cb_push_back(cb_out, block_ht);
}
