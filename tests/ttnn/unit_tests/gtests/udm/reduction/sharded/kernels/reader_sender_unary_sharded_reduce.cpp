// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "tt_metal/hw/inc/experimental/udm/udm_api.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

/**
 * @brief Sender (coordinator) kernel for distributed SUM reduction
 *
 * @details Coordinates reduction across multiple cores in a mesh row:
 * 1. Wait for local partial reduction results
 * 2. Coordinate waiting for all cores' partials
 * 3. Read remote partial results for global combine
 * 4. Wait for all cores' global combines to finish
 * 5. Gather all global results
 * 6. Distribute (unicast) final results to all receivers
 *
 * @note Based on LayerNorm mcast_sender style but simplified for SUM reduction only
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
    // Runtime arguments - all cores' coordinates (including self)
    // ============================================================================
    uint32_t arg_idx = 0;

    // Parse all core coordinates: index 0 = self (sender), indices 1..N = receivers
    // Store as simple arrays - the UDM API accepts array-like types
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
    constexpr uint32_t cb_out = tt::CBIndex::c_5;       // Output (gathered results)

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
    // Phase 2: Wait for local partial results
    // ============================================================================
    cb_wait_front(cb_partial, block_ht);

    // ============================================================================
    // Phase 3: Coordinate partials - wait for all cores
    // ============================================================================
    tt::tt_fabric::experimental::udm::semaphore_wait(receiver_semaphore_addr, num_blocks - 1);
    tt::tt_fabric::experimental::udm::semaphore_set(receiver_semaphore_addr, 0);

    // Signal all receivers to proceed with reading partials
    for (uint32_t i = 1; i < num_blocks; ++i) {
        tt::tt_fabric::experimental::udm::semaphore_inc(all_coords[i], 1, sender_semaphore_addr);
    }
    tt::tt_fabric::experimental::udm::atomic_barrier();

    // ============================================================================
    // Phase 4: Read remote partial results for THIS CORE's assigned rows
    // ============================================================================
    // Sender (core 0) is responsible for rows [0, num_rows_per_worker)
    // For each assigned row, read that row's partial from ALL cores (including self)
    uint32_t l1_read_addr_partial_base = get_read_ptr(cb_partial);

    for (uint32_t row = 0; row < num_rows_per_worker; ++row) {
        // Read row's partial from all cores uniformly using async_read
        cb_reserve_back(cb_external, num_blocks);
        uint32_t l1_write_addr_external = get_write_ptr(cb_external);
        uint32_t remote_partial_offset = l1_read_addr_partial_base + row * single_tile_size_bytes;

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
    // Phase 5: Wait for all cores' global combines to finish
    // ============================================================================
    cb_wait_front(cb_reduced, num_rows_per_worker);

    tt::tt_fabric::experimental::udm::semaphore_wait(receiver_semaphore_addr, num_blocks - 1);
    tt::tt_fabric::experimental::udm::semaphore_set(receiver_semaphore_addr, 0);

    // ============================================================================
    // Phase 6: Gather all global results from all cores directly to output
    // ============================================================================
    uint32_t remote_reduced_offset = get_read_ptr(cb_reduced);
    uint32_t l1_write_addr_out = get_write_ptr(cb_out);
    cb_reserve_back(cb_out, block_ht);

    for (uint32_t core = 0; core < num_blocks; ++core) {
        uint32_t num_rows = (core == num_blocks - 1) ? num_rows_per_worker_last : num_rows_per_worker;
        uint32_t num_bytes = num_rows * single_tile_size_bytes;

        // async_read(coord, local_dst_addr, size, remote_offset)
        // Read from remote core's cb_reduced, store to local cb_out
        tt::tt_fabric::experimental::udm::async_read(
            all_coords[core], l1_write_addr_out, num_bytes, remote_reduced_offset);
        l1_write_addr_out += num_bytes;
    }
    tt::tt_fabric::experimental::udm::async_read_barrier();
    cb_push_back(cb_out, block_ht);

    // ============================================================================
    // Phase 7: Distribute final results to all receivers
    // ============================================================================
    // TODO(#34705): Once UDM supports multicast, replace unicast with mcast for efficiency
    uint32_t l1_read_addr_out = get_read_ptr(cb_out);
    uint32_t total_bytes = block_ht * single_tile_size_bytes;

    // Unicast complete result to all receivers (skip index 0 = self)
    for (uint32_t i = 1; i < num_blocks; ++i) {
        // async_write(coord, local_src_addr, size, remote_offset)
        // Write from local cb_out to remote core's cb_out (same address)
        tt::tt_fabric::experimental::udm::async_write(
            all_coords[i],
            l1_read_addr_out,
            total_bytes,
            l1_read_addr_out);  // Same L1 address on receiver
    }
    tt::tt_fabric::experimental::udm::async_write_barrier();

    // Signal all receivers that data is ready (increment to 2)
    for (uint32_t i = 1; i < num_blocks; ++i) {
        tt::tt_fabric::experimental::udm::semaphore_inc(all_coords[i], 1, sender_semaphore_addr);
    }
    tt::tt_fabric::experimental::udm::atomic_barrier();
}
