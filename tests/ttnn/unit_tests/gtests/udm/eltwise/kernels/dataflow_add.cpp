// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Kernel: Dataflow Add (Reader/Writer)
 *
 * Reads tiles from two input tensors into circular buffers for compute kernel,
 * then writes the result from output CB to output tensor.
 *
 * Flow:
 * 1. Read tile from input A into CB 0
 * 2. Read tile from input B into CB 1
 * 3. Push tiles to compute kernel
 * 4. Wait for compute to produce result in CB 2
 * 5. Write result tile to output tensor
 */

#include <cstdint>
#include <array>
#include "api/dataflow/dataflow_api.h"
#include "experimental/udm/accessor/mesh_tensor_accessor.h"
#include "tt_metal/hw/inc/experimental/udm/udm_api.hpp"
#include "tests/ttnn/unit_tests/gtests/udm/nd_iter_args.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

void kernel_main() {
    // ==================== Create MeshTensorAccessor from Compile-Time Args ====================
    // Input A tensor accessor: starts at offset 0
    auto input_a_args = MeshTensorAccessorArgs<0, 0>();

    // Input B tensor accessor: starts after input A args
    auto input_b_args = MeshTensorAccessorArgs<decltype(input_a_args)::next_compile_time_args_offset(), 0>();

    // Output tensor accessor: starts after input B args
    auto output_args = MeshTensorAccessorArgs<decltype(input_b_args)::next_compile_time_args_offset(), 0>();

    // CB indices for dataflow/compute communication
    constexpr uint32_t cb_in0 = 0;  // Input A
    constexpr uint32_t cb_in1 = 1;  // Input B
    constexpr uint32_t cb_out = 2;  // Output (from compute)

    // ==================== Get Runtime Arguments ====================
    NDIterArgs<> iter_args;
    iter_args.parse(0);

    // ==================== Create MeshTensorAccessors ====================
    auto input_a_accessor = MeshTensorAccessor(input_a_args);
    auto input_b_accessor = MeshTensorAccessor(input_b_args);
    auto output_accessor = MeshTensorAccessor(output_args);

    // ==================== Process Tiles ====================
    uint32_t page_size = input_a_accessor.page_size();

    uint32_t total_pages = iter_args.total_pages();
    if (total_pages == 0) {
        return;
    }

    // Initialize indices and compute initial page_id
    std::array<uint32_t, iter_args.max_rank> indices = {0};
    uint32_t page_id = iter_args.initial_page_id();

    // Iterate through all pages
    for (uint32_t iter = 0; iter < total_pages; ++iter) {
        // === READER PHASE ===
        // Read input A tile into CB 0
        cb_reserve_back(cb_in0, 1);
        uint32_t l1_addr_a = get_write_ptr(cb_in0);
        tt::tt_fabric::experimental::udm::async_read(input_a_accessor, page_id, l1_addr_a, page_size);
        tt::tt_fabric::experimental::udm::async_read_barrier();
        cb_push_back(cb_in0, 1);

        // Read input B tile into CB 1
        cb_reserve_back(cb_in1, 1);
        uint32_t l1_addr_b = get_write_ptr(cb_in1);
        tt::tt_fabric::experimental::udm::async_read(input_b_accessor, page_id, l1_addr_b, page_size);
        tt::tt_fabric::experimental::udm::async_read_barrier();
        cb_push_back(cb_in1, 1);

        // === WRITER PHASE ===
        // Wait for compute kernel to produce result in CB output
        cb_wait_front(cb_out, 1);
        uint32_t l1_addr_out = get_read_ptr(cb_out);

        // Write result to output tensor using UDM API
        tt::tt_fabric::experimental::udm::async_write(output_accessor, page_id, l1_addr_out, page_size);
        tt::tt_fabric::experimental::udm::async_write_barrier();

        cb_pop_front(cb_out, 1);

        // Increment indices and update page_id for next iteration
        for (int d = iter_args.rank - 1; d >= 0; --d) {
            page_id += iter_args.dim_strides[d];  // Move to next in this dimension
            if (++indices[d] < iter_args.dim_pages[d]) {
                break;  // No carry needed
            }
            // Carry: reset this dimension and continue to next
            indices[d] = 0;
            page_id -= iter_args.dim_pages[d] * iter_args.dim_strides[d];
        }
    }
}
