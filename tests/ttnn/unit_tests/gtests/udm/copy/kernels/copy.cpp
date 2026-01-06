// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Kernel: Copy
 *
 * Demonstrates mesh tensor access using MeshTensorAccessor:
 * 1. Constructs MeshTensorAccessor from compile-time args
 * 2. Receives page offsets via runtime args
 * 3. Iterates over pages and copies from input to output via NOC
 */

#include <cstdint>
#include <array>
#include "api/dataflow/dataflow_api.h"
#include "udm/accessor/mesh_tensor_accessor.h"
#include "tt_metal/hw/inc/udm/udm_api.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"

void kernel_main() {
    // TODO(#34735): move fabric counter init to fw kernel init
    tt::tt_fabric::udm::fabric_local_state_init();

    // ==================== Create MeshTensorAccessor from Compile-Time Args ====================
    // MeshTensorAccessorArgs extracts all mesh/grid/dspec config from compile-time args
    // Input tensor accessor: starts at offset 0
    auto input_args = MeshTensorAccessorArgs<0, 0>();

    // Output tensor accessor: starts after input args
    auto output_args = MeshTensorAccessorArgs<decltype(input_args)::next_compile_time_args_offset(), 0>();

    constexpr uint32_t cb_id = 0;  // using CB0

    // ==================== Get Runtime Arguments ====================
    // General ND iteration format:
    // - rank: number of dimensions
    // - For each dimension: (num_pages, offset, stride)
    uint32_t rank = get_arg_val<uint32_t>(0);

    constexpr uint32_t MAX_RANK = 8;
    std::array<uint32_t, MAX_RANK> dim_pages = {0};
    std::array<uint32_t, MAX_RANK> dim_offsets = {0};
    std::array<uint32_t, MAX_RANK> dim_strides = {0};

    uint32_t arg_idx = 1;
    for (uint32_t d = 0; d < rank; ++d) {
        dim_pages[d] = get_arg_val<uint32_t>(arg_idx++);
        dim_offsets[d] = get_arg_val<uint32_t>(arg_idx++);
        dim_strides[d] = get_arg_val<uint32_t>(arg_idx++);
    }

    // ==================== Create MeshTensorAccessor ====================
    // Template parameter is deduced from args via deduction guide
    // Buffer address is obtained from compile-time args
    auto input_mesh_accessor = MeshTensorAccessor(input_args);
    auto output_mesh_accessor = MeshTensorAccessor(output_args);

    // ==================== Read Pages ====================
    // Get page size from the mesh accessor (comes from CTA args)
    uint32_t page_size = input_mesh_accessor.page_size();

    // Calculate total number of iterations (product of all dim_pages)
    uint32_t total_pages = 1;
    for (uint32_t d = 0; d < rank; ++d) {
        total_pages *= dim_pages[d];
    }
    if (total_pages == 0) {
        return;
    }

    // Initialize indices and compute initial page_id
    std::array<uint32_t, MAX_RANK> indices = {0};
    uint32_t page_id = 0;
    for (uint32_t d = 0; d < rank; ++d) {
        page_id += dim_offsets[d] * dim_strides[d];
    }

    // Iterate through all pages with incremental page_id computation
    cb_reserve_back(cb_id, 1);
    uint32_t l1_addr = get_write_ptr(cb_id);
    for (uint32_t iter = 0; iter < total_pages; ++iter) {
        // Read from input tensor using UDM API (handles both local and remote)
        tt::tt_fabric::experimental::udm::async_read(input_mesh_accessor, page_id, l1_addr, page_size);
        tt::tt_fabric::experimental::udm::async_read_barrier();

        // TODO(#34704): once allowed two kernels in one tensix to use fabric connection, move to writer kernel
        // Write to output tensor using UDM API
        tt::tt_fabric::experimental::udm::async_write(output_mesh_accessor, page_id, l1_addr, page_size);
        tt::tt_fabric::experimental::udm::async_write_barrier();

        // Increment indices and update page_id for next iteration
        for (int d = rank - 1; d >= 0; --d) {
            page_id += dim_strides[d];  // Move to next in this dimension
            if (++indices[d] < dim_pages[d]) {
                break;  // No carry needed
            }
            // Carry: reset this dimension and continue to next
            indices[d] = 0;
            page_id -= dim_pages[d] * dim_strides[d];
        }
    }

    // TODO(#34736): remove once we have persistent connection across programs
    tt::tt_fabric::udm::close_fabric_connection();
}
