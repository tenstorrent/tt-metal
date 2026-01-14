// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Kernel: Dataflow Reduce (Reader/Writer) - Interleaved Version
 *
 * Reads rows of tiles from input tensor, pushes to compute for width reduction,
 * then writes reduced tiles (1 per row) to output tensor.
 *
 * Flow per row:
 * 1. Read row of tiles from input into CB 0 (width tiles)
 * 2. Push to compute kernel for reduction
 * 3. Wait for compute to produce 1 reduced tile in CB 2
 * 4. Write reduced tile to output tensor
 *
 * Runtime args:
 *   - rank: number of dimensions
 *   - For each dimension: (num_pages, offset, stride) - for input tensor
 *   - For each dimension (rank-1): output_stride - for output tensor iteration
 */

#include <cstdint>
#include <array>
#include "api/dataflow/dataflow_api.h"
#include "experimental/udm/accessor/mesh_tensor_accessor.h"
#include "tt_metal/hw/inc/experimental/udm/udm_api.hpp"
#include "tests/ttnn/unit_tests/gtests/udm/nd_iter_args.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "api/debug/dprint.h"

void kernel_main() {
    // ==================== Create MeshTensorAccessor from Compile-Time Args ====================
    // Input tensor accessor: starts at offset 0
    auto input_args = MeshTensorAccessorArgs<0, 0>();

    // Output tensor accessor: starts after input args
    auto output_args = MeshTensorAccessorArgs<decltype(input_args)::next_compile_time_args_offset(), 0>();

    // Packed scaler value for reduction (1.0 for SUM, 1/W for MEAN)
    constexpr uint32_t scaler_packed = get_compile_time_arg_val(decltype(output_args)::next_compile_time_args_offset());

    // CB indices
    constexpr uint32_t cb_in0 = 0;     // Input tiles (row of width tiles)
    constexpr uint32_t cb_scaler = 1;  // Scaler for reduction
    constexpr uint32_t cb_out = 2;     // Output tiles (1 reduced tile per row)

    // ==================== Get Runtime Arguments ====================
    NDIterArgs<> iter_args;
    uint32_t arg_idx = iter_args.parse(0);

    // Read output strides for each row dimension (all dims except width)
    std::array<uint32_t, iter_args.max_rank> output_strides = {0};
    for (uint32_t d = 0; d < iter_args.rank - 1; ++d) {
        output_strides[d] = get_arg_val<uint32_t>(arg_idx++);
    }

    uint32_t num_rows = iter_args.total_rows();
    uint32_t input_width_tiles = iter_args.last_dim_pages();

    // Early exit for non-workers
    if (num_rows == 0 || input_width_tiles == 0) {
        return;
    }

    // ==================== Create MeshTensorAccessors ====================
    auto input_accessor = MeshTensorAccessor(input_args);
    auto output_accessor = MeshTensorAccessor(output_args);

    uint32_t page_size = input_accessor.page_size();

    // ==================== Initialize Scaler CB ====================
    // Generate scaler tile for reduction (1.0 for SUM, 1/W for MEAN)
    cb_reserve_back(cb_scaler, 1);
    generate_reduce_scaler(cb_scaler, scaler_packed);
    cb_push_back(cb_scaler, 1);

    // ==================== Compute Initial Page IDs ====================
    uint32_t input_page_id = iter_args.initial_page_id();

    // Output page_id: use output strides from mesh_tensor_shape
    uint32_t output_page_id = 0;
    for (uint32_t d = 0; d < iter_args.rank - 1; ++d) {
        output_page_id += iter_args.dim_offsets[d] * output_strides[d];
    }

    // ==================== Initialize Row Indices ====================
    std::array<uint32_t, iter_args.max_rank> row_indices = {0};

    // ==================== Process Each Row ====================
    for (uint32_t row = 0; row < num_rows; ++row) {
        // === READER: Read all tiles in this row ===
        cb_reserve_back(cb_in0, input_width_tiles);
        uint32_t l1_write_addr = get_write_ptr(cb_in0);

        for (uint32_t col = 0; col < input_width_tiles; ++col) {
            uint32_t tile_page_id = input_page_id + col * iter_args.dim_strides[iter_args.rank - 1];
            tt::tt_fabric::experimental::udm::async_read(input_accessor, tile_page_id, l1_write_addr, page_size);
            l1_write_addr += page_size;
        }
        tt::tt_fabric::experimental::udm::async_read_barrier();
        cb_push_back(cb_in0, input_width_tiles);

        // === WRITER: Wait for compute to produce reduced tile ===
        cb_wait_front(cb_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_out);

        // Write reduced tile to output tensor
        tt::tt_fabric::experimental::udm::async_write(output_accessor, output_page_id, l1_read_addr, page_size);
        tt::tt_fabric::experimental::udm::async_write_barrier();

        cb_pop_front(cb_out, 1);

        // Increment row indices and update page_ids for next row
        // We iterate over all dims except the last (width) which we reduce
        for (int d = iter_args.rank - 2; d >= 0; --d) {
            input_page_id += iter_args.dim_strides[d];
            output_page_id += output_strides[d];
            if (++row_indices[d] < iter_args.dim_pages[d]) {
                break;  // No carry needed
            }
            // Carry: reset this dimension
            row_indices[d] = 0;
            input_page_id -= iter_args.dim_pages[d] * iter_args.dim_strides[d];
            output_page_id -= iter_args.dim_pages[d] * output_strides[d];
        }
    }
}
