// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>
#include "dataflow_api.h"
#include "udm/accessor/mesh_tensor_accessor.h"
#include "udm/accessor/mesh_gcore_accessor.h"
#include "tt_metal/hw/inc/udm/udm_api.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

/**
 * @brief Dataflow kernel (reader + writer) for width reduction
 *
 * Algorithm:
 * 1. Each gcore: Read input tiles and compute local reduction across width → CB2
 * 2. All gcores: Send all reduced tiles to first gcore's CB3
 * 3. First gcore: Receive and write final accumulated results to output
 */

// Compile-time arguments
constexpr uint32_t packed_scaler_value = get_compile_time_arg_val(0);
constexpr uint32_t cb_id_in = get_compile_time_arg_val(1);
constexpr uint32_t cb_id_scaler = get_compile_time_arg_val(2);
constexpr uint32_t cb_id_reduced = get_compile_time_arg_val(3);
constexpr uint32_t cb_id_received = get_compile_time_arg_val(4);
constexpr uint32_t cb_id_add_result = get_compile_time_arg_val(5);
constexpr uint32_t semaphore_id = get_compile_time_arg_val(6);

constexpr uint32_t MAX_RANK = 8;

/**
 * @brief Step 1: Read input tiles for local reduction
 */
void read_input_tensor(
    const MeshTensorAccessor& accessor,
    uint32_t num_rows,
    uint32_t pages_per_row,
    uint32_t rank,
    uint32_t page_id,
    const std::array<uint32_t, MAX_RANK>& dim_pages,
    const std::array<uint32_t, MAX_RANK>& dim_strides,
    uint32_t page_size) {
    std::array<uint32_t, MAX_RANK> row_indices = {0};
    uint32_t current_page_offset = page_id;

    for (uint32_t row = 0; row < num_rows; ++row) {
        uint32_t current_page_id = current_page_offset;
        for (uint32_t w = 0; w < pages_per_row; ++w) {
            cb_reserve_back(cb_id_in, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in);

            tt::tt_fabric::experimental::udm::async_read(accessor, current_page_id, l1_write_addr, page_size);
            tt::tt_fabric::experimental::udm::async_read_barrier();

            cb_push_back(cb_id_in, 1);

            current_page_id += dim_strides[rank - 1];
        }

        // Increment current_page_offset for next row
        for (int d = rank - 2; d >= 0; --d) {
            current_page_offset += dim_strides[d];
            if (++row_indices[d] < dim_pages[d]) {
                break;
            }
            row_indices[d] = 0;
            current_page_offset -= dim_pages[d] * dim_strides[d];
        }
    }
}

/**
 * @brief Step 2: Send reduced tiles to first gcore
 */
void send_to_first_gcore(
    uint32_t num_rows,
    uint32_t gcore_idx,
    const std::array<uint32_t, MESH_NUM_DIMS>& coord,
    uint32_t semaphore_addr,
    uint32_t page_size) {
    constexpr uint32_t cb_received_base = get_write_ptr(cb_id_received);
    uint32_t gcore_offset_in_cb = gcore_idx * num_rows * page_size;
    uint32_t receive_buffer_addr = cb_received_base + gcore_offset_in_cb;

    // All gcores use the same code path - write to first gcore via NOC
    for (uint32_t row = 0; row < num_rows; ++row) {
        cb_wait_front(cb_id_reduced, 1);
        uint32_t src_addr = get_read_ptr(cb_id_reduced);

        // Write to first gcore's receive buffer at this row's offset
        uint32_t dest_addr = receive_buffer_addr + row * page_size;
        tt::tt_fabric::experimental::udm::async_write(coord, src_addr, page_size, dest_addr);
        tt::tt_fabric::experimental::udm::async_write_barrier();

        cb_pop_front(cb_id_reduced, 1);
    }

    // Increment semaphore on first gcore ONCE to signal all tiles sent
    tt::tt_fabric::experimental::udm::semaphore_inc(coord, 1, semaphore_addr);
}

/**
 * @brief Step 3: First gcore writes accumulated results
 */
void write_accumulated_results(
    const MeshTensorAccessor& accessor,
    uint32_t num_rows,
    uint32_t total_gcores,
    uint32_t rank,
    uint32_t page_id,
    const std::array<uint32_t, MAX_RANK>& dim_pages,
    const std::array<uint32_t, MAX_RANK>& strides,
    uint32_t semaphore_addr,
    uint32_t page_size) {
    std::array<uint32_t, MAX_RANK> row_indices = {0};
    uint32_t current_page_id = page_id;

    // Reserve space for all tiles in CB3 (from all gcores)
    const uint32_t total_all_tiles = total_gcores * num_rows;
    cb_reserve_back(cb_id_received, total_all_tiles);

    tt::tt_fabric::experimental::udm::semaphore_wait(semaphore_addr, total_gcores);
    tt::tt_fabric::experimental::udm::semaphore_set(semaphore_addr, 0);

    cb_push_back(cb_id_received, total_all_tiles);

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Compute kernel reduces all tiles from CB3 for this row
        cb_wait_front(cb_id_add_result, 1);
        uint32_t addr = get_read_ptr(cb_id_add_result);

        // Write final result to output
        tt::tt_fabric::experimental::udm::async_write(accessor, current_page_id, addr, page_size);
        tt::tt_fabric::experimental::udm::async_write_barrier();

        // Pop consumed CB
        cb_pop_front(cb_id_add_result, 1);

        // Increment current_page_id for next row
        for (int d = rank - 2; d >= 0; --d) {
            current_page_id += strides[d];
            if (++row_indices[d] < dim_pages[d]) {
                break;
            }
            row_indices[d] = 0;
            current_page_id -= dim_pages[d] * strides[d];
        }
    }

    // Pop all tiles from CB3 after processing all rows
    cb_pop_front(cb_id_received, total_all_tiles);
}

void kernel_main() {
    // TODO: move fabric counter init to fw kernel init
    tt::tt_fabric::udm::fabric_local_state_init();

    // ==================== Create Accessors ====================
    auto input_args = MeshTensorAccessorArgs<7, 0>();  // Offset 7 for scaler + CB IDs + semaphore
    auto input_accessor = MeshTensorAccessor(input_args);

    auto output_args = MeshTensorAccessorArgs<decltype(input_args)::next_compile_time_args_offset(), 0>();
    auto output_accessor = MeshTensorAccessor(output_args);

    // ==================== Get Runtime Arguments ====================
    uint32_t rank = get_arg_val<uint32_t>(0);

    std::array<uint32_t, MAX_RANK> dim_pages = {0};
    std::array<uint32_t, MAX_RANK> dim_offsets = {0};
    std::array<uint32_t, MAX_RANK> dim_strides = {0};

    uint32_t arg_idx = 1;
    for (uint32_t d = 0; d < rank; ++d) {
        dim_pages[d] = get_arg_val<uint32_t>(arg_idx++);
        dim_offsets[d] = get_arg_val<uint32_t>(arg_idx++);
        dim_strides[d] = get_arg_val<uint32_t>(arg_idx++);
    }

    std::array<uint32_t, MAX_RANK> output_strides = {0};
    for (uint32_t d = 0; d < rank; ++d) {
        output_strides[d] = get_arg_val<uint32_t>(arg_idx++);
    }

    uint32_t gcore_idx = get_arg_val<uint32_t>(arg_idx++);
    uint32_t total_gcores = get_arg_val<uint32_t>(arg_idx++);

    uint32_t first_gcore_rank = get_arg_val<uint32_t>(arg_idx++);
    std::array<uint32_t, MESH_NUM_DIMS> first_coord = {0};
    for (uint32_t d = 0; d < first_gcore_rank; ++d) {
        first_coord[d] = get_arg_val<uint32_t>(arg_idx++);
    }

    uint32_t semaphore_addr = get_semaphore(semaphore_id);
    uint32_t page_size = input_accessor.page_size();

    // Generate scaler tile for reduction
    generate_reduce_scaler(cb_id_scaler, packed_scaler_value);

    // Calculate number of output rows (product of all dims except last)
    uint32_t num_rows = 1;
    for (uint32_t d = 0; d < rank - 1; ++d) {
        num_rows *= dim_pages[d];
    }
    if (num_rows == 0) {
        return;
    }

    uint32_t pages_per_row = dim_pages[rank - 1];

    // Compute initial page IDs
    uint32_t input_page_id = 0;
    uint32_t output_page_id = 0;
    for (uint32_t d = 0; d < rank; ++d) {
        input_page_id += dim_offsets[d] * dim_strides[d];
    }
    for (uint32_t d = 0; d < rank - 1; ++d) {
        output_page_id += dim_offsets[d] * output_strides[d];
    }

    // Step 1: Read input tiles for local reduction
    read_input_tensor(input_accessor, num_rows, pages_per_row, rank, input_page_id, dim_pages, dim_strides, page_size);

    // Step 2: Send reduced tiles to first gcore
    send_to_first_gcore(num_rows, gcore_idx, first_coord, semaphore_addr, page_size);

    // Step 3: First gcore writes accumulated results
    if (gcore_idx == 0) {
        write_accumulated_results(
            output_accessor,
            num_rows,
            total_gcores,
            rank,
            output_page_id,
            dim_pages,
            output_strides,
            semaphore_addr,
            page_size);
    }

    // TODO: remove once we have persistent connection across programs
    tt::tt_fabric::udm::close_fabric_connection();
}
