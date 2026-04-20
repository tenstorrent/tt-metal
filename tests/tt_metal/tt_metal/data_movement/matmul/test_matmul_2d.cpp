// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "test_matmul_common.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <distributed/mesh_device_impl.hpp>
#include "tt_metal/impl/allocator/allocator.hpp"
#include "tt_metal/impl/dispatch/dispatch_query_manager.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::two_d_matmul {

using MatmulTestConfig = unit_tests::dm::matmul::MatmulTestConfig;

uint32_t runtime_host_id = 0;

/// @brief 2D matmul data movement with rotating sender for both in0 and in1.
///        in0 K subblocks are distributed round-robin across columns; each column multicasts across its row.
///        in1 K subblocks are distributed round-robin across rows; each row multicasts across its column.
///        No DRAM is used — all data is written to L1 by the host.
bool run_dm_2d_matmul(const shared_ptr<distributed::MeshDevice>& mesh_device, const MatmulTestConfig& test_config) {
    IDevice* device = mesh_device->impl().get_device(0);

    // Check that the requested grid fits within the device's compute grid.
    auto compute_grid = device->compute_with_storage_grid_size();
    if (test_config.end_logical_core.x >= compute_grid.x || test_config.end_logical_core.y >= compute_grid.y) {
        log_info(
            tt::LogTest,
            "Skipping test {}: requested grid end ({},{}) exceeds compute grid ({}x{})",
            test_config.test_id,
            test_config.end_logical_core.x,
            test_config.end_logical_core.y,
            compute_grid.x,
            compute_grid.y);
        return true;
    }

    Program program = CreateProgram();

    uint32_t grid_cols = test_config.end_logical_core.x - test_config.start_logical_core.x + 1;
    uint32_t grid_rows = test_config.end_logical_core.y - test_config.start_logical_core.y + 1;
    if (grid_cols != test_config.num_subblocks_c_dim) {
        log_error(
            tt::LogTest,
            "Grid columns ({}) must match num_subblocks_c_dim ({})",
            grid_cols,
            test_config.num_subblocks_c_dim);
        return false;
    }
    if (grid_rows != test_config.num_subblocks_r_dim) {
        log_error(
            tt::LogTest,
            "Grid rows ({}) must match num_subblocks_r_dim ({})",
            grid_rows,
            test_config.num_subblocks_r_dim);
        return false;
    }

    CoreRangeSet matmul_cores({CoreRange(test_config.start_logical_core, test_config.end_logical_core)});
    vector<CoreCoord> matmul_cores_list = corerange_to_cores(matmul_cores);

    CoreCoord matmul_physical_start_coord = device->worker_core_from_logical_core(test_config.start_logical_core);
    CoreCoord matmul_physical_end_coord = device->worker_core_from_logical_core(test_config.end_logical_core);

    // Build per-column physical X coordinate array (for in0 row-wise multicast)
    vector<uint32_t> col_phys_x(grid_cols);
    for (uint32_t c = 0; c < grid_cols; c++) {
        CoreCoord logical_col_core(test_config.start_logical_core.x + c, test_config.start_logical_core.y);
        CoreCoord phys = device->worker_core_from_logical_core(logical_col_core);
        col_phys_x[c] = phys.x;
    }

    // Build per-row physical Y coordinate array (for in1 column-wise multicast)
    vector<uint32_t> row_phys_y(grid_rows);
    for (uint32_t r = 0; r < grid_rows; r++) {
        CoreCoord logical_row_core(test_config.start_logical_core.x, test_config.start_logical_core.y + r);
        CoreCoord phys = device->worker_core_from_logical_core(logical_row_core);
        row_phys_y[r] = phys.y;
    }

    // ---- Size calculations ----
    uint32_t K = test_config.num_subblocks_k_dim;
    uint32_t C = test_config.num_subblocks_c_dim;
    uint32_t R = test_config.num_subblocks_r_dim;

    // in0 K subblock: subblock_r_dim * subblock_k_dim pages (one K subblock for one row)
    uint32_t in0_k_subblock_size_bytes =
        test_config.subblock_r_dim * test_config.subblock_k_dim * test_config.page_size_bytes;

    // in1 K subblock: subblock_k_dim * subblock_c_dim pages (one K subblock for one column)
    uint32_t in1_k_subblock_size_bytes =
        test_config.subblock_k_dim * test_config.subblock_c_dim * test_config.page_size_bytes;

    // Maximum K subblocks any column holds for in0
    uint32_t max_in0_k_per_col = (K + C - 1) / C;
    uint32_t in0_max_source_data_bytes = max_in0_k_per_col * in0_k_subblock_size_bytes;

    // Maximum K subblocks any row holds for in1
    uint32_t max_in1_k_per_row = (K + R - 1) / R;
    uint32_t in1_max_source_data_bytes = max_in1_k_per_row * in1_k_subblock_size_bytes;

    uint32_t l1_base_address = unit_tests::dm::get_l1_address_and_size(mesh_device, matmul_cores_list[0]).base_address;

    // ---- L1 Memory Layout ----
    // l1_base:                                    in0 source data (this col's K subblocks)
    // in0_source_end + pad:                       in0 mcast output (K * in0_k_sub_size)
    // in0_mcast_end + pad:                        in1 source data (this row's K subblocks)
    // in1_source_end + pad:                       in1 mcast output (K * in1_k_sub_size)
    // align16(in1_mcast_end):                     RISCV_0 barrier scratch (16 bytes)
    // align16(risc0_scratch + 16):                RISCV_1 barrier scratch (16 bytes)

    uint32_t in0_mcast_output_addr = l1_base_address + in0_max_source_data_bytes + matmul::L1_DEBUG_PADDING_BYTES;
    uint32_t in0_output_total_bytes = K * in0_k_subblock_size_bytes;

    uint32_t in1_source_addr = in0_mcast_output_addr + in0_output_total_bytes + matmul::L1_DEBUG_PADDING_BYTES;
    uint32_t in1_mcast_output_addr = in1_source_addr + in1_max_source_data_bytes + matmul::L1_DEBUG_PADDING_BYTES;
    uint32_t in1_output_total_bytes = K * in1_k_subblock_size_bytes;

    uint32_t risc0_local_barrier_addr = (in1_mcast_output_addr + in1_output_total_bytes + 15) & ~15U;
    uint32_t risc1_local_barrier_addr = risc0_local_barrier_addr + 16;

    // ---- Generate in0 data ----
    uint32_t in0_pages = (R * test_config.subblock_r_dim) * (K * test_config.subblock_k_dim);
    uint32_t in0_pages_bytes = in0_pages * test_config.page_size_bytes;
    vector<uint32_t> in0_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, in0_pages_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    uint32_t in0_row_data_size_uint32 = in0_input.size() / R;
    uint32_t in0_k_subblock_size_uint32 = in0_k_subblock_size_bytes / sizeof(uint32_t);

    // ---- Generate in1 data ----
    // in1 is laid out as C column groups, each with K contiguous subblocks
    uint32_t in1_pages = (K * test_config.subblock_k_dim) * (C * test_config.subblock_c_dim);
    uint32_t in1_pages_bytes = in1_pages * test_config.page_size_bytes;
    vector<uint32_t> in1_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, in1_pages_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    uint32_t in1_col_data_size_uint32 = in1_input.size() / C;
    uint32_t in1_k_subblock_size_uint32 = in1_k_subblock_size_bytes / sizeof(uint32_t);

    // ---- Clear all matmul cores L1 ----
    uint32_t total_clear_size = risc1_local_barrier_addr + 16 - l1_base_address;
    vector<uint32_t> zeros(total_clear_size / sizeof(uint32_t), 0);
    for (const auto& core : matmul_cores_list) {
        detail::WriteToDeviceL1(device, core, l1_base_address, zeros);
    }

    // ---- Distribute in0 data round-robin across columns ----
    // Golden data map: row_y -> full row data (for in0 verification)
    unordered_map<uint32_t, vector<uint32_t>> dim_r_to_in0_pages_map;

    for (uint32_t r = 0; r < R; r++) {
        vector<uint32_t> row_data(
            in0_input.begin() + r * in0_row_data_size_uint32, in0_input.begin() + (r + 1) * in0_row_data_size_uint32);
        dim_r_to_in0_pages_map[test_config.start_logical_core.y + r] = row_data;
    }

    // ---- Distribute in1 data round-robin across rows ----
    // Golden data map: col_x -> full column data (for in1 verification)
    unordered_map<uint32_t, vector<uint32_t>> dim_c_to_in1_pages_map;

    for (uint32_t c = 0; c < C; c++) {
        vector<uint32_t> col_data(
            in1_input.begin() + c * in1_col_data_size_uint32, in1_input.begin() + (c + 1) * in1_col_data_size_uint32);
        dim_c_to_in1_pages_map[test_config.start_logical_core.x + c] = col_data;
    }

    // ---- Write per-core in0 and in1 source data ----
    for (const auto& core : matmul_cores_list) {
        uint32_t col_idx = core.x - test_config.start_logical_core.x;
        uint32_t row_idx = core.y - test_config.start_logical_core.y;

        // in0: column c gets K subblocks {c, c+C, c+2C, ...} from row r
        vector<uint32_t> core_in0_data;
        uint32_t in0_row_base = row_idx * in0_row_data_size_uint32;
        for (uint32_t k = col_idx; k < K; k += C) {
            uint32_t k_offset = k * in0_k_subblock_size_uint32;
            for (uint32_t e = 0; e < in0_k_subblock_size_uint32; e++) {
                core_in0_data.push_back(in0_input[in0_row_base + k_offset + e]);
            }
        }
        if (!core_in0_data.empty()) {
            detail::WriteToDeviceL1(device, core, l1_base_address, core_in0_data);
        }

        // in1: row r gets K subblocks {r, r+R, r+2R, ...} from column c
        vector<uint32_t> core_in1_data;
        uint32_t in1_col_base = col_idx * in1_col_data_size_uint32;
        for (uint32_t k = row_idx; k < K; k += R) {
            uint32_t k_offset = k * in1_k_subblock_size_uint32;
            for (uint32_t e = 0; e < in1_k_subblock_size_uint32; e++) {
                core_in1_data.push_back(in1_input[in1_col_base + k_offset + e]);
            }
        }
        if (!core_in1_data.empty()) {
            detail::WriteToDeviceL1(device, core, in1_source_addr, core_in1_data);
        }
    }

    // ---- Semaphores ----
    // in0 multicast semaphores (row-wise, same as v2)
    uint32_t in0_sender_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t in0_sender_valid_sem_id = CreateSemaphore(program, matmul_cores, 1);
    uint32_t in0_receiver_sem_id = CreateSemaphore(program, matmul_cores, 0);

    // in1 multicast semaphores (column-wise)
    uint32_t in1_sender_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t in1_sender_valid_sem_id = CreateSemaphore(program, matmul_cores, 1);
    uint32_t in1_receiver_sem_id = CreateSemaphore(program, matmul_cores, 0);

    // Barrier synchronization semaphores
    uint32_t risc0_barrier_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t risc0_barrier_done_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t risc1_barrier_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t risc1_barrier_done_sem_id = CreateSemaphore(program, matmul_cores, 0);

    CoreCoord barrier_coordinator_phys = device->worker_core_from_logical_core(matmul_cores_list[0]);
    uint32_t num_cores = matmul_cores_list.size();

    // ---- Compile-time args ----
    vector<uint32_t> risc0_compile_args = {
        test_config.test_id,                      // 0  Test ID
        (uint32_t)matmul_physical_start_coord.x,  // 1  Physical start x
        (uint32_t)matmul_physical_start_coord.y,  // 2  Physical start y
        (uint32_t)matmul_physical_end_coord.x,    // 3  Physical end x
        (uint32_t)matmul_physical_end_coord.y,    // 4  Physical end y
        C,                                        // 5  Number of cores per row (C dim)
        in0_sender_sem_id,                        // 6  in0 sender semaphore ID
        in0_sender_valid_sem_id,                  // 7  in0 sender-valid semaphore ID (init 1)
        in0_receiver_sem_id,                      // 8  in0 receiver semaphore ID
    };

    vector<uint32_t> risc1_compile_args = {
        test_config.test_id,                      // 0  Test ID
        (uint32_t)matmul_physical_start_coord.x,  // 1  Physical start x
        (uint32_t)matmul_physical_start_coord.y,  // 2  Physical start y
        (uint32_t)matmul_physical_end_coord.x,    // 3  Physical end x
        (uint32_t)matmul_physical_end_coord.y,    // 4  Physical end y
        R,                                        // 5  Number of cores per column (R dim)
        in1_sender_sem_id,                        // 6  in1 sender semaphore ID
        in1_sender_valid_sem_id,                  // 7  in1 sender-valid semaphore ID (init 1)
        in1_receiver_sem_id,                      // 8  in1 receiver semaphore ID
    };

    // ---- Kernels ----
    auto risc0_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/matmul/kernels/in0_kernel_2d.cpp",
        matmul_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = risc0_compile_args});

    auto risc1_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/matmul/kernels/in1_kernel_2d.cpp",
        matmul_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = risc1_compile_args});

    // ---- Assign Runtime Args ----
    for (const auto& core : matmul_cores_list) {
        uint32_t col_idx = core.x - test_config.start_logical_core.x;
        uint32_t row_idx = core.y - test_config.start_logical_core.y;

        // Count in0 K subblocks this column sends
        uint32_t num_in0_k_this_core = 0;
        for (uint32_t k = col_idx; k < K; k += C) {
            num_in0_k_this_core++;
        }

        // Count in1 K subblocks this row sends
        uint32_t num_in1_k_this_core = 0;
        for (uint32_t k = row_idx; k < K; k += R) {
            num_in1_k_this_core++;
        }

        vector<uint32_t> risc0_core_runtime_args = {
            l1_base_address,                       // 0  L1 source address for in0 data
            num_in0_k_this_core,                   // 1  Number of in0 K subblocks this core sends
            in0_k_subblock_size_bytes,             // 2  Bytes per in0 K subblock
            in0_mcast_output_addr,                 // 3  in0 multicast output base address
            col_idx,                               // 4  This core's column index
            K,                                     // 5  Total K iterations
            risc0_barrier_sem_id,                  // 6  Barrier arrival semaphore ID
            (uint32_t)barrier_coordinator_phys.x,  // 7  Barrier coordinator physical X
            (uint32_t)barrier_coordinator_phys.y,  // 8  Barrier coordinator physical Y
            num_cores,                             // 9  Total number of cores in barrier
            risc0_local_barrier_addr,              // 10 Local L1 scratch addr for barrier
            risc0_barrier_done_sem_id,             // 11 Barrier done semaphore ID
        };
        // Append per-column physical X coordinates
        for (uint32_t c = 0; c < C; c++) {
            risc0_core_runtime_args.push_back(col_phys_x[c]);
        }

        vector<uint32_t> risc1_core_runtime_args = {
            in1_source_addr,                       // 0  L1 source address for in1 data
            num_in1_k_this_core,                   // 1  Number of in1 K subblocks this core sends
            in1_k_subblock_size_bytes,             // 2  Bytes per in1 K subblock
            in1_mcast_output_addr,                 // 3  in1 multicast output base address
            row_idx,                               // 4  This core's row index
            K,                                     // 5  Total K iterations
            risc1_barrier_sem_id,                  // 6  Barrier arrival semaphore ID
            (uint32_t)barrier_coordinator_phys.x,  // 7  Barrier coordinator physical X
            (uint32_t)barrier_coordinator_phys.y,  // 8  Barrier coordinator physical Y
            num_cores,                             // 9  Total number of cores in barrier
            risc1_local_barrier_addr,              // 10 Local L1 scratch addr for barrier
            risc1_barrier_done_sem_id,             // 11 Barrier done semaphore ID
        };
        // Append per-row physical Y coordinates
        for (uint32_t r = 0; r < R; r++) {
            risc1_core_runtime_args.push_back(row_phys_y[r]);
        }

        tt::tt_metal::SetRuntimeArgs(program, risc0_kernel, core, risc0_core_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, risc1_kernel, core, risc1_core_runtime_args);
    }

    // ---- Launch ----
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, runtime_host_id);
    program.set_runtime_id(runtime_host_id++);

    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // ---- Verify in0 multicast output ----
    // After the K loop, each core should have all K subblocks for its row in order
    for (const auto& core : matmul_cores_list) {
        vector<uint32_t> in0_read_output;
        detail::ReadFromDeviceL1(device, core, in0_mcast_output_addr, in0_output_total_bytes, in0_read_output);
        const vector<uint32_t>& expected_in0 = dim_r_to_in0_pages_map[core.y];
        bool is_equal = (expected_in0 == in0_read_output);
        if (!is_equal) {
            log_error(tt::LogTest, "Core {}: in0 multicast output does not match golden output!", core.str());
            log_info(tt::LogTest, "Golden vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(expected_in0));
            log_info(tt::LogTest, "Output vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(in0_read_output));
            return is_equal;
        }
    }

    // ---- Verify in1 multicast output ----
    // After the K loop, each core should have all K subblocks for its column in order
    for (const auto& core : matmul_cores_list) {
        vector<uint32_t> in1_read_output;
        detail::ReadFromDeviceL1(device, core, in1_mcast_output_addr, in1_output_total_bytes, in1_read_output);
        const vector<uint32_t>& expected_in1 = dim_c_to_in1_pages_map[core.x];
        bool is_equal = (expected_in1 == in1_read_output);
        if (!is_equal) {
            log_error(tt::LogTest, "Core {}: in1 multicast output does not match golden output!", core.str());
            log_info(tt::LogTest, "Golden vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(expected_in1));
            log_info(tt::LogTest, "Output vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(in1_read_output));
            return is_equal;
        }
    }

    return true;
}

bool run_single_test(const shared_ptr<distributed::MeshDevice>& mesh_device, MatmulTestConfig test_config) {
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);
    test_config.page_size_bytes = bytes_per_page;
    test_config.end_logical_core = CoreCoord(
        test_config.start_logical_core.x + test_config.num_subblocks_c_dim - 1,
        test_config.start_logical_core.y + test_config.num_subblocks_r_dim - 1);
    return run_dm_2d_matmul(mesh_device, test_config);
}

}  // namespace unit_tests::dm::two_d_matmul

TEST_P(Matmul2DParamFixture, Test2DMatmul) {
    EXPECT_TRUE(unit_tests::dm::two_d_matmul::run_single_test(get_mesh_device(), GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    Matmul2DSweep,
    Matmul2DParamFixture,
    ::testing::ValuesIn(unit_tests::dm::matmul::get_matmul_test_configs()),
    unit_tests::dm::matmul::MatmulTestNameGenerator());

}  // namespace tt::tt_metal
