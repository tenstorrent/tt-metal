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

namespace unit_tests::dm::one_d_matmul {

using MatmulTestConfig = unit_tests::dm::matmul::MatmulTestConfig;

uint32_t runtime_host_id = 0;

/// @brief 1D matmul: column 0 is the fixed in0 sender, row 0 is the fixed in1 sender
///        (reads its full column from DRAM once, then K-loop multicasts down the column).
bool run_dm_1d_matmul(const shared_ptr<distributed::MeshDevice>& mesh_device, const MatmulTestConfig& test_config) {
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

    // First column receives in0 source data from the host.
    CoreCoord in0_logical_start_coord = test_config.start_logical_core;
    CoreCoord in0_logical_end_coord = CoreCoord(test_config.start_logical_core.x, test_config.end_logical_core.y);
    CoreRangeSet in0_cores({CoreRange(in0_logical_start_coord, in0_logical_end_coord)});
    vector<CoreCoord> in0_cores_list = corerange_to_cores(in0_cores);

    CoreCoord matmul_physical_start_coord = device->worker_core_from_logical_core(test_config.start_logical_core);
    CoreCoord matmul_physical_end_coord = device->worker_core_from_logical_core(test_config.end_logical_core);

    uint32_t in0_pages = (test_config.num_subblocks_r_dim * test_config.subblock_r_dim) *
                         (test_config.num_subblocks_k_dim * test_config.subblock_k_dim);
    uint32_t in1_pages = (test_config.num_subblocks_k_dim * test_config.subblock_k_dim) *
                         (test_config.num_subblocks_c_dim * test_config.subblock_c_dim);
    uint32_t in0_pages_bytes = in0_pages * test_config.page_size_bytes;
    uint32_t in1_pages_bytes = in1_pages * test_config.page_size_bytes;

    uint32_t k_subblock_size_bytes =
        test_config.subblock_r_dim * test_config.subblock_k_dim * test_config.page_size_bytes;
    uint32_t in1_k_subblock_size_bytes =
        test_config.subblock_k_dim * test_config.subblock_c_dim * test_config.page_size_bytes;

    uint32_t l1_base_address = unit_tests::dm::get_l1_address_and_size(mesh_device, in0_cores_list[0]).base_address;

    vector<uint32_t> in0_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, in0_pages_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    vector<uint32_t> in0_per_core_pages;
    unordered_map<uint32_t, vector<uint32_t>> dim_r_to_in0_pages_map;
    uint32_t pages_per_core = in0_input.size() / in0_cores_list.size();
    uint32_t pages_per_core_size_bytes = pages_per_core * sizeof(uint32_t);
    for (uint32_t i = 0; i < in0_cores_list.size(); i++) {
        for (uint32_t j = 0; j < pages_per_core; j++) {
            in0_per_core_pages.push_back(in0_input[i * pages_per_core + j]);
        }
        dim_r_to_in0_pages_map[in0_cores_list[i].y] =
            vector<uint32_t>(in0_per_core_pages.begin(), in0_per_core_pages.end());
        detail::WriteToDeviceL1(device, in0_cores_list[i], l1_base_address, in0_per_core_pages);
        in0_per_core_pages.clear();
    }

    vector<uint32_t> in1_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, in1_pages_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    DramAddressInfo dram_info = unit_tests::dm::get_dram_address_and_size();
    uint32_t input_dram_address = dram_info.base_address;
    detail::WriteToDeviceDRAMChannel(device, test_config.dram_bank_id, input_dram_address, in1_input);
    MetalContext::instance().get_cluster().dram_barrier(device->id());

    uint32_t in1_per_core_read_size_bytes = in1_pages_bytes / test_config.num_subblocks_c_dim;
    vector<uint32_t> in1_per_core_read_addr;
    in1_per_core_read_addr.reserve(test_config.num_subblocks_c_dim);
    for (uint32_t i = 0; i < test_config.num_subblocks_c_dim; i++) {
        in1_per_core_read_addr.push_back(input_dram_address + i * in1_per_core_read_size_bytes);
    }

    uint32_t in0_mcast_output_addr = l1_base_address + pages_per_core_size_bytes + matmul::L1_DEBUG_PADDING_BYTES;

    // 64-byte alignment required so low bits match the DRAM source address (Blackhole).
    uint32_t in1_l1_source_addr_unaligned =
        l1_base_address + ((pages_per_core_size_bytes + matmul::L1_DEBUG_PADDING_BYTES) << 1);
    uint32_t in1_l1_source_addr = (in1_l1_source_addr_unaligned + 63) & ~63U;
    uint32_t in1_mcast_output_addr = in1_l1_source_addr + in1_per_core_read_size_bytes + matmul::L1_DEBUG_PADDING_BYTES;

    // Three-semaphore mcast protocol: receivers inc sender_sem to signal ready; sender
    // multicasts data, then multicasts sender_valid_sem (init 1) into receiver_sem on all
    // cores; receivers wait for receiver_sem == 1 and reset it.
    uint32_t sender_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t sender_valid_sem_id = CreateSemaphore(program, matmul_cores, 1);
    uint32_t receiver_sem_id = CreateSemaphore(program, matmul_cores, 0);

    // Same protocol, column-wise on RISCV_1 (in1 path, row 0 is the fixed sender).
    uint32_t risc1_sender_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t risc1_sender_valid_sem_id = CreateSemaphore(program, matmul_cores, 1);
    uint32_t risc1_receiver_sem_id = CreateSemaphore(program, matmul_cores, 0);

    // Independent barriers per RISC: arrival counter + done broadcast.
    uint32_t risc0_barrier_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t risc0_barrier_done_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t risc1_barrier_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t risc1_barrier_done_sem_id = CreateSemaphore(program, matmul_cores, 0);

    CoreCoord barrier_coordinator_phys = device->worker_core_from_logical_core(matmul_cores_list[0]);
    uint32_t num_cores = matmul_cores_list.size();

    uint32_t risc0_local_barrier_addr = (in1_mcast_output_addr + in1_per_core_read_size_bytes + 15) & ~15U;
    uint32_t risc1_local_barrier_addr = risc0_local_barrier_addr + 16;

    vector<uint32_t> risc0_compile_args = {
        test_config.test_id,                      // 0  Test ID
        (uint32_t)matmul_physical_start_coord.x,  // 1  Physical start x (== sender column x)
        (uint32_t)matmul_physical_start_coord.y,  // 2  Physical start y
        (uint32_t)matmul_physical_end_coord.x,    // 3  Physical end x
        (uint32_t)matmul_physical_end_coord.y,    // 4  Physical end y
        test_config.num_subblocks_c_dim,          // 5  Number of cores per row (C dim)
        sender_sem_id,                            // 6  Sender semaphore ID
        sender_valid_sem_id,                      // 7  Sender-valid semaphore ID (init 1)
        receiver_sem_id,                          // 8  Receiver semaphore ID
    };

    vector<uint32_t> risc1_compile_args = {
        test_config.test_id,                      // 0  Test ID
        test_config.dram_bank_id,                 // 1  DRAM bank row-0 cores read from
        (uint32_t)matmul_physical_start_coord.x,  // 2  Physical start x (for barrier mcast)
        (uint32_t)matmul_physical_start_coord.y,  // 3  Physical start y (== sender row y)
        (uint32_t)matmul_physical_end_coord.x,    // 4  Physical end x
        (uint32_t)matmul_physical_end_coord.y,    // 5  Physical end y
        test_config.num_subblocks_r_dim,          // 6  Number of cores per column (R dim)
        risc1_sender_sem_id,                      // 7  in1 sender semaphore ID
        risc1_sender_valid_sem_id,                // 8  in1 sender-valid semaphore ID (init 1)
        risc1_receiver_sem_id,                    // 9  in1 receiver semaphore ID
    };

    auto risc0_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/matmul/kernels/in0_kernel.cpp",
        matmul_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = risc0_compile_args});

    auto risc1_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/matmul/kernels/in1_kernel.cpp",
        matmul_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = risc1_compile_args});

    for (auto & i : matmul_cores_list) {
        vector<uint32_t> risc0_core_runtime_args = {
            l1_base_address,                       // 0  Sender: source addr of in0 data (all K subblocks contiguous)
            test_config.num_subblocks_k_dim,       // 1  Number of K subblocks (loop iterations)
            k_subblock_size_bytes,                 // 2  Bytes per K subblock multicast
            in0_mcast_output_addr,                 // 3  All cores: L1 dest base addr for multicast data
            risc0_barrier_sem_id,                  // 4  Barrier arrival semaphore ID
            (uint32_t)barrier_coordinator_phys.x,  // 5  Barrier coordinator physical X
            (uint32_t)barrier_coordinator_phys.y,  // 6  Barrier coordinator physical Y
            num_cores,                             // 7  Total number of cores in barrier
            risc0_local_barrier_addr,              // 8  Local L1 scratch addr for barrier
            risc0_barrier_done_sem_id,             // 9  Barrier done semaphore ID
        };
        uint32_t col_idx = i.x - test_config.start_logical_core.x;
        vector<uint32_t> risc1_core_runtime_args = {
            in1_per_core_read_addr[col_idx],       // 0  Per-column DRAM offset (row 0 only reads it)
            in1_per_core_read_size_bytes,          // 1  Total bytes to read from DRAM (row 0 only)
            test_config.num_subblocks_k_dim,       // 2  K iterations
            in1_k_subblock_size_bytes,             // 3  Bytes per K subblock multicast
            in1_l1_source_addr,                    // 4  Row-0 DRAM read destination in L1
            in1_mcast_output_addr,                 // 5  All cores: column multicast dest base
            risc1_barrier_sem_id,                  // 6  Barrier arrival semaphore ID
            (uint32_t)barrier_coordinator_phys.x,  // 7  Barrier coordinator physical X
            (uint32_t)barrier_coordinator_phys.y,  // 8  Barrier coordinator physical Y
            num_cores,                             // 9  Total number of cores in barrier
            risc1_local_barrier_addr,              // 10 Local L1 scratch addr for barrier
            risc1_barrier_done_sem_id,             // 11 Barrier done semaphore ID
        };
        tt::tt_metal::SetRuntimeArgs(program, risc0_kernel, i, risc0_core_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, risc1_kernel, i, risc1_core_runtime_args);
    }

    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    for (auto & i : matmul_cores_list) {
        vector<uint32_t> in0_read_output;
        detail::ReadFromDeviceL1(
            device, i, in0_mcast_output_addr, pages_per_core_size_bytes, in0_read_output);
        const vector<uint32_t>& expected_in0_read_output = dim_r_to_in0_pages_map[i.y];
        bool is_equal = (expected_in0_read_output == in0_read_output);
        if (!is_equal) {
            log_error(
                tt::LogTest, "Core {}: in0 read output does not match golden output!", i.str());
            log_info(tt::LogTest, "Golden vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(in0_input));
            log_info(tt::LogTest, "Output vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(in0_read_output));

            return is_equal;
        }
    }

    vector<uint32_t> golden_in1_read_output;
    uint32_t total_in1_read_elements = in1_per_core_read_size_bytes / sizeof(uint32_t);
    for (auto & i : matmul_cores_list) {
        vector<uint32_t> in1_read_output;
        detail::ReadFromDeviceL1(device, i, in1_mcast_output_addr, in1_per_core_read_size_bytes, in1_read_output);
        uint32_t cur_c_dim = i.x - test_config.start_logical_core.x;
        for (uint32_t j = 0; j < total_in1_read_elements; j++) {
            golden_in1_read_output.push_back(in1_input[cur_c_dim * total_in1_read_elements + j]);
        }
        bool is_equal = (golden_in1_read_output == in1_read_output);
        if (!is_equal) {
            log_error(
                tt::LogTest, "Core {}: in1 read output does not match golden output!", i.str());
            log_info(tt::LogTest, "Golden vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(in1_input));
            log_info(tt::LogTest, "Packed Output vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(golden_in1_read_output));
            log_info(tt::LogTest, "Output vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(in1_read_output));

            return is_equal;
        }
        golden_in1_read_output.clear();
    }

    return true;
}

bool run_single_test(const shared_ptr<distributed::MeshDevice>& mesh_device, MatmulTestConfig test_config) {
    test_config.test_id += unit_tests::dm::matmul::MATMUL_1D_TEST_ID_OFFSET;
    test_config.page_size_bytes = 2048;
    test_config.end_logical_core = CoreCoord(
        test_config.start_logical_core.x + test_config.num_subblocks_c_dim - 1,
        test_config.start_logical_core.y + test_config.num_subblocks_r_dim - 1);
    return run_dm_1d_matmul(mesh_device, test_config);
}

bool run_multiple_test(const shared_ptr<distributed::MeshDevice>& mesh_device, const MatmulTestConfig& test_config) {
    auto or_scalar = [](const std::vector<uint32_t>& v, uint32_t scalar) {
        return v.empty() ? std::vector<uint32_t>{scalar} : v;
    };

    auto nr_vals = or_scalar(test_config.num_subblocks_r_dim_sweep, test_config.num_subblocks_r_dim);
    auto nc_vals = or_scalar(test_config.num_subblocks_c_dim_sweep, test_config.num_subblocks_c_dim);
    auto nk_vals = or_scalar(test_config.num_subblocks_k_dim_sweep, test_config.num_subblocks_k_dim);
    auto sr_vals = or_scalar(test_config.subblock_r_dim_sweep, test_config.subblock_r_dim);
    auto sc_vals = or_scalar(test_config.subblock_c_dim_sweep, test_config.subblock_c_dim);
    auto sk_vals = or_scalar(test_config.subblock_k_dim_sweep, test_config.subblock_k_dim);

    bool all_pass = true;
    for (uint32_t nr : nr_vals) {
        for (uint32_t nc : nc_vals) {
            for (uint32_t nk : nk_vals) {
                for (uint32_t sr : sr_vals) {
                    for (uint32_t sc : sc_vals) {
                        for (uint32_t sk : sk_vals) {
                            MatmulTestConfig single = test_config;
                            single.num_subblocks_r_dim = nr;
                            single.num_subblocks_c_dim = nc;
                            single.num_subblocks_k_dim = nk;
                            single.subblock_r_dim = sr;
                            single.subblock_c_dim = sc;
                            single.subblock_k_dim = sk;
                            all_pass &= run_single_test(mesh_device, single);
                        }
                    }
                }
            }
        }
    }
    return all_pass;
}

}  // namespace unit_tests::dm::one_d_matmul

TEST_P(Matmul1DParamFixture, Test1DMatmul) {
    EXPECT_TRUE(unit_tests::dm::one_d_matmul::run_multiple_test(get_mesh_device(), GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    Matmul1DSweep,
    Matmul1DParamFixture,
    ::testing::ValuesIn(unit_tests::dm::matmul::get_matmul_test_configs()),
    unit_tests::dm::matmul::MatmulTestNameGenerator());

}  // namespace tt::tt_metal
