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

namespace unit_tests::dm::one_d_matmul_v2 {

using MatmulTestConfig = unit_tests::dm::matmul::MatmulTestConfig;

uint32_t runtime_host_id = 0;

/// @brief 1D matmul v2: in0 senders rotate round-robin across columns; in1 keeps the
///        v1 row-0 fixed-sender DRAM-read path.
bool run_dm_1d_matmul_v2(const shared_ptr<distributed::MeshDevice>& mesh_device, const MatmulTestConfig& test_config) {
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

    vector<uint32_t> col_phys_x(grid_cols);
    for (uint32_t c = 0; c < grid_cols; c++) {
        CoreCoord logical_col_core(test_config.start_logical_core.x + c, test_config.start_logical_core.y);
        CoreCoord phys = device->worker_core_from_logical_core(logical_col_core);
        col_phys_x[c] = phys.x;
    }

    uint32_t K = test_config.num_subblocks_k_dim;
    uint32_t C = test_config.num_subblocks_c_dim;
    uint32_t R = test_config.num_subblocks_r_dim;

    uint32_t k_subblock_size_bytes =
        test_config.subblock_r_dim * test_config.subblock_k_dim * test_config.page_size_bytes;
    uint32_t max_k_per_col = (K + C - 1) / C;
    uint32_t max_source_data_bytes = max_k_per_col * k_subblock_size_bytes;

    uint32_t in1_pages = (test_config.num_subblocks_k_dim * test_config.subblock_k_dim) *
                         (test_config.num_subblocks_c_dim * test_config.subblock_c_dim);
    uint32_t in1_pages_bytes = in1_pages * test_config.page_size_bytes;
    uint32_t in1_per_core_read_size_bytes = in1_pages_bytes / test_config.num_subblocks_c_dim;
    uint32_t in1_k_subblock_size_bytes =
        test_config.subblock_k_dim * test_config.subblock_c_dim * test_config.page_size_bytes;

    uint32_t l1_base_address = unit_tests::dm::get_l1_address_and_size(mesh_device, matmul_cores_list[0]).base_address;

    // L1 layout: in0 source -> in0 mcast out -> in1 source (64B-aligned) -> in1 mcast out -> barrier scratch.
    uint32_t in0_mcast_output_addr = l1_base_address + max_source_data_bytes + matmul::L1_DEBUG_PADDING_BYTES;
    uint32_t in0_output_total_bytes = K * k_subblock_size_bytes;

    uint32_t in1_l1_source_addr_unaligned =
        in0_mcast_output_addr + in0_output_total_bytes + matmul::L1_DEBUG_PADDING_BYTES;
    uint32_t in1_l1_source_addr = (in1_l1_source_addr_unaligned + 63) & ~63U;
    uint32_t in1_mcast_output_addr = in1_l1_source_addr + in1_per_core_read_size_bytes + matmul::L1_DEBUG_PADDING_BYTES;

    uint32_t in0_pages = (R * test_config.subblock_r_dim) * (K * test_config.subblock_k_dim);
    uint32_t in0_pages_bytes = in0_pages * test_config.page_size_bytes;
    vector<uint32_t> in0_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, in0_pages_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    uint32_t row_data_size_uint32 = in0_input.size() / R;
    uint32_t k_subblock_size_uint32 = k_subblock_size_bytes / sizeof(uint32_t);

    // Golden in0 row data, indexed by row's logical y.
    unordered_map<uint32_t, vector<uint32_t>> dim_r_to_in0_pages_map;
    for (uint32_t r = 0; r < R; r++) {
        vector<uint32_t> row_data(
            in0_input.begin() + r * row_data_size_uint32, in0_input.begin() + (r + 1) * row_data_size_uint32);
        dim_r_to_in0_pages_map[test_config.start_logical_core.y + r] = row_data;
    }

    // Each core (c, r) gets K subblocks {c, c+C, c+2C, ...} from row r.
    for (auto & core_idx : matmul_cores_list) {
        uint32_t col_idx = core_idx.x - test_config.start_logical_core.x;
        uint32_t row_idx = core_idx.y - test_config.start_logical_core.y;

        vector<uint32_t> core_in0_data;
        uint32_t row_base = row_idx * row_data_size_uint32;
        for (uint32_t k = col_idx; k < K; k += C) {
            uint32_t k_offset = k * k_subblock_size_uint32;
            for (uint32_t e = 0; e < k_subblock_size_uint32; e++) {
                core_in0_data.push_back(in0_input[row_base + k_offset + e]);
            }
        }

        if (!core_in0_data.empty()) {
            detail::WriteToDeviceL1(device, core_idx, l1_base_address, core_in0_data);
        }
    }

    vector<uint32_t> in1_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, in1_pages_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    DramAddressInfo dram_info = unit_tests::dm::get_dram_address_and_size();
    uint32_t input_dram_address = dram_info.base_address;
    detail::WriteToDeviceDRAMChannel(device, test_config.dram_bank_id, input_dram_address, in1_input);
    MetalContext::instance().get_cluster().dram_barrier(device->id());

    vector<uint32_t> in1_per_core_read_addr;
    in1_per_core_read_addr.reserve(C);
    for (uint32_t i = 0; i < C; i++) {
        in1_per_core_read_addr.push_back(input_dram_address + i * in1_per_core_read_size_bytes);
    }

    // in0 row-wise mcast (rotating sender) and in1 column-wise mcast (row 0 fixed sender).
    uint32_t sender_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t sender_valid_sem_id = CreateSemaphore(program, matmul_cores, 1);
    uint32_t receiver_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t risc1_sender_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t risc1_sender_valid_sem_id = CreateSemaphore(program, matmul_cores, 1);
    uint32_t risc1_receiver_sem_id = CreateSemaphore(program, matmul_cores, 0);

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
        (uint32_t)matmul_physical_start_coord.x,  // 1  Physical start x
        (uint32_t)matmul_physical_start_coord.y,  // 2  Physical start y
        (uint32_t)matmul_physical_end_coord.x,    // 3  Physical end x
        (uint32_t)matmul_physical_end_coord.y,    // 4  Physical end y
        C,                                        // 5  Number of cores per row (C dim)
        sender_sem_id,                            // 6  Sender semaphore ID
        sender_valid_sem_id,                      // 7  Sender-valid semaphore ID (init 1)
        receiver_sem_id,                          // 8  Receiver semaphore ID
    };

    vector<uint32_t> risc1_compile_args = {
        test_config.test_id,                      // 0  Test ID
        test_config.dram_bank_id,                 // 1  DRAM bank row-0 cores read from
        (uint32_t)matmul_physical_start_coord.x,  // 2  Physical start x (for barrier)
        (uint32_t)matmul_physical_start_coord.y,  // 3  Physical start y (== sender row y)
        (uint32_t)matmul_physical_end_coord.x,    // 4  Physical end x
        (uint32_t)matmul_physical_end_coord.y,    // 5  Physical end y
        R,                                        // 6  Number of cores per column (R dim)
        risc1_sender_sem_id,                      // 7  in1 sender semaphore ID
        risc1_sender_valid_sem_id,                // 8  in1 sender-valid semaphore ID (init 1)
        risc1_receiver_sem_id,                    // 9  in1 receiver semaphore ID
    };

    auto risc0_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/matmul/kernels/in0_kernel_v2.cpp",
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
        uint32_t col_idx = i.x - test_config.start_logical_core.x;

        uint32_t num_k_this_core = 0;
        for (uint32_t k = col_idx; k < K; k += C) {
            num_k_this_core++;
        }

        vector<uint32_t> risc0_core_runtime_args = {
            l1_base_address,                       // 0  L1 source address for in0 data
            num_k_this_core,                       // 1  Number of K subblocks this core sends
            k_subblock_size_bytes,                 // 2  Bytes per K subblock
            in0_mcast_output_addr,                 // 3  Multicast output base address
            col_idx,                               // 4  This core's column index
            K,                                     // 5  Total K iterations
            risc0_barrier_sem_id,                  // 6  Barrier arrival semaphore ID
            (uint32_t)barrier_coordinator_phys.x,  // 7  Barrier coordinator physical X
            (uint32_t)barrier_coordinator_phys.y,  // 8  Barrier coordinator physical Y
            num_cores,                             // 9  Total number of cores in barrier
            risc0_local_barrier_addr,              // 10 Local L1 scratch addr for barrier
            risc0_barrier_done_sem_id,             // 11 Barrier done semaphore ID
        };
        for (uint32_t c = 0; c < C; c++) {
            risc0_core_runtime_args.push_back(col_phys_x[c]);
        }

        vector<uint32_t> risc1_core_runtime_args = {
            in1_per_core_read_addr[col_idx],       // 0  Per-column DRAM offset (row 0 only reads it)
            in1_per_core_read_size_bytes,          // 1  Total bytes to read from DRAM (row 0 only)
            K,                                     // 2  K iterations
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

    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, runtime_host_id);
    program.set_runtime_id(runtime_host_id++);

    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // Each core should hold all K subblocks for its row, in order.
    for (auto & i : matmul_cores_list) {
        vector<uint32_t> in0_read_output;
        detail::ReadFromDeviceL1(
            device, i, in0_mcast_output_addr, in0_output_total_bytes, in0_read_output);
        const vector<uint32_t>& expected_in0_read_output = dim_r_to_in0_pages_map[i.y];
        bool is_equal = (expected_in0_read_output == in0_read_output);
        if (!is_equal) {
            log_error(
                tt::LogTest, "Core {}: in0 read output does not match golden output!", i.str());
            log_info(tt::LogTest, "Golden vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(expected_in0_read_output));
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
    test_config.test_id += unit_tests::dm::matmul::MATMUL_1D_V2_TEST_ID_OFFSET;
    test_config.page_size_bytes = 2048;
    test_config.end_logical_core = CoreCoord(
        test_config.start_logical_core.x + test_config.num_subblocks_c_dim - 1,
        test_config.start_logical_core.y + test_config.num_subblocks_r_dim - 1);
    return run_dm_1d_matmul_v2(mesh_device, test_config);
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

}  // namespace unit_tests::dm::one_d_matmul_v2

TEST_P(Matmul1DV2ParamFixture, Test1DMatmulV2) {
    EXPECT_TRUE(unit_tests::dm::one_d_matmul_v2::run_multiple_test(get_mesh_device(), GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    Matmul1DV2Sweep,
    Matmul1DV2ParamFixture,
    ::testing::ValuesIn(unit_tests::dm::matmul::get_matmul_test_configs()),
    unit_tests::dm::matmul::MatmulTestNameGenerator());

}  // namespace tt::tt_metal
