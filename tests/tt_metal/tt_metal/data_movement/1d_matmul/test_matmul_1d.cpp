// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
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

uint32_t runtime_host_id = 0;

// Test config
struct one_d_Matmul_Config {
    uint32_t test_id = 0;
    CoreCoord origin_logical_core;
    CoreCoord start_logical_core;
    CoreCoord end_logical_core;
    uint32_t num_subblocks_r_dim = 2;  // how many subblocks in r dim
    uint32_t num_subblocks_c_dim = 2;
    uint32_t num_subblocks_k_dim = 1;
    uint32_t subblock_r_dim = 1;  // how many pages each subblock in r dim takes up
    uint32_t subblock_c_dim = 1;
    uint32_t subblock_k_dim = 1;
    uint32_t page_size_bytes = 1;
    DataFormat l1_data_format = DataFormat::Float16_b;
    uint32_t dram_bank_id = 0;  // dram bank that all cores will read from
};

/// @brief Reads from DRAM to L1 with each core reading only its adjacent bank
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test
/// @return true if test passes
bool run_dm_1d_matmul(const shared_ptr<distributed::MeshDevice>& mesh_device, const one_d_Matmul_Config& test_config) {
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

    // Validate grid dimensions match config
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

    // Logical core sets for in0. The first column of cores will need the host writing the content into L1.
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

    uint32_t l1_base_address = unit_tests::dm::get_l1_address_and_size(mesh_device, in0_cores_list[0]).base_address;

    // in0 Input
    vector<uint32_t> in0_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, in0_pages_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    // Clear all matmul cores L1 with zeros before writing in0 input
    vector<uint32_t> zeros(((in0_pages_bytes + in1_pages_bytes) << 1) / sizeof(uint32_t), 0);
    for (const auto& core : matmul_cores_list) {
        detail::WriteToDeviceL1(device, core, l1_base_address, zeros);
    }

    // Write in0 input to L1 of the first column of cores
    vector<uint32_t> in0_per_core_pages;
    unordered_map<uint32_t, vector<uint32_t>> dim_r_to_in0_pages_map;
    uint32_t pages_per_core = in0_input.size() / in0_cores_list.size();
    uint32_t pages_per_core_size_bytes = pages_per_core * sizeof(uint32_t);
    for (uint32_t i = 0; i < in0_cores_list.size(); i++) {
        // in0_per_core_pages should contain the pages that the i-th core in in0_cores_list will read
        for (uint32_t j = 0; j < pages_per_core; j++) {
            in0_per_core_pages.push_back(in0_input[i * pages_per_core + j]);
        }
        dim_r_to_in0_pages_map[in0_cores_list[i].y] =
            vector<uint32_t>(in0_per_core_pages.begin(), in0_per_core_pages.end());
        detail::WriteToDeviceL1(device, in0_cores_list[i], l1_base_address, in0_per_core_pages);
        in0_per_core_pages.clear();
    }

    // in1 Input
    vector<uint32_t> in1_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, in1_pages_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    // DRAM Address
    DramAddressInfo dram_info = unit_tests::dm::get_dram_address_and_size();
    uint32_t input_dram_address = dram_info.base_address;

    // Write Input to DRAM
    detail::WriteToDeviceDRAMChannel(device, test_config.dram_bank_id, input_dram_address, in1_input);
    MetalContext::instance().get_cluster().dram_barrier(device->id());

    // in1_per_core_read_size_bytes is how much each core should read from the DRAM bank
    uint32_t in1_per_core_read_size_bytes = in1_pages_bytes / test_config.num_subblocks_c_dim;
    // in1_per_core_read_addr stores the memory address where each core should read from DRAM bank
    vector<uint32_t> in1_per_core_read_addr;
    for (uint32_t i = 0; i < test_config.num_subblocks_c_dim; i++) {
        in1_per_core_read_addr.push_back(input_dram_address + i * in1_per_core_read_size_bytes);
    }
    // in0_mcast_output_addr is the memory address where each core will leave the in0 mcast output in L1
    uint32_t in0_mcast_output_addr = l1_base_address + pages_per_core_size_bytes + 0x10;

    // in1_output_addr is the memory address where each core will leave the in1 read output in L1. It is placed after
    // in0 mcast output in L1. Must be aligned to NOC_DRAM_READ_ALIGNMENT_BYTES (64 on Blackhole) so the low bits match
    // the DRAM source address.
    uint32_t in1_output_addr_unaligned = l1_base_address + ((pages_per_core_size_bytes + 0x10) << 1);
    uint32_t in1_output_addr = (in1_output_addr_unaligned + 63) & ~63U;

    // ---- in0 multicast semaphores ----
    // sender_sem: lives on every core, but only the sender (first-column) core waits on it.
    //   Receivers increment it via noc_semaphore_inc to signal they are ready.
    //   Sender waits until value == (num_subblocks_c_dim - 1).
    uint32_t sender_sem_id = CreateSemaphore(program, matmul_cores, 0);

    // sender_valid_sem: lives on every core, initialised to 1.
    //   The sender uses its local copy as the *source value* when it calls
    //   noc_semaphore_set_multicast_loopback_src to set the receiver_sem on
    //   all cores in the row to 1.
    uint32_t sender_valid_sem_id = CreateSemaphore(program, matmul_cores, 1);

    // receiver_sem: lives on every core, initialised to 0.
    //   After the sender multicasts data, it also multicasts sender_valid_sem's
    //   value (1) into receiver_sem on every core in the row.
    //   Each receiver waits until receiver_sem == 1, then resets it to 0.
    uint32_t receiver_sem_id = CreateSemaphore(program, matmul_cores, 0);

    // ---- barrier synchronization semaphores ----
    // One barrier per RISC processor so RISCV_0 and RISCV_1 synchronize independently.
    // Each barrier uses two semaphores: one for arrival counting, one for done broadcast.
    uint32_t risc0_barrier_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t risc0_barrier_done_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t risc1_barrier_sem_id = CreateSemaphore(program, matmul_cores, 0);
    uint32_t risc1_barrier_done_sem_id = CreateSemaphore(program, matmul_cores, 0);

    // Coordinator = first matmul core; all cores increment its semaphore, coordinator multicasts done.
    CoreCoord barrier_coordinator_phys = device->worker_core_from_logical_core(matmul_cores_list[0]);
    uint32_t num_cores = matmul_cores_list.size();

    // Local scratch addresses for barrier (placed after in1 output region, 16-byte aligned)
    uint32_t risc0_local_barrier_addr = (in1_output_addr + in1_per_core_read_size_bytes + 15) & ~15U;
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
        test_config.dram_bank_id,                 // 1  DRAM bank that all cores will read from
        (uint32_t)matmul_physical_start_coord.x,  // 2  Physical start x (for barrier mcast)
        (uint32_t)matmul_physical_start_coord.y,  // 3  Physical start y
        (uint32_t)matmul_physical_end_coord.x,    // 4  Physical end x
        (uint32_t)matmul_physical_end_coord.y,    // 5  Physical end y
    };

    // Kernels
    auto risc0_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/1d_matmul/kernels/in0_kernel.cpp",
        matmul_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = risc0_compile_args});

    auto risc1_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/1d_matmul/kernels/in1_kernel.cpp",
        matmul_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = risc1_compile_args});

    // Assign Runtime Args
    for (int i = 0; i < matmul_cores_list.size(); i++) {
        vector<uint32_t> risc0_core_runtime_args = {
            l1_base_address,                       // 0  Sender: source addr of in0 data in L1
            pages_per_core_size_bytes,             // 1  Sender: number of bytes to multicast
            in0_mcast_output_addr,                 // 2  All cores: L1 dest addr for multicast data
            risc0_barrier_sem_id,                  // 3  Barrier arrival semaphore ID
            (uint32_t)barrier_coordinator_phys.x,  // 4  Barrier coordinator physical X
            (uint32_t)barrier_coordinator_phys.y,  // 5  Barrier coordinator physical Y
            num_cores,                             // 6  Total number of cores in barrier
            risc0_local_barrier_addr,              // 7  Local L1 scratch addr for barrier
            risc0_barrier_done_sem_id,             // 8  Barrier done semaphore ID
        };
        uint32_t col_idx = matmul_cores_list[i].x - test_config.start_logical_core.x;
        vector<uint32_t> risc1_core_runtime_args = {
            in1_per_core_read_addr[col_idx],       // 0  Each core reads from addr based on its column
            in1_per_core_read_size_bytes,          // 1  Each core reads the same amount of data
            in1_output_addr,                       // 2  Each core writes to the same address in L1
            risc1_barrier_sem_id,                  // 3  Barrier arrival semaphore ID
            (uint32_t)barrier_coordinator_phys.x,  // 4  Barrier coordinator physical X
            (uint32_t)barrier_coordinator_phys.y,  // 5  Barrier coordinator physical Y
            num_cores,                             // 6  Total number of cores in barrier
            risc1_local_barrier_addr,              // 7  Local L1 scratch addr for barrier
            risc1_barrier_done_sem_id,             // 8  Barrier done semaphore ID
        };
        tt::tt_metal::SetRuntimeArgs(program, risc0_kernel, matmul_cores_list[i], risc0_core_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, risc1_kernel, matmul_cores_list[i], risc1_core_runtime_args);
    }

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // LAUNCH PROGRAM - Use mesh workload approach
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // Verify in0 read output in L1 for each core
    for (int i = 0; i < matmul_cores_list.size(); i++) {
        vector<uint32_t> in0_read_output;
        detail::ReadFromDeviceL1(
            device, matmul_cores_list[i], in0_mcast_output_addr, pages_per_core_size_bytes, in0_read_output);
        const vector<uint32_t>& expected_in0_read_output = dim_r_to_in0_pages_map[matmul_cores_list[i].y];
        bool is_equal = (expected_in0_read_output == in0_read_output);
        if (!is_equal) {
            log_error(
                tt::LogTest, "Core {}: in0 read output does not match golden output!", matmul_cores_list[i].str());
            log_info(tt::LogTest, "Golden vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(in0_input));
            log_info(tt::LogTest, "Output vector");
            print_vector(unpack_vector<bfloat16, uint32_t>(in0_read_output));

            return is_equal;
        }
    }

    // Verify in1 read output in L1 for each core
    vector<uint32_t> golden_in1_read_output;
    uint32_t total_in1_read_elements = in1_per_core_read_size_bytes / sizeof(uint32_t);
    for (int i = 0; i < matmul_cores_list.size(); i++) {
        vector<uint32_t> in1_read_output;
        detail::ReadFromDeviceL1(
            device, matmul_cores_list[i], in1_output_addr, in1_per_core_read_size_bytes, in1_read_output);
        uint32_t cur_c_dim = matmul_cores_list[i].x - test_config.start_logical_core.x;
        for (uint32_t j = 0; j < total_in1_read_elements; j++) {
            golden_in1_read_output.push_back(in1_input[cur_c_dim * total_in1_read_elements + j]);
        }
        bool is_equal = (golden_in1_read_output == in1_read_output);
        if (!is_equal) {
            log_error(
                tt::LogTest, "Core {}: in1 read output does not match golden output!", matmul_cores_list[i].str());
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

bool run_single_test(const shared_ptr<distributed::MeshDevice>& mesh_device, one_d_Matmul_Config test_config) {
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);
    test_config.page_size_bytes = bytes_per_page;
    test_config.origin_logical_core = test_config.start_logical_core;
    test_config.end_logical_core = CoreCoord(
        test_config.start_logical_core.x + test_config.num_subblocks_c_dim - 1,
        test_config.start_logical_core.y + test_config.num_subblocks_r_dim - 1);
    return run_dm_1d_matmul(mesh_device, test_config);
}

}  // namespace unit_tests::dm::one_d_matmul

/* =================================== */
/* =========== TEST CASES ============ */
/* =================================== */

using unit_tests::dm::one_d_matmul::run_single_test;

/* ======== GRID SHAPE TESTS ======== */

// 2x2 grid (default config)
TEST_F(GenericMeshDeviceFixture, Test1DMatmulIdeal) {
    EXPECT_TRUE(run_single_test(get_mesh_device(), {.test_id = 650}));
}

// 1x1: sender multicasts to itself, reads full in1 from DRAM
TEST_F(GenericMeshDeviceFixture, Test1DMatmulSingleCore) {
    EXPECT_TRUE(
        run_single_test(get_mesh_device(), {.test_id = 651, .num_subblocks_r_dim = 1, .num_subblocks_c_dim = 1}));
}

// 1x3 row: one sender multicasts in0 to 3 receivers
TEST_F(GenericMeshDeviceFixture, Test1DMatmulSingleRow) {
    EXPECT_TRUE(
        run_single_test(get_mesh_device(), {.test_id = 652, .num_subblocks_r_dim = 1, .num_subblocks_c_dim = 3}));
}

// 3x1 column: 3 independent senders each multicast to self
TEST_F(GenericMeshDeviceFixture, Test1DMatmulSingleColumn) {
    EXPECT_TRUE(
        run_single_test(get_mesh_device(), {.test_id = 653, .num_subblocks_r_dim = 3, .num_subblocks_c_dim = 1}));
}

// 2x3 non-square grid (R=3, C=2)
TEST_F(GenericMeshDeviceFixture, Test1DMatmulNonSquare2x3) {
    EXPECT_TRUE(
        run_single_test(get_mesh_device(), {.test_id = 654, .num_subblocks_r_dim = 3, .num_subblocks_c_dim = 2}));
}

// 3x2 non-square grid (R=2, C=3)
TEST_F(GenericMeshDeviceFixture, Test1DMatmulNonSquare3x2) {
    EXPECT_TRUE(
        run_single_test(get_mesh_device(), {.test_id = 655, .num_subblocks_r_dim = 2, .num_subblocks_c_dim = 3}));
}

// 4x4 large grid
TEST_F(GenericMeshDeviceFixture, Test1DMatmulLargeGrid4x4) {
    EXPECT_TRUE(
        run_single_test(get_mesh_device(), {.test_id = 656, .num_subblocks_r_dim = 4, .num_subblocks_c_dim = 4}));
}

// 6x6 large grid with K=2: 36 cores, 6-way multicast per row
TEST_F(GenericMeshDeviceFixture, Test1DMatmulLargeGrid6x6) {
    EXPECT_TRUE(run_single_test(
        get_mesh_device(),
        {.test_id = 657, .num_subblocks_r_dim = 6, .num_subblocks_c_dim = 6, .num_subblocks_k_dim = 2}));
}

// 1x8 wide row: max multicast fan-out (8 receivers)
TEST_F(GenericMeshDeviceFixture, Test1DMatmulWideMulticast8Col) {
    EXPECT_TRUE(
        run_single_test(get_mesh_device(), {.test_id = 658, .num_subblocks_r_dim = 1, .num_subblocks_c_dim = 8}));
}

// 8x1 tall column with deep K=4: 8 independent senders, deep accumulation
TEST_F(GenericMeshDeviceFixture, Test1DMatmulTall8RowDeepK) {
    EXPECT_TRUE(run_single_test(
        get_mesh_device(),
        {.test_id = 659, .num_subblocks_r_dim = 8, .num_subblocks_c_dim = 1, .num_subblocks_k_dim = 4}));
}

/* ======== NON-ORIGIN START TESTS ======== */

// 2x2 grid starting at logical core (2,2)
TEST_F(GenericMeshDeviceFixture, Test1DMatmulNonOriginStart) {
    EXPECT_TRUE(run_single_test(get_mesh_device(), {.test_id = 660, .start_logical_core = CoreCoord(2, 2)}));
}

// 5x3 non-square grid starting at (2,3) with K=2
TEST_F(GenericMeshDeviceFixture, Test1DMatmulNonOriginLargeNonSquare) {
    EXPECT_TRUE(run_single_test(
        get_mesh_device(),
        {.test_id = 661,
         .start_logical_core = CoreCoord(2, 3),
         .num_subblocks_r_dim = 3,
         .num_subblocks_c_dim = 5,
         .num_subblocks_k_dim = 2}));
}

/* ======== K DIMENSION TESTS ======== */

// K=2 on 2x2 grid
TEST_F(GenericMeshDeviceFixture, Test1DMatmulLargerK) {
    EXPECT_TRUE(run_single_test(get_mesh_device(), {.test_id = 662, .num_subblocks_k_dim = 2}));
}

// K=3 on 3x2 grid (R=3, C=2)
TEST_F(GenericMeshDeviceFixture, Test1DMatmulMultipleKSubblocks) {
    EXPECT_TRUE(run_single_test(
        get_mesh_device(),
        {.test_id = 663, .num_subblocks_r_dim = 3, .num_subblocks_c_dim = 2, .num_subblocks_k_dim = 3}));
}

/* ======== SUBBLOCK DIMENSION TESTS ======== */

// subblock_r=2 on 2x2 grid
TEST_F(GenericMeshDeviceFixture, Test1DMatmulLargerSubblockR) {
    EXPECT_TRUE(run_single_test(get_mesh_device(), {.test_id = 664, .subblock_r_dim = 2}));
}

// subblock_c=2 on 2x2 grid
TEST_F(GenericMeshDeviceFixture, Test1DMatmulLargerSubblockC) {
    EXPECT_TRUE(run_single_test(get_mesh_device(), {.test_id = 665, .subblock_c_dim = 2}));
}

// subblock_k=2 on 2x2 grid
TEST_F(GenericMeshDeviceFixture, Test1DMatmulLargerSubblockK) {
    EXPECT_TRUE(run_single_test(get_mesh_device(), {.test_id = 666, .subblock_k_dim = 2}));
}

// All subblocks=2 with K=2 on 2x2 grid
TEST_F(GenericMeshDeviceFixture, Test1DMatmulAllSubblocksLarger) {
    EXPECT_TRUE(run_single_test(
        get_mesh_device(),
        {.test_id = 667, .num_subblocks_k_dim = 2, .subblock_r_dim = 2, .subblock_c_dim = 2, .subblock_k_dim = 2}));
}

// All subblocks=4 with K=2 on 2x2 grid: maximum subblock dimensions
TEST_F(GenericMeshDeviceFixture, Test1DMatmulMaxSubblockDims) {
    EXPECT_TRUE(run_single_test(
        get_mesh_device(),
        {.test_id = 668, .num_subblocks_k_dim = 2, .subblock_r_dim = 4, .subblock_c_dim = 4, .subblock_k_dim = 4}));
}

// Asymmetric subblocks (R=2, C=6, K=3, sub_r=3, sub_c=2, sub_k=2)
TEST_F(GenericMeshDeviceFixture, Test1DMatmulAsymmetricSubblocks) {
    EXPECT_TRUE(run_single_test(
        get_mesh_device(),
        {.test_id = 669,
         .num_subblocks_r_dim = 2,
         .num_subblocks_c_dim = 6,
         .num_subblocks_k_dim = 3,
         .subblock_r_dim = 3,
         .subblock_c_dim = 2,
         .subblock_k_dim = 2}));
}

/* ======== STRESS TESTS ======== */

// 4x4 grid, K=4, all subblocks=2: stress all dimensions simultaneously
TEST_F(GenericMeshDeviceFixture, Test1DMatmulLargeGridDeepKLargeSubblocks) {
    EXPECT_TRUE(run_single_test(
        get_mesh_device(),
        {.test_id = 670,
         .num_subblocks_r_dim = 4,
         .num_subblocks_c_dim = 4,
         .num_subblocks_k_dim = 4,
         .subblock_r_dim = 2,
         .subblock_c_dim = 2,
         .subblock_k_dim = 2}));
}

// 1x6 wide row, K=3, sub_r=3, sub_c=2, sub_k=2: big multicast + big DRAM reads
TEST_F(GenericMeshDeviceFixture, Test1DMatmulWideMulticastLargePayload) {
    EXPECT_TRUE(run_single_test(
        get_mesh_device(),
        {.test_id = 671,
         .num_subblocks_r_dim = 1,
         .num_subblocks_c_dim = 6,
         .num_subblocks_k_dim = 3,
         .subblock_r_dim = 3,
         .subblock_c_dim = 2,
         .subblock_k_dim = 2}));
}

/* ======== DRAM BANK TESTS ======== */

// 2x2 grid reading in1 from DRAM bank 1
TEST_F(GenericMeshDeviceFixture, Test1DMatmulDramBank1) {
    EXPECT_TRUE(run_single_test(get_mesh_device(), {.test_id = 672, .dram_bank_id = 1}));
}

}  // namespace tt::tt_metal
