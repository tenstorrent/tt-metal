// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "../dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace test_utils;

namespace unit_tests::dm::multicast_atomics {

struct MulticastAtomicConfig {
    uint32_t test_id = 0;
    vector<CoreCoord> sender_cores;
    CoreCoord dst_grid_start = {0, 0};
    CoreCoord dst_grid_size = {3, 4};
    uint32_t num_of_transactions = 1;
    uint32_t atomic_inc_value = 1;
    NOC noc_id = NOC::NOC_0;
    bool use_2_0_api = false;
};

/// @brief Runs multicast atomic semaphore test
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test
/// @return true if test passes, false otherwise
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const MulticastAtomicConfig& test_config) {
    IDevice* device = mesh_device->get_device(0);

    uint32_t num_senders = test_config.sender_cores.size();
    uint32_t num_dests = test_config.dst_grid_size.x * test_config.dst_grid_size.y;
    uint32_t expected_value = num_senders * test_config.num_of_transactions * test_config.atomic_inc_value;

    /* ================ SETUP ================ */

    Program program = CreateProgram();

    // Calculate destination grid end coordinates
    CoreCoord dst_grid_end = {
        test_config.dst_grid_start.x + test_config.dst_grid_size.x - 1,
        test_config.dst_grid_start.y + test_config.dst_grid_size.y - 1};

    // Get physical coordinates for destination grid
    CoreCoord physical_dst_start = device->worker_core_from_logical_core(test_config.dst_grid_start);
    CoreCoord physical_dst_end = device->worker_core_from_logical_core(dst_grid_end);

    vector<CoreCoord> dst_cores;
    for (uint32_t y = test_config.dst_grid_start.y; y <= dst_grid_end.y; y++) {
        for (uint32_t x = test_config.dst_grid_start.x; x <= dst_grid_end.x; x++) {
            dst_cores.push_back({x, y});
        }
    }

    // Validate that sender cores are not in the destination grid
    // The noc_semaphore_inc_multicast does not support loopback
    for (const auto& sender_core : test_config.sender_cores) {
        for (const auto& dst_core : dst_cores) {
            if (sender_core.x == dst_core.x && sender_core.y == dst_core.y) {
                log_error(
                    LogTest,
                    "Sender core ({},{}) is part of destination grid - this is not supported",
                    sender_core.x,
                    sender_core.y);
                return false;
            }
        }
    }

    CoreRangeSet dst_core_range_set({CoreRange(test_config.dst_grid_start, dst_grid_end)});

    // Create semaphore on all cores that will access it (senders and receivers)
    CoreRangeSet sender_core_range_set;
    for (const auto& sender_core : test_config.sender_cores) {
        sender_core_range_set = sender_core_range_set.merge(CoreRangeSet({CoreRange(sender_core)}));
    }
    CoreRangeSet all_cores = dst_core_range_set.merge(sender_core_range_set);
    uint32_t sem_id = CreateSemaphore(program, all_cores, 0);

    // Kernel paths
    std::string kernels_dir = "tests/tt_metal/tt_metal/data_movement/multicast_atomics/kernels/";
    std::string sender_kernel_filename = "multicast_atomic_sender";
    std::string receiver_kernel_filename = "multicast_atomic_receiver";
    if (test_config.use_2_0_api) {
        sender_kernel_filename += "_2_0";
        receiver_kernel_filename += "_2_0";
    }
    std::string sender_kernel_path = kernels_dir + sender_kernel_filename + ".cpp";
    std::string receiver_kernel_path = kernels_dir + receiver_kernel_filename + ".cpp";

    // Create sender kernels
    for (const auto& sender_core : test_config.sender_cores) {
        vector<uint32_t> sender_compile_args = {
            sem_id,
            test_config.num_of_transactions,
            test_config.atomic_inc_value,
            num_dests,
            physical_dst_start.x,
            physical_dst_start.y,
            physical_dst_end.x,
            physical_dst_end.y,
            test_config.test_id};

        CreateKernel(
            program,
            sender_kernel_path,
            sender_core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = test_config.noc_id,
                .compile_args = sender_compile_args});
    }

    // Create receiver kernels on all destination cores
    vector<uint32_t> receiver_compile_args = {sem_id, expected_value, test_config.test_id};

    CreateKernel(
        program,
        receiver_kernel_path,
        dst_core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = test_config.noc_id,
            .compile_args = receiver_compile_args});

    // Assign unique runtime ID
    log_info(LogTest, "Running Test ID: {}, Runtime ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    /* ================ RUNNING THE PROGRAM ================ */

    // Launch the program
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // Verification is implicit - if the program completes, the receivers
    // saw the expected semaphore value (otherwise they would hang waiting).
    // Note: We cannot verify that the value is not above expected since
    // CreateSemaphore manages the L1 address internally.
    log_info(
        LogTest, "Test completed successfully - all receivers received expected semaphore value: {}", expected_value);

    return true;
}

}  // namespace unit_tests::dm::multicast_atomics

/* ========== TEST CASES ========== */

TEST_F(GenericMeshDeviceFixture, MulticastAtomicSingleSource) {
    uint32_t test_id = 321;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);

    auto grid_size = device->compute_with_storage_grid_size();
    if (grid_size.x < 5 || grid_size.y < 4) {
        GTEST_SKIP() << "Grid size too small for this test (need at least 5x4)";
    }

    unit_tests::dm::multicast_atomics::MulticastAtomicConfig config = {
        .test_id = test_id,
        .sender_cores = {{4, 0}},
        .dst_grid_start = {0, 0},
        .dst_grid_size = {3, 4},
        .num_of_transactions = 1,
        .atomic_inc_value = 1,
        .noc_id = NOC::NOC_0};

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicMultiSource) {
    uint32_t test_id = 322;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);

    auto grid_size = device->compute_with_storage_grid_size();
    if (grid_size.x < 5 || grid_size.y < 4) {
        GTEST_SKIP() << "Grid size too small for this test (need at least 5x4)";
    }

    unit_tests::dm::multicast_atomics::MulticastAtomicConfig config = {
        .test_id = test_id,
        .sender_cores = {{4, 0}, {4, 1}, {4, 2}, {4, 3}},
        .dst_grid_start = {0, 0},
        .dst_grid_size = {3, 4},
        .num_of_transactions = 1,
        .atomic_inc_value = 1,
        .noc_id = NOC::NOC_0};

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicSingleSourceNOC1) {
    uint32_t test_id = 323;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);

    auto grid_size = device->compute_with_storage_grid_size();
    if (grid_size.x < 5 || grid_size.y < 4) {
        GTEST_SKIP() << "Grid size too small for this test (need at least 5x4)";
    }

    unit_tests::dm::multicast_atomics::MulticastAtomicConfig config = {
        .test_id = test_id,
        .sender_cores = {{4, 0}},
        .dst_grid_start = {0, 0},
        .dst_grid_size = {3, 4},
        .num_of_transactions = 1,
        .atomic_inc_value = 1,
        .noc_id = NOC::NOC_1};

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicMultiSourceNOC1) {
    uint32_t test_id = 324;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);

    auto grid_size = device->compute_with_storage_grid_size();
    if (grid_size.x < 5 || grid_size.y < 4) {
        GTEST_SKIP() << "Grid size too small for this test (need at least 5x4)";
    }

    unit_tests::dm::multicast_atomics::MulticastAtomicConfig config = {
        .test_id = test_id,
        .sender_cores = {{4, 0}, {4, 1}, {4, 2}, {4, 3}},
        .dst_grid_start = {0, 0},
        .dst_grid_size = {3, 4},
        .num_of_transactions = 1,
        .atomic_inc_value = 1,
        .noc_id = NOC::NOC_1};

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicLargerIncrement) {
    uint32_t test_id = 325;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);

    auto grid_size = device->compute_with_storage_grid_size();
    if (grid_size.x < 5 || grid_size.y < 4) {
        GTEST_SKIP() << "Grid size too small for this test (need at least 5x4)";
    }

    // Test with increment value of 5 and multiple transactions
    // 4 senders * 3 transactions * 5 increment = 60 expected final value
    unit_tests::dm::multicast_atomics::MulticastAtomicConfig config = {
        .test_id = test_id,
        .sender_cores = {{4, 0}, {4, 1}, {4, 2}, {4, 3}},
        .dst_grid_start = {0, 0},
        .dst_grid_size = {3, 4},
        .num_of_transactions = 3,
        .atomic_inc_value = 5,
        .noc_id = NOC::NOC_0};

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicLargerIncrementNOC1) {
    uint32_t test_id = 326;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);

    auto grid_size = device->compute_with_storage_grid_size();
    if (grid_size.x < 5 || grid_size.y < 4) {
        GTEST_SKIP() << "Grid size too small for this test (need at least 5x4)";
    }

    // Test with increment value of 5 and multiple transactions on NOC1
    // 4 senders * 3 transactions * 5 increment = 60 expected final value
    unit_tests::dm::multicast_atomics::MulticastAtomicConfig config = {
        .test_id = test_id,
        .sender_cores = {{4, 0}, {4, 1}, {4, 2}, {4, 3}},
        .dst_grid_start = {0, 0},
        .dst_grid_size = {3, 4},
        .num_of_transactions = 3,
        .atomic_inc_value = 5,
        .noc_id = NOC::NOC_1};

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

/* ========== NOC 2.0 API TEST CASES ========== */

TEST_F(GenericMeshDeviceFixture, MulticastAtomicSingleSource_2_0) {
    uint32_t test_id = 327;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);

    auto grid_size = device->compute_with_storage_grid_size();
    if (grid_size.x < 5 || grid_size.y < 4) {
        GTEST_SKIP() << "Grid size too small for this test (need at least 5x4)";
    }

    unit_tests::dm::multicast_atomics::MulticastAtomicConfig config = {
        .test_id = test_id,
        .sender_cores = {{4, 0}},
        .dst_grid_start = {0, 0},
        .dst_grid_size = {3, 4},
        .num_of_transactions = 1,
        .atomic_inc_value = 1,
        .noc_id = NOC::NOC_0,
        .use_2_0_api = true};

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicLargerIncrement_2_0) {
    uint32_t test_id = 328;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);

    auto grid_size = device->compute_with_storage_grid_size();
    if (grid_size.x < 5 || grid_size.y < 4) {
        GTEST_SKIP() << "Grid size too small for this test (need at least 5x4)";
    }

    // Test with increment value of 5 and multiple transactions
    // 4 senders * 3 transactions * 5 increment = 60 expected final value
    unit_tests::dm::multicast_atomics::MulticastAtomicConfig config = {
        .test_id = test_id,
        .sender_cores = {{4, 0}, {4, 1}, {4, 2}, {4, 3}},
        .dst_grid_start = {0, 0},
        .dst_grid_size = {3, 4},
        .num_of_transactions = 3,
        .atomic_inc_value = 5,
        .noc_id = NOC::NOC_0,
        .use_2_0_api = true};

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

}  // namespace tt::tt_metal
