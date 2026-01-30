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
};

/// @brief Runs multicast atomic semaphore test
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test
/// @return true if test passes, false otherwise
bool run_multicast_atomic_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device, const MulticastAtomicConfig& test_config) {
    IDevice* device = mesh_device->get_device(0);

    /* ================ SETUP ================ */

    Program program = CreateProgram();

    // Calculate destination grid end coordinates
    CoreCoord dst_grid_end = {
        test_config.dst_grid_start.x + test_config.dst_grid_size.x - 1,
        test_config.dst_grid_start.y + test_config.dst_grid_size.y - 1};

    // Get physical coordinates for destination grid
    CoreCoord physical_dst_start = device->worker_core_from_logical_core(test_config.dst_grid_start);
    CoreCoord physical_dst_end = device->worker_core_from_logical_core(dst_grid_end);

    // Calculate number of destination cores
    uint32_t num_dests = test_config.dst_grid_size.x * test_config.dst_grid_size.y;

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

    // Get L1 info from first destination core
    L1AddressInfo l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, dst_cores[0]);
    uint32_t semaphore_addr = l1_info.base_address;

    // Expected value: each sender increments by atomic_inc_value * num_of_transactions
    // Total expected = num_senders * num_of_transactions * atomic_inc_value
    uint32_t num_senders = test_config.sender_cores.size();
    uint32_t expected_value = num_senders * test_config.num_of_transactions * test_config.atomic_inc_value;

    log_info(
        LogTest,
        "Running test with {} sender(s), {} destinations, expected final value: {}",
        num_senders,
        num_dests,
        expected_value);

    // Create sender kernels
    string sender_kernel_path =
        "tests/tt_metal/tt_metal/data_movement/multicast_atomics/kernels/multicast_atomic_sender.cpp";

    for (const auto& sender_core : test_config.sender_cores) {
        vector<uint32_t> sender_compile_args = {
            semaphore_addr,
            test_config.num_of_transactions,
            test_config.atomic_inc_value,
            num_dests,
            physical_dst_start.x,
            physical_dst_start.y,
            physical_dst_end.x,
            physical_dst_end.y};

        CreateKernel(
            program,
            sender_kernel_path,
            sender_core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = test_config.noc_id,
                .compile_args = sender_compile_args});

        log_info(LogTest, "Created sender kernel on core ({},{})", sender_core.x, sender_core.y);
    }

    // Create receiver kernels on all destination cores
    string receiver_kernel_path =
        "tests/tt_metal/tt_metal/data_movement/multicast_atomics/kernels/multicast_atomic_receiver.cpp";

    CoreRangeSet dst_core_range_set({CoreRange(test_config.dst_grid_start, dst_grid_end)});

    vector<uint32_t> receiver_compile_args = {semaphore_addr, expected_value};

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

    // Initialize semaphores to zero on all destination cores
    vector<uint32_t> zero_semaphore = {0};
    for (const auto& dst_core : dst_cores) {
        detail::WriteToDeviceL1(device, dst_core, semaphore_addr, zero_semaphore);
    }

    // Barrier to ensure initialization is complete
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    // Launch the program
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    /* ================ VERIFICATION ================ */

    bool all_passed = true;
    for (const auto& dst_core : dst_cores) {
        vector<uint32_t> semaphore_result;
        detail::ReadFromDeviceL1(device, dst_core, semaphore_addr, sizeof(uint32_t), semaphore_result);

        uint32_t final_value = semaphore_result[0];

        if (final_value != expected_value) {
            log_error(
                LogTest,
                "Core ({},{}) semaphore check failed. Expected: {}, Got: {}",
                dst_core.x,
                dst_core.y,
                expected_value,
                final_value);
            all_passed = false;
        } else {
            log_info(LogTest, "Core ({},{}) semaphore value correct: {}", dst_core.x, dst_core.y, final_value);
        }
    }

    return all_passed;
}

/// @brief Test single source multicast atomic increment to a grid
void single_source_multicast_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id, NOC noc_id = NOC::NOC_0) {
    MulticastAtomicConfig config = {
        .test_id = test_id,
        .sender_cores = {{4, 0}},
        .dst_grid_start = {0, 0},
        .dst_grid_size = {3, 4},  // 12 destination cores
        .num_of_transactions = 1,
        .atomic_inc_value = 1,
        .noc_id = noc_id};

    log_info(
        LogTest,
        "Single source multicast atomic test: 1 sender -> {} destinations",
        config.dst_grid_size.x * config.dst_grid_size.y);

    EXPECT_TRUE(run_multicast_atomic_test(mesh_device, config));
}

/// @brief Test multiple sources multicast atomic increment to same grid
void multi_source_multicast_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_id, NOC noc_id = NOC::NOC_0) {
    MulticastAtomicConfig config = {
        .test_id = test_id,
        .sender_cores = {{4, 0}, {4, 1}, {4, 2}, {4, 3}},  // 4 senders in column 4
        .dst_grid_start = {0, 0},
        .dst_grid_size = {3, 4},  // 12 destination cores
        .num_of_transactions = 1,
        .atomic_inc_value = 1,
        .noc_id = noc_id};

    log_info(
        LogTest,
        "Multi source multicast atomic test: {} senders -> {} destinations, expected value: {}",
        config.sender_cores.size(),
        config.dst_grid_size.x * config.dst_grid_size.y,
        config.sender_cores.size() * config.num_of_transactions * config.atomic_inc_value);

    EXPECT_TRUE(run_multicast_atomic_test(mesh_device, config));
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

    unit_tests::dm::multicast_atomics::single_source_multicast_test(mesh_device, test_id, NOC::NOC_0);
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicMultiSource) {
    uint32_t test_id = 322;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);

    auto grid_size = device->compute_with_storage_grid_size();
    if (grid_size.x < 5 || grid_size.y < 4) {
        GTEST_SKIP() << "Grid size too small for this test (need at least 5x4)";
    }

    unit_tests::dm::multicast_atomics::multi_source_multicast_test(mesh_device, test_id, NOC::NOC_0);
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicSingleSourceNOC1) {
    uint32_t test_id = 323;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);

    auto grid_size = device->compute_with_storage_grid_size();
    if (grid_size.x < 5 || grid_size.y < 4) {
        GTEST_SKIP() << "Grid size too small for this test (need at least 5x4)";
    }

    unit_tests::dm::multicast_atomics::single_source_multicast_test(mesh_device, test_id, NOC::NOC_1);
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicMultiSourceNOC1) {
    uint32_t test_id = 324;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_device(0);

    auto grid_size = device->compute_with_storage_grid_size();
    if (grid_size.x < 5 || grid_size.y < 4) {
        GTEST_SKIP() << "Grid size too small for this test (need at least 5x4)";
    }

    unit_tests::dm::multicast_atomics::multi_source_multicast_test(mesh_device, test_id, NOC::NOC_1);
}

}  // namespace tt::tt_metal
