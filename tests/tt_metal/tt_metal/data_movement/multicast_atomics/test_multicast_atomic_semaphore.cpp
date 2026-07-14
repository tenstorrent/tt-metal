// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "../dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/semaphore_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

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
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const MulticastAtomicConfig& test_config) {
    IDevice* device = mesh_device->get_device(0);

    uint32_t num_senders = test_config.sender_cores.size();
    uint32_t num_dests = test_config.dst_grid_size.x * test_config.dst_grid_size.y;
    uint32_t expected_value = num_senders * test_config.num_of_transactions * test_config.atomic_inc_value;

    /* ================ SETUP ================ */

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

    // Sender core set (union of all sender cores).
    CoreRangeSet sender_core_range_set;
    for (const auto& sender_core : test_config.sender_cores) {
        sender_core_range_set = sender_core_range_set.merge(CoreRangeSet({CoreRange(sender_core)}));
    }
    CoreRangeSet all_cores = dst_core_range_set.merge(sender_core_range_set);

    using namespace tt::tt_metal::experimental;

    SemaphoreSpec atomic_sem{
        .unique_id = SemaphoreSpecName{"atomic_sem"},
        .target_nodes = all_cores,
    };

    KernelSpec::CompileTimeArgs sender_cta_bindings = {
        {"num_of_transactions", (uint32_t)test_config.num_of_transactions},
        {"atomic_inc_value", (uint32_t)test_config.atomic_inc_value},
        {"num_dests", (uint32_t)num_dests},
        {"test_id", (uint32_t)test_config.test_id}};

    DataMovementHardwareConfig sender_hw_config;
    if (device->arch() == tt::ARCH::QUASAR) {
        sender_hw_config = DataMovementGen2Config{};
    } else {
        sender_hw_config = DataMovementGen1Config{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = test_config.noc_id,
        };
    }
    KernelSpec sender_spec{
        .unique_id = KernelSpecName{"sender"},
        .source = "tests/tt_metal/tt_metal/data_movement/multicast_atomics/kernels/multicast_atomic_sender_2_0.cpp",
        .num_threads = 1,
        .semaphore_bindings = {KernelSpec::SemaphoreBinding{
            .semaphore_spec_name = atomic_sem.unique_id, .accessor_name = "sem_name"}},
        .compile_time_args = sender_cta_bindings,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"dst_start_x", "dst_start_y", "dst_end_x", "dst_end_y"},
            },
        .hw_config = sender_hw_config,
    };

    KernelSpec::CompileTimeArgs receiver_cta_bindings = {
        {"expected_value", (uint32_t)expected_value}, {"test_id", (uint32_t)test_config.test_id}};

    DataMovementHardwareConfig receiver_hw_config;
    if (device->arch() == tt::ARCH::QUASAR) {
        receiver_hw_config = DataMovementGen2Config{};
    } else {
        receiver_hw_config = DataMovementGen1Config{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = test_config.noc_id,
        };
    }
    KernelSpec receiver_spec{
        .unique_id = KernelSpecName{"receiver"},
        .source = "tests/tt_metal/tt_metal/data_movement/multicast_atomics/kernels/multicast_atomic_receiver_2_0.cpp",
        .num_threads = 1,
        .semaphore_bindings = {KernelSpec::SemaphoreBinding{
            .semaphore_spec_name = atomic_sem.unique_id, .accessor_name = "sem_name"}},
        .compile_time_args = receiver_cta_bindings,
        .hw_config = receiver_hw_config,
    };

    ProgramSpec spec{
        .name = "multicast_atomic_test",
        .kernels = {sender_spec, receiver_spec},
        .semaphores = {atomic_sem},
        .work_units =
            {
                WorkUnitSpec{
                    .name = "sender_wu",
                    .kernels = {sender_spec.unique_id},
                    .target_nodes = sender_core_range_set,
                },
                WorkUnitSpec{
                    .name = "receiver_wu",
                    .kernels = {receiver_spec.unique_id},
                    .target_nodes = dst_core_range_set,
                },
            },
    };

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunArgs run_params;
    ProgramRunArgs::KernelRunArgs sender_run_params{.kernel = sender_spec.unique_id};
    for (const auto& sender_core : test_config.sender_cores) {
        sender_run_params.runtime_arg_values.push_back(
            {.node = sender_core,
             .args = {
                 {"dst_start_x", (uint32_t)physical_dst_start.x},
                 {"dst_start_y", (uint32_t)physical_dst_start.y},
                 {"dst_end_x", (uint32_t)physical_dst_end.x},
                 {"dst_end_y", (uint32_t)physical_dst_end.y}}});
    }
    run_params.kernel_run_args.push_back(sender_run_params);

    ProgramRunArgs::KernelRunArgs receiver_run_params{.kernel = receiver_spec.unique_id};
    run_params.kernel_run_args.push_back(receiver_run_params);

    SetProgramRunArgs(program, run_params);

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
    uint32_t test_id = 342;

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
    uint32_t test_id = 343;

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
    uint32_t test_id = 344;

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
    uint32_t test_id = 345;

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
    uint32_t test_id = 346;

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
    uint32_t test_id = 347;

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
    uint32_t test_id = 348;

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
    };

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicLargerIncrement_2_0) {
    uint32_t test_id = 349;

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
    };

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicMultiSource_2_0) {
    uint32_t test_id = 350;

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
        .noc_id = NOC::NOC_0,
    };

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicSingleSourceNOC1_2_0) {
    uint32_t test_id = 351;

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
        .noc_id = NOC::NOC_1,
    };

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicMultiSourceNOC1_2_0) {
    uint32_t test_id = 352;

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
        .noc_id = NOC::NOC_1,
    };

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

TEST_F(GenericMeshDeviceFixture, MulticastAtomicLargerIncrementNOC1_2_0) {
    uint32_t test_id = 353;

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
        .num_of_transactions = 3,
        .atomic_inc_value = 5,
        .noc_id = NOC::NOC_1,
    };

    EXPECT_TRUE(unit_tests::dm::multicast_atomics::run_dm(mesh_device, config));
}

}  // namespace tt::tt_metal
