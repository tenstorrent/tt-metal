// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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

namespace unit_tests::dm::atomics {

// Test configuration for atomic semaphore bandwidth tests
struct AtomicSemaphoreConfig {
    uint32_t test_id = 0;
    CoreCoord sender_core_coord = {0, 0};
    CoreCoord receiver_core_coord = {0, 1};
    uint32_t num_of_transactions = 256;
    uint32_t semaphore_addr_offset = 4096;  // Place semaphore 4KB from base
    uint32_t atomic_inc_value = 1;
    NOC noc_id = NOC::NOC_0;
};

/// @brief Runs atomic semaphore bandwidth test with sender and receiver cores
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test
/// @return true if test passes, false otherwise
bool run_atomic_semaphore_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device, const AtomicSemaphoreConfig& test_config) {
    // Get the actual device for this single-device test
    IDevice* device = mesh_device->get_device(0);

    std::cerr << "Sender core location X,Y: " << test_config.sender_core_coord.x << ","
              << test_config.sender_core_coord.y << std::endl;
    std::cerr << "Receiver core location X,Y: " << test_config.receiver_core_coord.x << ","
              << test_config.receiver_core_coord.y << std::endl;

    /* ================ SETUP ================ */

    // Program
    Program program = CreateProgram();

    // Buffer Parameters
    const size_t total_data_size = test_config.semaphore_addr_offset + sizeof(uint32_t);

    // Obtain L1 Address for Storing Data
    L1AddressInfo sender_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.sender_core_coord);
    L1AddressInfo receiver_l1_info =
        unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.receiver_core_coord);

    // Check if cores have compatible L1 configurations
    if (sender_l1_info.base_address != receiver_l1_info.base_address || sender_l1_info.size != receiver_l1_info.size) {
        log_error(LogTest, "Mismatch in L1 address or size between sender and receiver cores");
        return false;
    }

    // Check if L1 size is sufficient
    if (sender_l1_info.size < total_data_size) {
        log_error(
            LogTest,
            "Insufficient L1 size for the test configuration. Need: {}, Have: {}",
            total_data_size,
            sender_l1_info.size);
        return false;
    }

    uint32_t l1_base_address = sender_l1_info.base_address;

    // Physical Core Coordinates
    CoreCoord physical_receiver_core = device->worker_core_from_logical_core(test_config.receiver_core_coord);
    uint32_t packed_receiver_core_coordinates = physical_receiver_core.x << 16 | (physical_receiver_core.y & 0xFFFF);

    // Compile-time arguments for sender kernel
    vector<uint32_t> sender_compile_args = {
        (uint32_t)l1_base_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.atomic_inc_value,
        (uint32_t)test_config.test_id,
        (uint32_t)packed_receiver_core_coordinates,
        (uint32_t)test_config.semaphore_addr_offset};

    // Compile-time arguments for receiver kernel
    vector<uint32_t> receiver_compile_args = {
        (uint32_t)l1_base_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.atomic_inc_value,
        (uint32_t)test_config.test_id,
        (uint32_t)test_config.semaphore_addr_offset};

    // Kernels
    std::string kernels_dir = "tests/tt_metal/tt_metal/data_movement/atomics/kernels/";
    std::string sender_kernel_path = kernels_dir + "atomic_semaphore_sender.cpp";
    std::string receiver_kernel_path = kernels_dir + "atomic_semaphore_receiver.cpp";

    CreateKernel(
        program,
        receiver_kernel_path,
        test_config.receiver_core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = test_config.noc_id,
            .compile_args = receiver_compile_args});

    CreateKernel(
        program,
        sender_kernel_path,
        test_config.sender_core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = test_config.noc_id,
            .compile_args = sender_compile_args});

    // Assign unique runtime ID
    log_info(
        LogTest,
        "Running Atomic Semaphore Test ID: {}, Runtime ID: {}",
        test_config.test_id,
        unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    /* ================ RUNNING THE PROGRAM ================ */

    // Initialize semaphores to zero on both cores
    vector<uint32_t> zero_semaphore = {0};
    detail::WriteToDeviceL1(
        device, test_config.sender_core_coord, l1_base_address + test_config.semaphore_addr_offset, zero_semaphore);
    detail::WriteToDeviceL1(
        device, test_config.receiver_core_coord, l1_base_address + test_config.semaphore_addr_offset, zero_semaphore);

    // Barrier to ensure initialization is complete
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    // Launch the program using mesh workload approach
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    /* ================ VERIFICATION ================ */

    // Read final semaphore values for verification
    vector<uint32_t> sender_semaphore_result, receiver_semaphore_result;
    detail::ReadFromDeviceL1(
        device,
        test_config.sender_core_coord,
        l1_base_address + test_config.semaphore_addr_offset,
        sizeof(uint32_t),
        sender_semaphore_result);
    detail::ReadFromDeviceL1(
        device,
        test_config.receiver_core_coord,
        l1_base_address + test_config.semaphore_addr_offset,
        sizeof(uint32_t),
        receiver_semaphore_result);

    uint32_t sender_final_semaphore = sender_semaphore_result[0];
    uint32_t receiver_final_semaphore = receiver_semaphore_result[0];
    uint32_t expected_receiver_semaphore = test_config.num_of_transactions * test_config.atomic_inc_value;

    log_info(LogTest, "Sender final semaphore: {}", sender_final_semaphore);
    log_info(LogTest, "Receiver final semaphore: {}", receiver_final_semaphore);
    log_info(LogTest, "Expected receiver semaphore: {}", expected_receiver_semaphore);

    // Verify that receiver semaphore has the expected value
    bool semaphore_check = (receiver_final_semaphore == expected_receiver_semaphore);

    if (!semaphore_check) {
        log_error(
            LogTest,
            "Receiver semaphore check failed. Expected: {}, Got: {}",
            expected_receiver_semaphore,
            receiver_final_semaphore);
    }

    return semaphore_check;
}

/// @brief Bandwidth sweep test that varies transaction sizes and counts
void increment_value_sweep_test(
    const shared_ptr<distributed::MeshDevice> mesh_device,
    uint32_t test_id,
    CoreCoord sender_core = {0, 1},
    CoreCoord receiver_core = {0, 0}) {
    // Test parameters for bandwidth sweep
    uint32_t max_transactions = 64;
    uint32_t max_increment_value = 4;

    // Sweep through different NOC configurations
    for (NOC noc_id : {NOC::NOC_0, NOC::NOC_1}) {
        // Sweep through different transaction counts
        for (uint32_t num_transactions = 16; num_transactions <= max_transactions; num_transactions *= 2) {
            // Sweep through different transaction sizes
            for (uint32_t atomic_inc_value = 1; atomic_inc_value <= max_increment_value; atomic_inc_value++) {
                AtomicSemaphoreConfig config = {
                    .test_id = test_id,
                    .sender_core_coord = sender_core,
                    .receiver_core_coord = receiver_core,
                    .num_of_transactions = num_transactions,
                    .semaphore_addr_offset = 4096,
                    .atomic_inc_value = atomic_inc_value,
                    .noc_id = noc_id};

                // Passing in atomic_inc_value twice: Once for "Pages" and once for "AtomicInc" to maintain consistency
                log_info(
                    LogTest,
                    "Testing: NOC={}, Transactions={}, Pages={}, AtomicInc={}",
                    (noc_id == NOC::NOC_0) ? "NOC_0" : "NOC_1",
                    num_transactions,
                    atomic_inc_value,
                    atomic_inc_value);

                EXPECT_TRUE(run_atomic_semaphore_test(mesh_device, config));
            }
        }
    }
}

/// @brief Directed performance test with optimal parameters
void directed_performance_test(
    shared_ptr<distributed::MeshDevice> mesh_device,
    uint32_t test_id,
    uint32_t atomic_inc_value = 1,
    CoreCoord sender_core = {0, 0},
    CoreCoord receiver_core = {0, 1}) {
    // Optimal parameters for performance
    uint32_t num_transactions = 512;

    AtomicSemaphoreConfig config = {
        .test_id = test_id,
        .sender_core_coord = sender_core,
        .receiver_core_coord = receiver_core,
        .num_of_transactions = num_transactions,
        .semaphore_addr_offset = 4096,
        .atomic_inc_value = atomic_inc_value,
        .noc_id = NOC::NOC_0};

    log_info(
        LogTest,
        "Running directed performance test with {} transactions of {} increment value each",
        num_transactions,
        atomic_inc_value);

    EXPECT_TRUE(run_atomic_semaphore_test(mesh_device, config));
}

}  // namespace unit_tests::dm::atomics

/* ========== TEST CASES ========== */

TEST_F(GenericMeshDeviceFixture, AtomicSemaphoreAdjacentIncrementValueSweep) {
    uint32_t test_id = 319;

    unit_tests::dm::atomics::increment_value_sweep_test(
        get_mesh_device(),
        test_id,
        CoreCoord(0, 1),  // Sender core
        CoreCoord(0, 0)   // Receiver core
    );
}

TEST_F(GenericMeshDeviceFixture, AtomicSemaphoreNonAdjacentIncrementValueSweep) {
    uint32_t test_id = 320;

    auto logical_grid_size = get_mesh_device()->logical_grid_size();

    TT_FATAL(logical_grid_size.x > 2 && logical_grid_size.y > 2, "Test assumes grid size is at least 3x3");

    // Sender core is pulled in 1 row & 1 column from opposite corner to avoid dispatch cores
    auto sender_core = CoreCoord(logical_grid_size.x - 2, logical_grid_size.y - 2);

    unit_tests::dm::atomics::increment_value_sweep_test(
        get_mesh_device(),
        test_id,
        sender_core,     // Sender core
        CoreCoord(0, 0)  // Receiver core
    );
}

}  // namespace tt::tt_metal
