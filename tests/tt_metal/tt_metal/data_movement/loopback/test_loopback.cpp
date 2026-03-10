// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace test_utils;

namespace unit_tests::dm::core_loopback {

constexpr uint32_t START_ID = 16;

// Test config, i.e. test parameters
struct LoopbackConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord = {0, 0};
    uint32_t num_of_transactions = 0;
    uint32_t transaction_size_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    NOC noc_id = NOC::NOC_0;

    // TODO: Add the following parameters
    //  1. Virtual Channel (only useful for unicast)
    //  2. Posted flag (posted multicast has much better performance at larger grid sizes, than non-posted due to
    //  response packets) (60, 45, 23, vs 60, 60, 60 at posted)
};

/// @brief Does L1 Sender Core --> L1 Receiver Cores
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @param fixture - DispatchFixture pointer for dispatch-aware operations
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const LoopbackConfig& test_config) {
    IDevice* device = mesh_device->impl().get_device(0);
    // Program
    Program program = CreateProgram();

    // Buffer Parameters
    const uint32_t transaction_size_bytes = test_config.transaction_size_pages * test_config.page_size_bytes;

    // (Logical) Core coordinates and ranges
    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    CoreRangeSet subordinate_core_set({CoreRange(test_config.master_core_coord)});

    // Obtain L1 Address for Storing Data
    L1AddressInfo master_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.master_core_coord);

    // Check if the L1 size is sufficient for the test configuration
    if (master_l1_info.size < transaction_size_bytes * 2) {
        log_error(LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }

    // Assign a "safe" L1 local address for the master core
    uint32_t master_l1_byte_address = master_l1_info.base_address;
    uint32_t subordinate_l1_byte_address =
        master_l1_info.base_address + transaction_size_bytes;  // Offset for subordinate data

    // Compile-time arguments for kernels
    vector<uint32_t> sender_compile_args = {
        (uint32_t)master_l1_byte_address,
        (uint32_t)subordinate_l1_byte_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id};

    // Kernels
    auto sender_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/loopback/kernels/sender.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = test_config.noc_id,
            .compile_args = sender_compile_args});

    // Semaphores
    CoreRangeSet sem_core_set = subordinate_core_set.merge<CoreRangeSet>(master_core_set);
    const uint32_t sem_id = CreateSemaphore(program, sem_core_set, 0);

    // Runtime Arguments
    CoreCoord worker = device->worker_core_from_logical_core(test_config.master_core_coord);
    vector<uint32_t> master_run_args = {sem_id, worker.x, worker.y};
    SetRuntimeArgs(program, sender_kernel, master_core_set, master_run_args);

    // Assign unique id
    log_info(LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f,
        100.0f,
        transaction_size_bytes / sizeof(bfloat16),
        chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    vector<uint32_t> packed_golden = packed_input;

    // Write Input to Master L1
    detail::WriteToDeviceL1(device, test_config.master_core_coord, master_l1_byte_address, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    // Launch program and record outputs
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // Record Output from Subordinate L1 (same core, different address)
    vector<uint32_t> packed_output;
    detail::ReadFromDeviceL1(
        device, test_config.master_core_coord, subordinate_l1_byte_address, transaction_size_bytes, packed_output);

    // Results comparison
    bool is_equal = (packed_output == packed_golden);
    if (!is_equal) {
        log_error(LogTest, "Equality Check failed");
        log_info(LogTest, "Golden vector");
        print_vector<uint32_t>(packed_golden);
        log_info(LogTest, "Output vector");
        print_vector<uint32_t>(packed_output);
    }
    return is_equal;
}
}  // namespace unit_tests::dm::core_loopback

/* ========== Test case for loopback data movement; ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementLoopbackPacketSizes) {
    auto mesh_device = get_mesh_device();
    auto arch_ = mesh_device->impl().get_device(0)->arch();

    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages =
        arch_ == ARCH::BLACKHOLE ? 1024 : 2048;                     // Max total transaction size == 64 KB
    uint32_t page_size_bytes = arch_ == ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    NOC noc_id = NOC::NOC_0;

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            // Test config
            unit_tests::dm::core_loopback::LoopbackConfig test_config = {
                .test_id = unit_tests::dm::core_loopback::START_ID + 0,
                .master_core_coord = master_core_coord,
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .noc_id = noc_id,
            };

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementLoopbackDirectedIdeal) {
    auto mesh_device = get_mesh_device();

    uint32_t test_id = 55;

    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    uint32_t num_of_transactions = 128;
    uint32_t transaction_size_pages =
        max_transmittable_pages / (num_of_transactions * 2);  // Since we need to fit 2 buffers, we divide by 2

    CoreCoord master_core_coord = {0, 0};
    NOC noc_id = NOC::NOC_0;

    unit_tests::dm::core_loopback::LoopbackConfig test_config = {
        .test_id = test_id,
        .master_core_coord = master_core_coord,
        .num_of_transactions = num_of_transactions,
        .transaction_size_pages = transaction_size_pages,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
        .noc_id = noc_id};

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

}  // namespace tt::tt_metal
