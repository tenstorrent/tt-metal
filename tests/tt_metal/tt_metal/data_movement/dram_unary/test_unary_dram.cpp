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

namespace unit_tests::dm::dram {
// Test config, i.e. test parameters
struct DramConfig {
    uint32_t test_id = 0;
    uint32_t num_of_transactions = 0;
    uint32_t pages_per_transaction = 0;
    uint32_t bytes_per_page = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    CoreCoord core_coord = {0, 0};
    uint32_t dram_channel = 0;
    uint32_t virtual_channel = 0;
    bool use_2_0_api = false;  // Use Device 2.0 API
};

/// @brief Does Dram --> Reader --> L1 CB --> Writer --> Dram.
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @param fixture - DispatchFixture pointer for dispatch-aware operations
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const DramConfig& test_config) {
    IDevice* device = mesh_device->impl().get_device(0);
    // SETUP

    // Program
    Program program = CreateProgram();

    const size_t total_size_bytes = test_config.pages_per_transaction * test_config.bytes_per_page;

    // DRAM Address
    DramAddressInfo dram_info = unit_tests::dm::get_dram_address_and_size();

    uint32_t input_dram_address = dram_info.base_address;
    uint32_t output_dram_address = input_dram_address + total_size_bytes;

    // L1 Address
    L1AddressInfo l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.core_coord);

    uint32_t l1_address = l1_info.base_address;

    // Redundant but as an extra measure add a check to ensure both addresses are within DRAM bounds
    // Checks also needed for L1 maybe

    // Initialize semaphore ID
    CoreRangeSet core_range_set = CoreRangeSet({CoreRange(test_config.core_coord)});
    const uint32_t sem_id = CreateSemaphore(program, core_range_set, 0);

    // Compile-time arguments for kernels
    vector<uint32_t> reader_compile_args = {
        (uint32_t)test_config.test_id,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.pages_per_transaction,
        (uint32_t)test_config.bytes_per_page,
        (uint32_t)input_dram_address,
        (uint32_t)test_config.dram_channel,
        (uint32_t)l1_address,
        (uint32_t)sem_id};

    vector<uint32_t> writer_compile_args = {
        (uint32_t)test_config.test_id,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.pages_per_transaction,
        (uint32_t)test_config.bytes_per_page,
        (uint32_t)output_dram_address,
        (uint32_t)test_config.dram_channel,
        (uint32_t)l1_address,
        (uint32_t)sem_id,
        (uint32_t)test_config.virtual_channel};

    // Kernels
    std::string kernels_dir = "tests/tt_metal/tt_metal/data_movement/dram_unary/kernels/";
    std::string reader_kernel_filename = "reader_unary";
    std::string writer_kernel_filename = "writer_unary";
    if (test_config.use_2_0_api) {
        reader_kernel_filename += "_2_0";
        writer_kernel_filename += "_2_0";
    }
    std::string reader_kernel_path = kernels_dir + reader_kernel_filename + ".cpp";
    std::string writer_kernel_path = kernels_dir + writer_kernel_filename + ".cpp";

    CreateKernel(
        program,
        reader_kernel_path,
        test_config.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_args});

    CreateKernel(
        program,
        writer_kernel_path,
        test_config.core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_args});

    // Assign unique id
    log_info(LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // RUNNING THE PROGRAM

    // Setup Input and Golden Output
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, total_size_bytes / sizeof(bfloat16), chrono::system_clock::now().time_since_epoch().count());

    // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
    vector<uint32_t> packed_golden = packed_input;

    // Write Input to DRAM
    detail::WriteToDeviceDRAMChannel(device, test_config.dram_channel, input_dram_address, packed_input);
    MetalContext::instance().get_cluster().dram_barrier(device->id());

    // LAUNCH PROGRAM - Use mesh workload approach
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // Read Intermediate Output from L1 (for debugging purposes)
    vector<uint32_t> packed_intermediate_output;
    detail::ReadFromDeviceL1(device, test_config.core_coord, l1_address, total_size_bytes, packed_intermediate_output);

    // Read Output from DRAM
    vector<uint32_t> packed_output;
    detail::ReadFromDeviceDRAMChannel(
        device, test_config.dram_channel, output_dram_address, total_size_bytes, packed_output);

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

void directed_ideal_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    CoreCoord core_coord = {0, 0},
    uint32_t dram_channel = 0,
    uint32_t virtual_channel = 0) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    uint32_t num_of_transactions = 256;
    uint32_t pages_per_transaction = max_transmittable_pages;

    // Test config
    unit_tests::dm::dram::DramConfig test_config = {
        .test_id = test_case_id,
        .num_of_transactions = num_of_transactions,
        .pages_per_transaction = pages_per_transaction,
        .bytes_per_page = bytes_per_page,
        .l1_data_format = DataFormat::Float16_b,
        .core_coord = core_coord,
        .dram_channel = dram_channel,
        .virtual_channel = virtual_channel};

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

void packet_sizes_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    CoreCoord core_coord = {0, 0},
    uint32_t dram_channel = 0,
    bool use_2_0_api = false) {
    auto [bytes_per_page, max_reservable_bytes, max_reservable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_pages_per_transaction = 256;

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            if (num_of_transactions * pages_per_transaction * bytes_per_page >= max_reservable_bytes) {
                continue;
            }

            // Test config
            unit_tests::dm::dram::DramConfig test_config = {
                .test_id = test_case_id,
                .num_of_transactions = num_of_transactions,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
                .core_coord = core_coord,
                .dram_channel = dram_channel,
                .virtual_channel = 0,
                .use_2_0_api = use_2_0_api};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

}  // namespace unit_tests::dm::dram

/* ========== Test case for varying transaction numbers and sizes; Test id = 0 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMPacketSizes) {
    unit_tests::dm::dram::packet_sizes_test(
        get_mesh_device(),
        0,      // Test case ID
        {0, 0}  // Core coordinates (default)
    );
}

/* ========== Test case for varying core locations; Test id = 1 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMCoreLocations) {
    uint32_t test_case_id = 1;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    CoreCoord core_coord;
    uint32_t dram_channel = 0;

    // Cores
    auto grid_size = device->compute_with_storage_grid_size();
    log_info(LogTest, "Grid size x: {}, y: {}", grid_size.x, grid_size.y);

    for (unsigned int x = 0; x < grid_size.x; x++) {
        for (unsigned int y = 0; y < grid_size.y; y++) {
            core_coord = {x, y};

            unit_tests::dm::dram::directed_ideal_test(mesh_device, test_case_id, core_coord, dram_channel);
        }
    }
}

// DRAM channels

/* ========== Test case for varying DRAM channels; Test id = 2 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMChannels) {
    uint32_t test_case_id = 2;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    CoreCoord core_coord = {0, 0};

    for (unsigned int dram_channel = 0; dram_channel < device->num_dram_channels(); dram_channel++) {
        for (unsigned int vc = 0; vc < 4; vc++) {
            unit_tests::dm::dram::directed_ideal_test(mesh_device, test_case_id, core_coord, dram_channel, vc);
        }
    }
}

/* ========== Directed ideal test case; Test id = 3 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMDirectedIdeal) {
    // Test ID (Arbitrary)
    uint32_t test_id = 3;

    unit_tests::dm::dram::directed_ideal_test(get_mesh_device(), test_id);
}

/* ========== Test case for varying transaction numbers and sizes with 2.0 API; Test id = 40 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementDRAMPacketSizes2_0) {
    unit_tests::dm::dram::packet_sizes_test(
        get_mesh_device(),
        40,      // Test case ID
        {0, 0},  // Core coordinates (default)
        0,       // DRAM channel (default)
        true     // Use 2.0 API
    );
}

}  // namespace tt::tt_metal
