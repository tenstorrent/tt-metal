// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>
#include "multi_device_fixture.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::direct_write {

// Test config for direct write performance comparison
struct DirectWriteConfig {
    uint32_t test_id = 0;
    CoreCoord sender_core_coord;
    CoreCoord receiver_core_coord;
    uint32_t num_writes = 100;               // Number of direct writes to perform
    uint32_t write_value_base = 0x12340000;  // Base value for writes
    bool use_posted_writes = false;          // Posted vs non-posted writes
    bool same_destination = true;            // All writes to same address vs different addresses
    bool use_stateful_approach = true;       // Stateful vs non-stateful approach
    bool same_value = false;                 // Write same value each time vs different values (stateful only)
    uint32_t addr_stride = 4;                // Address increment for different destinations
    NOC noc_id = NOC::NOC_0;
};

/// @brief Run direct write test comparing stateful vs non-stateful approaches
/// @param mesh_device MeshDevice to run on
/// @param test_config Test configuration
/// @return Success status
bool run_dm(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device, const DirectWriteConfig& test_config) {
    // Get the actual device for this single-device test
    IDevice* device = mesh_device->impl().get_device(0);

    // Program
    Program program = CreateProgram();

    // (Logical) Core coordinates and ranges
    CoreRangeSet sender_core_set({CoreRange(test_config.sender_core_coord)});
    CoreRangeSet receiver_core_set({CoreRange(test_config.receiver_core_coord)});

    // Get L1 Address info
    L1AddressInfo sender_l1_info =
        tt::tt_metal::unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.sender_core_coord);
    L1AddressInfo receiver_l1_info =
        tt::tt_metal::unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.receiver_core_coord);

    // Validate L1 memory
    if (sender_l1_info.base_address != receiver_l1_info.base_address || sender_l1_info.size != receiver_l1_info.size) {
        log_error(tt::LogTest, "Mismatch in L1 address or size between sender and receiver cores");
        return false;
    }

    // Check if we have enough space for the test
    uint32_t required_bytes = test_config.same_destination ? 4 : (test_config.num_writes * test_config.addr_stride);
    if (receiver_l1_info.size < required_bytes) {
        log_error(tt::LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }

    uint32_t l1_base_address = sender_l1_info.base_address;

    // Physical Core Coordinates
    CoreCoord physical_receiver_core = device->worker_core_from_logical_core(test_config.receiver_core_coord);
    uint32_t packed_receiver_core_coordinates = physical_receiver_core.x << 16 | (physical_receiver_core.y & 0xFFFF);

    // Compile-time arguments for sender kernel
    vector<uint32_t> sender_compile_args = {
        test_config.test_id,
        test_config.num_writes,
        test_config.write_value_base,
        test_config.use_posted_writes ? 1u : 0u,
        test_config.same_destination ? 1u : 0u,
        test_config.same_value ? 1u : 0u,
        l1_base_address,
        test_config.addr_stride,
        packed_receiver_core_coordinates,
        static_cast<uint32_t>(test_config.noc_id)};

    // Choose kernel based on approach
    std::string kernels_dir = "tests/tt_metal/tt_metal/data_movement/direct_write/kernels/";
    std::string sender_kernel_filename =
        test_config.use_stateful_approach ? "sender_stateful.cpp" : "sender_non_stateful.cpp";
    std::string sender_kernel_path = kernels_dir + sender_kernel_filename;

    CreateKernel(
        program,
        sender_kernel_path,
        test_config.sender_core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = test_config.noc_id,
            .compile_args = sender_compile_args});

    // Assign unique id
    log_info(
        tt::LogTest,
        "Running Direct Write Test ID: {}, Approach: {}, Posted: {}, Same Destination: {}, Same Value: {}, Writes : {}",
        test_config.test_id,
        test_config.use_stateful_approach ? "Stateful" : "Non-Stateful",
        test_config.use_posted_writes ? "True" : "False",
        test_config.same_destination ? "True" : "False",
        test_config.same_value ? "True" : "False",
        test_config.num_writes);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Initialize receiver memory to known pattern
    uint32_t init_words = test_config.same_destination ? 1 : test_config.num_writes;
    std::vector<uint32_t> init_data(init_words, 0x00000000);  // Initialize to zero
    tt_metal::detail::WriteToDeviceL1(device, test_config.receiver_core_coord, l1_base_address, init_data);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    // Launch the program - Use mesh workload approach
    auto mesh_workload = distributed::MeshWorkload();
    auto target_devices =
        distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, 0));  // Single device at (0,0)
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // Read back and validate results
    std::vector<uint32_t> output_data;
    uint32_t read_bytes = init_words * sizeof(uint32_t);
    tt_metal::detail::ReadFromDeviceL1(
        device, test_config.receiver_core_coord, l1_base_address, read_bytes, output_data);

    // Validation
    bool pass = true;
    if (test_config.same_destination) {
        // All writes went to same location
        uint32_t expected_final_value;
        if (test_config.same_value) {
            // When writing same value, final value should be the base value
            expected_final_value = test_config.write_value_base;
        } else {
            // When writing different values, should have the last written value
            expected_final_value = test_config.write_value_base + test_config.num_writes - 1;
        }
        if (output_data[0] != expected_final_value) {
            log_error(
                tt::LogTest, "Expected final value: 0x{:08x}, Got: 0x{:08x}", expected_final_value, output_data[0]);
            pass = false;
        }
    } else {
        // Different destinations - check first few values

        uint32_t check_count =
            std::min({static_cast<uint32_t>(output_data.size()), test_config.num_writes, 10u});
        for (uint32_t i = 0; i < check_count; i++) {
            uint32_t expected_value;
            if (test_config.same_value) {
                // When writing same value, all locations should have base value
                expected_value = test_config.write_value_base;
            } else {
                // When writing different values, each location should have incremented value
                expected_value = test_config.write_value_base + i;
            }
            if (output_data[i] != expected_value) {
                log_error(
                    tt::LogTest, "Index {}: Expected: 0x{:08x}, Got: 0x{:08x}", i, expected_value, output_data[i]);
                pass = false;
            }
        }
    }

    if (!pass) {
        log_error(tt::LogTest, "Direct write test validation failed");
        log_info(tt::LogTest, "Output data (first 8 words):");
        for (size_t i = 0; i < std::min(output_data.size(), size_t(8)); i++) {
            log_info(tt::LogTest, "  [{}]: 0x{:08x}", i, output_data[i]);
        }
    }

    return pass;
}

void performance_comparison_test(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord sender_core = {0, 0},
    CoreCoord receiver_core = {1, 1}) {
    uint32_t max_transactions = 1024;                 // Show scaling advantage
    vector<bool> stateful_options = {false, true};    // Non-stateful vs stateful
    vector<bool> posted_options = {false, true};      // Non-posted vs posted

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 2) {
        for (bool posted : posted_options) {
            for (bool stateful : stateful_options) {
                DirectWriteConfig test_config = {
                    .test_id = test_id,
                    .sender_core_coord = sender_core,
                    .receiver_core_coord = receiver_core,
                    .num_writes = num_of_transactions,
                    .use_posted_writes = posted,
                    .same_destination = true,  // Same dest to show stateful advantage
                    .use_stateful_approach = stateful};

                EXPECT_TRUE(run_dm(mesh_device, test_config));
            }
        }
    }
}

void address_pattern_test(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord sender_core = {0, 0},
    CoreCoord receiver_core = {1, 1}) {
    uint32_t max_transactions = 1024;                   // Show scaling advantage
    vector<bool> destination_patterns = {true, false};  // Same vs different destinations
    vector<bool> stateful_options = {false, true};
    vector<bool> same_value_options = {false, true};  // Different values vs same value (stateful only)

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 2) {
        for (bool same_dest : destination_patterns) {
            for (bool stateful : stateful_options) {
                for (bool same_value : same_value_options) {
                    DirectWriteConfig test_config = {
                        .test_id = test_id,
                        .sender_core_coord = sender_core,
                        .receiver_core_coord = receiver_core,
                        .num_writes = num_of_transactions,
                        .same_destination = same_dest,
                        .use_stateful_approach = stateful,
                        .same_value = same_value};

                    EXPECT_TRUE(run_dm(mesh_device, test_config));
                }
            }
        }
    }
}

}  // namespace unit_tests::dm::direct_write

TEST_F(GenericMeshDeviceFixture, TensixDirectWritePerformanceComparison) {
    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    if (arch == tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Skipping test for Blackhole architecture";
    }

    uint32_t test_id = 500;
    unit_tests::dm::direct_write::performance_comparison_test(get_mesh_device(), test_id);
}

TEST_F(GenericMeshDeviceFixture, TensixDirectWriteAddressPatterns) {
    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    if (arch == tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Skipping test for Blackhole architecture";
    }

    uint32_t test_id = 501;
    unit_tests::dm::direct_write::address_pattern_test(get_mesh_device(), test_id);
}

}  // namespace tt::tt_metal
