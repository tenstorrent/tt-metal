// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "dm_common.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::direct_write {

// Test config for direct write performance comparison
struct DirectWriteConfig {
    uint32_t test_id = 0;
    CoreCoord sender_core_coord = CoreCoord();
    CoreCoord receiver_core_coord = CoreCoord();
    uint32_t num_writes = 100;               // Number of direct writes to perform
    uint32_t write_value_base = 0x12340000;  // Base value for writes
    bool use_posted_writes = false;          // Posted vs non-posted writes
    bool same_destination = true;            // All writes to same address vs different addresses
    bool use_stateful_approach = true;       // Stateful vs non-stateful approach
    uint32_t dest_l1_addr_offset = 0x20000;  // L1 address offset on receiver
    uint32_t addr_stride = 4;                // Address increment for different destinations
    NOC noc_id = NOC::NOC_0;
};

/// @brief Run direct write test comparing stateful vs non-stateful approaches
/// @param device Device to run on
/// @param test_config Test configuration
/// @return Success status
bool run_dm(IDevice* device, const DirectWriteConfig& test_config) {
    // Program
    Program program = CreateProgram();

    // (Logical) Core coordinates and ranges
    CoreRangeSet sender_core_set({CoreRange(test_config.sender_core_coord)});
    CoreRangeSet receiver_core_set({CoreRange(test_config.receiver_core_coord)});
    CoreRangeSet combined_core_set = sender_core_set.merge<CoreRangeSet>(receiver_core_set);

    // Get L1 Address info
    L1AddressInfo sender_l1_info =
        tt::tt_metal::unit_tests::dm::get_l1_address_and_size(device, test_config.sender_core_coord);
    L1AddressInfo receiver_l1_info =
        tt::tt_metal::unit_tests::dm::get_l1_address_and_size(device, test_config.receiver_core_coord);

    // Validate L1 memory
    if (sender_l1_info.base_address != receiver_l1_info.base_address || sender_l1_info.size != receiver_l1_info.size) {
        log_error(tt::LogTest, "Mismatch in L1 address or size between sender and receiver cores");
        return false;
    }

    // Check if we have enough space for the test
    uint32_t required_bytes = test_config.same_destination ? 4 : (test_config.num_writes * test_config.addr_stride);
    if (receiver_l1_info.size < test_config.dest_l1_addr_offset + required_bytes) {
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
        l1_base_address + test_config.dest_l1_addr_offset,  // Destination L1 address
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
        "Running Direct Write Test ID: {}, Approach: {}, Writes: {}",
        test_config.test_id,
        test_config.use_stateful_approach ? "Stateful" : "Non-Stateful",
        test_config.num_writes);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Initialize receiver memory to known pattern
    uint32_t init_words = test_config.same_destination ? 1 : test_config.num_writes;
    std::vector<uint32_t> init_data(init_words, 0x00000000);  // Initialize to zero
    tt_metal::detail::WriteToDeviceL1(
        device, test_config.receiver_core_coord, l1_base_address + test_config.dest_l1_addr_offset, init_data);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    // Launch the program
    detail::LaunchProgram(device, program);

    // Read back and validate results
    std::vector<uint32_t> output_data;
    uint32_t read_bytes = init_words * sizeof(uint32_t);
    tt_metal::detail::ReadFromDeviceL1(
        device,
        test_config.receiver_core_coord,
        l1_base_address + test_config.dest_l1_addr_offset,
        read_bytes,
        output_data);

    // Validation
    bool pass = true;
    if (test_config.same_destination) {
        // All writes went to same location - should have the last written value
        uint32_t expected_final_value = test_config.write_value_base + test_config.num_writes - 1;
        if (output_data[0] != expected_final_value) {
            log_error(
                tt::LogTest, "Expected final value: 0x{:08x}, Got: 0x{:08x}", expected_final_value, output_data[0]);
            pass = false;
        }
    } else {
        // Different destinations - check first few values
        uint32_t check_count =
            std::min(static_cast<uint32_t>(output_data.size()), std::min(test_config.num_writes, 10u));
        for (uint32_t i = 0; i < check_count; i++) {
            uint32_t expected_value = test_config.write_value_base + i;
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
    ARCH arch_,
    vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_id,
    CoreCoord sender_core = {0, 0},
    CoreCoord receiver_core = {0, 1}) {
    vector<uint32_t> write_counts = {10, 100, 1000};  // Show scaling advantage
    vector<bool> stateful_options = {false, true};    // Non-stateful vs stateful
    vector<bool> posted_options = {false, true};      // Non-posted vs posted

    for (uint32_t num_writes : write_counts) {
        for (bool posted : posted_options) {
            for (bool stateful : stateful_options) {
                DirectWriteConfig test_config = {
                    .test_id = test_id,
                    .sender_core_coord = sender_core,
                    .receiver_core_coord = receiver_core,
                    .num_writes = num_writes,
                    .use_posted_writes = posted,
                    .same_destination = true,  // Same dest to show stateful advantage
                    .use_stateful_approach = stateful};

                for (unsigned int id = 0; id < num_devices_; id++) {
                    EXPECT_TRUE(run_dm(devices_.at(id), test_config));
                }
            }
        }
    }
}

void address_pattern_test(
    ARCH arch_,
    vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_id,
    CoreCoord sender_core = {0, 0},
    CoreCoord receiver_core = {1, 0}) {
    vector<bool> destination_patterns = {true, false};  // Same vs different destinations
    vector<bool> stateful_options = {false, true};

    for (bool same_dest : destination_patterns) {
        for (bool stateful : stateful_options) {
            // Skip non-stateful with different destinations for performance reasons
            if (!stateful && !same_dest) {
                continue;
            }

            DirectWriteConfig test_config = {
                .test_id = test_id,
                .sender_core_coord = sender_core,
                .receiver_core_coord = receiver_core,
                .num_writes = 50,
                .same_destination = same_dest,
                .use_stateful_approach = stateful};

            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

void noc_comparison_test(
    ARCH arch_,
    vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_id,
    CoreCoord sender_core = {1, 1},
    CoreCoord receiver_core = {2, 2}) {
    vector<NOC> nocs = {NOC::NOC_0, NOC::NOC_1};

    for (NOC noc_id : nocs) {
        DirectWriteConfig test_config = {
            .test_id = test_id,
            .sender_core_coord = sender_core,
            .receiver_core_coord = receiver_core,
            .num_writes = 200,
            .use_posted_writes = true,
            .use_stateful_approach = true,  // Use best performing approach
            .noc_id = noc_id};

        for (unsigned int id = 0; id < num_devices_; id++) {
            EXPECT_TRUE(run_dm(devices_.at(id), test_config));
        }
    }
}

void correctness_test(
    ARCH arch_,
    vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_id,
    CoreCoord sender_core = {0, 0},
    CoreCoord receiver_core = {0, 1}) {
    // Simple correctness test with known pattern
    DirectWriteConfig test_config = {
        .test_id = test_id,
        .sender_core_coord = sender_core,
        .receiver_core_coord = receiver_core,
        .num_writes = 20,
        .write_value_base = 0xDEADBEE0,  // Easy to recognize pattern
        .same_destination = false,       // Test different addresses
        .use_stateful_approach = true};

    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

}  // namespace unit_tests::dm::direct_write

/* ========== TEST CASES ========== */

TEST_F(DeviceFixture, TensixDirectWritePerformanceComparison) {
    uint32_t test_id = 500;
    unit_tests::dm::direct_write::performance_comparison_test(arch_, devices_, num_devices_, test_id);
}

TEST_F(DeviceFixture, TensixDirectWriteAddressPatternsPacketSizes) {
    uint32_t test_id = 501;
    unit_tests::dm::direct_write::address_pattern_test(arch_, devices_, num_devices_, test_id);
}

TEST_F(DeviceFixture, TensixDirectWriteNOCComparisonPacketSizes) {
    uint32_t test_id = 502;
    unit_tests::dm::direct_write::noc_comparison_test(arch_, devices_, num_devices_, test_id);
}

TEST_F(DeviceFixture, TensixDirectWriteCorrectness) {
    uint32_t test_id = 503;
    unit_tests::dm::direct_write::correctness_test(arch_, devices_, num_devices_, test_id);
}

}  // namespace tt::tt_metal
