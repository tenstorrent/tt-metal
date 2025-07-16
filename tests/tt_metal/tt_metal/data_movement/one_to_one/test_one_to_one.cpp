// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::core_to_core {

// Test config, i.e. test parameters
struct OneToOneConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord = CoreCoord();
    CoreCoord subordinate_core_coord = CoreCoord();
    uint32_t num_of_transactions = 0;
    uint32_t pages_per_transaction = 0;
    uint32_t bytes_per_page = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    uint32_t virtual_channel = 0;  // Virtual channel for the NOC

    // TODO: Add the following parameters
    //  1. Which NOC to use
    //  2. Posted flag
};

/// @brief Does L1 Sender Core --> L1 Receiver Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const OneToOneConfig& test_config) {
    /* ================ SETUP ================ */

    // Program
    Program program = CreateProgram();

    // Buffer Parameters
    const size_t bytes_per_transaction = test_config.pages_per_transaction * test_config.bytes_per_page;
    const size_t total_size_bytes = bytes_per_transaction * test_config.num_of_transactions;

    // (Logical) Core coordinates and ranges
    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    CoreRangeSet subordinate_core_set({CoreRange(test_config.subordinate_core_coord)});
    CoreRangeSet combined_core_set = master_core_set.merge<CoreRangeSet>(subordinate_core_set);

    // Obtain L1 Address for Storing Data
    // NOTE: We don't know if the whole block of memory is actually available.
    //       This is something that could probably be checked
    L1AddressInfo master_l1_info =
        tt::tt_metal::unit_tests::dm::get_l1_address_and_size(device, test_config.master_core_coord);
    L1AddressInfo subordinate_l1_info =
        tt::tt_metal::unit_tests::dm::get_l1_address_and_size(device, test_config.subordinate_core_coord);
    // Checks that both master and subordinate cores have the same L1 base address and size
    if (master_l1_info.base_address != subordinate_l1_info.base_address ||
        master_l1_info.size != subordinate_l1_info.size) {
        log_error(tt::LogTest, "Mismatch in L1 address or size between master and subordinate cores");
        return false;
    }
    // Check if the L1 size is sufficient for the test configuration
    if (master_l1_info.size < bytes_per_transaction) {
        log_error(tt::LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }
    // Assigns a "safe" L1 local address for the master and subordinate cores
    uint32_t l1_base_address = master_l1_info.base_address;

    // Semaphores
    const uint32_t sem_id = CreateSemaphore(program, combined_core_set, 0);

    // Physical Core Coordinates
    CoreCoord physical_subordinate_core = device->worker_core_from_logical_core(test_config.subordinate_core_coord);
    uint32_t packed_subordinate_core_coordinates =
        physical_subordinate_core.x << 16 | (physical_subordinate_core.y & 0xFFFF);

    // Compile-time arguments for kernels
    vector<uint32_t> sender_compile_args = {
        (uint32_t)l1_base_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)bytes_per_transaction,
        (uint32_t)test_config.test_id,
        (uint32_t)sem_id,
        (uint32_t)packed_subordinate_core_coordinates,
        (uint32_t)test_config.virtual_channel};

    // Kernels
    std::string kernels_dir = "tests/tt_metal/tt_metal/data_movement/one_to_one/kernels/";
    std::string sender_kernel_filename = "sender.cpp";
    std::string sender_kernel_path = kernels_dir + sender_kernel_filename;

    auto sender_kernel = CreateKernel(
        program,
        sender_kernel_path,
        test_config.master_core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = sender_compile_args});

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    /* ================ RUNNING THE PROGRAM ================ */

    // Setup Input
    // NOTE: The converted vector (uint32_t -> bfloat16) preserves the number of bytes,
    // but the number of elements is bound to change
    // l1_data_format is assumed to be bfloat16
    size_t element_size_bytes = bfloat16::SIZEOF;
    uint32_t num_elements = bytes_per_transaction / element_size_bytes;
    std::vector<uint32_t> packed_input = tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, num_elements, chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_golden = packed_input;

    // Write Input to Master L1
    tt_metal::detail::WriteToDeviceL1(device, test_config.master_core_coord, l1_base_address, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    // LAUNCH THE PROGRAM
    detail::LaunchProgram(device, program);

    // Record Output from Subordinate L1
    std::vector<uint32_t> packed_output;
    tt_metal::detail::ReadFromDeviceL1(
        device, test_config.subordinate_core_coord, l1_base_address, bytes_per_transaction, packed_output);

    // Compare output with golden vector
    bool pcc = is_close_packed_vectors<bfloat16, uint32_t>(
        packed_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });

    if (!pcc) {
        log_error(tt::LogTest, "PCC Check failed");
        log_info(tt::LogTest, "Golden vector");
        print_vector<uint32_t>(packed_golden);
        log_info(tt::LogTest, "Output vector");
        print_vector<uint32_t>(packed_output);
    }

    return pcc;
}

void directed_ideal_test(
    ARCH arch_,
    vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_id,
    CoreCoord master_core_coord = {0, 0},
    CoreCoord subordinate_core_coord = {0, 1},
    uint32_t virtual_channel = 0) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(arch_, devices_.at(0));

    // Adjustable Parameters
    // Ideal: Less transactions, more data per transaction
    uint32_t num_of_transactions = 256;
    uint32_t pages_per_transaction = max_transmittable_pages;

    // Cores
    // NOTE: May be worth considering the performance of this test with different pairs of adjacent cores
    //       for a different test case

    // Test Config
    unit_tests::dm::core_to_core::OneToOneConfig test_config = {
        .test_id = test_id,
        .master_core_coord = master_core_coord,
        .subordinate_core_coord = subordinate_core_coord,
        .num_of_transactions = num_of_transactions,
        .pages_per_transaction = pages_per_transaction,
        .bytes_per_page = bytes_per_page,
        .l1_data_format = DataFormat::Float16_b,
        .virtual_channel = virtual_channel};

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

void virtual_channels_test(
    ARCH arch_,
    vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_id,
    CoreCoord master_core_coord = {0, 0},
    CoreCoord subordinate_core_coord = {1, 1}) {
    for (uint32_t virtual_channel = 0; virtual_channel < 4;
         virtual_channel++) {  // FIND the constant that stores the number of unicast virtual channels
        directed_ideal_test(
            arch_, devices_, num_devices_, test_id, master_core_coord, subordinate_core_coord, virtual_channel);
    }
}

void packet_sizes_test(
    ARCH arch_,
    vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_id,
    CoreCoord master_core_coord = {0, 0},
    CoreCoord subordinate_core_coord = {1, 1}) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(arch_, devices_.at(0));

    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_pages_per_transaction =
        arch_ == tt::ARCH::BLACKHOLE ? 1024 : 2048;  // Max total transaction size == 64 KB

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            // Check if the total page size is within the limits
            if (pages_per_transaction > max_transmittable_pages) {
                continue;
            }

            // Test config
            unit_tests::dm::core_to_core::OneToOneConfig test_config = {
                .test_id = test_id,
                .master_core_coord = master_core_coord,
                .subordinate_core_coord = subordinate_core_coord,
                .num_of_transactions = num_of_transactions,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
            };

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

}  // namespace unit_tests::dm::core_to_core

/* ========== Test case for one to one data movement; Test id = 4 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToOnePacketSizes) {
    // Test ID
    uint32_t test_id = 4;

    unit_tests::dm::core_to_core::packet_sizes_test(arch_, devices_, num_devices_, test_id);
}

/* ========== Test case for one to one data movement; Test id = 50 ========== */  // Arbitrary test id

/*
    This test case is for directed ideal data movement from one L1 to another L1.
        1. Largest/most performant transaction size
        2. Large enough number of transactions to amortize the cycles for initialization
        3. Core locations with minimal number of hops
*/

TEST_F(DeviceFixture, TensixDataMovementOneToOneDirectedIdeal) {
    // Test ID (Arbitrary)
    uint32_t test_id = 50;

    unit_tests::dm::core_to_core::directed_ideal_test(
        arch_,
        devices_,
        num_devices_,
        test_id,
        CoreCoord(0, 0),  // Master Core
        CoreCoord(0, 1)   // Subordinate Core
    );
}

TEST_F(DeviceFixture, TensixDataMovementOneToOneVirtualChannels) {  // May be redundant
    // Test ID (Arbitrary)
    uint32_t test_id = 120;

    unit_tests::dm::core_to_core::virtual_channels_test(
        arch_,
        devices_,
        num_devices_,
        test_id,
        CoreCoord(0, 0),  // Master Core
        CoreCoord(0, 1)   // Subordinate Core
    );
}

TEST_F(DeviceFixture, TensixDataMovementOneToOneCoreLocationsVirtualChannels) {
    // Test ID (Arbitrary)
    uint32_t test_id = 132;

    CoreCoord mst_coord;
    CoreCoord sub_coord;

    CoreCoord stagnant_coord = {0, 0};

    auto grid_size = devices_.at(0)->compute_with_storage_grid_size();  // May need to fix this in a bit
    log_info(tt::LogTest, "Grid size x: {}, y: {}", grid_size.x, grid_size.y);

    for (unsigned int iteration = 0; iteration < 2; iteration++) {
        for (unsigned int x = 0; x < grid_size.x; x++) {
            for (unsigned int y = 0; y < grid_size.y; y++) {
                if (x == stagnant_coord.x && y == stagnant_coord.y) {
                    // Skip the first core in the first iteration
                    continue;
                }

                if (iteration == 0) {
                    mst_coord = stagnant_coord;
                    sub_coord = {x, y};
                } else {
                    mst_coord = {x, y};
                    sub_coord = stagnant_coord;
                }

                log_info(
                    tt::LogTest,
                    "Master Core: ({}, {}), Subordinate Core: ({}, {})",
                    mst_coord.x,
                    mst_coord.y,
                    sub_coord.x,
                    sub_coord.y);

                unit_tests::dm::core_to_core::virtual_channels_test(
                    arch_, devices_, num_devices_, test_id, mst_coord, sub_coord);
            }
        }
    }
}

TEST_F(DeviceFixture, TensixDataMovementOneToOneVirtualChannelsParallel) {
    // Test ID (Arbitrary)
    uint32_t test_id = 121;

    unit_tests::dm::core_to_core::directed_ideal_test(
        arch_,
        devices_,
        num_devices_,
        test_id,
        CoreCoord(0, 0),  // Master Core
        CoreCoord(0, 1)   // Subordinate Core
    );
}

}  // namespace tt::tt_metal
