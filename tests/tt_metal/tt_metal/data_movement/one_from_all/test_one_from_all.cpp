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
struct OneFromAllConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord = CoreCoord();
    CoreRangeSet subordinate_core_set;
    uint32_t num_of_transactions = 0;
    uint32_t transaction_size_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;

    // TODO: Add the following parameters
    //  1. Virtual Channel
    //  2. Which NOC to use
};

/// @brief Does Gatherer Core --> L1 Responder Cores --> L1 Gatherer Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const OneFromAllConfig& test_config) {
    // Program
    Program program = CreateProgram();

    // Sharded L1 buffers
    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    size_t total_subordinate_cores = test_config.subordinate_core_set.num_cores();

    const size_t transaction_size_bytes = test_config.transaction_size_pages * test_config.page_size_bytes;
    const size_t total_size_bytes = test_config.num_of_transactions * transaction_size_bytes * total_subordinate_cores;

    // Obtain L1 Address for Storing Data
    // NOTE: We don't know if the whole block of memory is actually available.
    //       This is something that could probably be checked
    L1AddressInfo master_l1_info =
        tt::tt_metal::unit_tests::dm::get_l1_address_and_size(device, test_config.master_core_coord);

    auto sub_core_list = corerange_to_cores(test_config.subordinate_core_set);
    for (auto& core : sub_core_list) {
        L1AddressInfo subordinate_l1_info = tt::tt_metal::unit_tests::dm::get_l1_address_and_size(device, core);
        // Checks that both master and subordinate cores have the same L1 base address and size
        if (master_l1_info.base_address != subordinate_l1_info.base_address ||
            master_l1_info.size != subordinate_l1_info.size) {
            log_error(tt::LogTest, "Mismatch in L1 address or size between master and subordinate cores");
            return false;
        }
    }
    // Check if the L1 size is sufficient for the test configuration
    if (master_l1_info.size < transaction_size_bytes) {
        log_error(tt::LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }
    // Assigns a "safe" L1 local address for the master and subordinate cores
    uint32_t l1_base_address = master_l1_info.base_address;

    const size_t subordinate_size_bytes = test_config.num_of_transactions * transaction_size_bytes;

    // Compile-time arguments for kernels
    vector<uint32_t> gatherer_compile_args = {
        (uint32_t)l1_base_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)transaction_size_bytes,
        (uint32_t)test_config.test_id,
        (uint32_t)total_subordinate_cores,
    };

    // Kernels
    auto gatherer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/one_from_all/kernels/gatherer.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = gatherer_compile_args});

    // Runtime Arguments
    vector<uint32_t> master_runtime_args;

    for (auto& core : sub_core_list) {
        CoreCoord physical_core = device->worker_core_from_logical_core(core);
        master_runtime_args.push_back(physical_core.x);
        master_runtime_args.push_back(physical_core.y);
    }
    SetRuntimeArgs(program, gatherer_kernel, master_core_set, master_runtime_args);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f,
        100.0f,
        transaction_size_bytes / bfloat16::SIZEOF,
        chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    vector<uint32_t> packed_golden = packed_input;

    // Launch program and record outputs
    for (size_t i = 0; i < total_subordinate_cores; i++) {
        detail::WriteToDeviceL1(device, sub_core_list[i], l1_base_address, packed_input);
    }
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    detail::LaunchProgram(device, program);

    vector<uint32_t> packed_output;
    detail::ReadFromDeviceL1(
        device, test_config.master_core_coord, l1_base_address, transaction_size_bytes, packed_output);

    // Results comparison
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
}  // namespace unit_tests::dm::core_to_core

/* ========== Test case for one from all data movement; Test id = 15 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneFromAllPacketSizes) {
    // Physical Constraints
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(arch_, devices_.at(0));

    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages =
        arch_ == tt::ARCH::BLACKHOLE ? 1024 : 2048;  // Max total transaction size == 64 KB
    // Cores
    CoreCoord master_core_coord = {0, 0};
    CoreRangeSet subordinate_core_set = {CoreRange(CoreCoord(1, 1), CoreCoord(4, 4))};
    size_t total_subordinate_cores = subordinate_core_set.num_cores();

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            if (transaction_size_pages > max_transmittable_pages) {
                continue;
            }

            // Test config
            unit_tests::dm::core_to_core::OneFromAllConfig test_config = {
                .test_id = 15,
                .master_core_coord = master_core_coord,
                .subordinate_core_set = subordinate_core_set,
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
            };

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ========== Test case for one from all data movement; Test id = 30 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneFromAllDirectedIdeal) {
    // Physical Constraints
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(arch_, devices_.at(0));

    // Cores
    CoreCoord master_core_coord = {0, 0};
    CoreRangeSet subordinate_core_set = {CoreRange(CoreCoord(1, 1), CoreCoord(4, 4))};
    size_t total_subordinate_cores = subordinate_core_set.num_cores();

    // Parameters
    // Ideal: Less transactions, more data per transaction
    uint32_t num_of_transactions = 1;
    if (max_transmittable_pages % (num_of_transactions * total_subordinate_cores) != 0) {
        log_error(tt::LogTest, "Max transmittable pages not evenly divisible by transactions and subordinate cores");
    }
    uint32_t transaction_size_pages = max_transmittable_pages / num_of_transactions / total_subordinate_cores;

    // Test config
    unit_tests::dm::core_to_core::OneFromAllConfig test_config = {
        .test_id = 30,
        .master_core_coord = master_core_coord,
        .subordinate_core_set = subordinate_core_set,
        .num_of_transactions = num_of_transactions,
        .transaction_size_pages = transaction_size_pages,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
    };

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
