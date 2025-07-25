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

namespace unit_tests::dm::core_to_and_from_core {

// Test config, i.e. test parameters
struct OneToAndFromOneConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord = CoreCoord();
    CoreCoord subordinate_core_coord = CoreCoord();
    uint32_t num_of_transactions = 0;
    uint32_t pages_per_transaction = 0;
    uint32_t bytes_per_page = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    uint32_t write_vc = 0;  // Virtual channel for the NOC
    bool same_kernel = false;

    // TODO: Add the following parameters
    //  1. Which NOC to use
    //  2. Posted flag
};

/// @brief Does L1 Sender Core --> L1 Receiver Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const OneToAndFromOneConfig& test_config) {
    /* ================ SETUP ================ */

    // Program
    Program program = CreateProgram();

    // Buffer Parameters
    const size_t bytes_per_transaction = test_config.pages_per_transaction * test_config.bytes_per_page;
    const size_t total_size_bytes = bytes_per_transaction * test_config.num_of_transactions;

    // Obtain L1 Address for Storing Data

    L1AddressInfo master_l1_info =
        tt::tt_metal::unit_tests::dm::get_l1_address_and_size(device, test_config.master_core_coord);
    L1AddressInfo subordinate_l1_info =
        tt::tt_metal::unit_tests::dm::get_l1_address_and_size(device, test_config.subordinate_core_coord);

    // Check if the L1 size is sufficient for the test configuration
    if (master_l1_info.size / 2 < bytes_per_transaction) {
        log_error(tt::LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }

    // Assigns a "safe" L1 local address for the master and subordinate cores
    uint32_t l1_base_write_address = master_l1_info.base_address;
    uint32_t l1_base_read_address = l1_base_write_address + master_l1_info.size / 2;

    // Physical Core Coordinates
    CoreCoord physical_subordinate_core = device->worker_core_from_logical_core(test_config.subordinate_core_coord);
    uint32_t packed_subordinate_core_coordinates =
        physical_subordinate_core.x << 16 | (physical_subordinate_core.y & 0xFFFF);

    // KERNELS

    std::string kernels_dir = "tests/tt_metal/tt_metal/data_movement/one_to_and_from_one/kernels/";

    if (test_config.same_kernel) {
        // Sender and Requestor Kernel

        std::string sender_and_requestor_kernel_filename = "sender_and_requestor.cpp";
        std::string sender_and_requestor_kernel_path = kernels_dir + sender_and_requestor_kernel_filename;

        vector<uint32_t> sender_and_requestor_compile_args = {
            (uint32_t)test_config.test_id,
            (uint32_t)l1_base_write_address,
            (uint32_t)l1_base_read_address,
            (uint32_t)test_config.num_of_transactions,
            (uint32_t)bytes_per_transaction,
            (uint32_t)packed_subordinate_core_coordinates,
            (uint32_t)test_config.write_vc};

        auto sender_and_requestor_kernel = CreateKernel(
            program,
            sender_and_requestor_kernel_path,
            test_config.master_core_coord,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = sender_and_requestor_compile_args});

    } else {
        // Sender Kernel

        std::string sender_kernel_filename = "sender.cpp";
        std::string sender_kernel_path = kernels_dir + sender_kernel_filename;

        vector<uint32_t> sender_compile_args = {
            (uint32_t)test_config.test_id,
            (uint32_t)l1_base_write_address,
            (uint32_t)test_config.num_of_transactions,
            (uint32_t)bytes_per_transaction,
            (uint32_t)packed_subordinate_core_coordinates,
            (uint32_t)test_config.write_vc};

        auto sender_kernel = CreateKernel(
            program,
            sender_kernel_path,
            test_config.master_core_coord,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = sender_compile_args});

        // Requestor Kernel

        std::string requestor_kernel_filename = "requestor.cpp";
        std::string requestor_kernel_path = kernels_dir + requestor_kernel_filename;

        vector<uint32_t> requestor_compile_args = {
            (uint32_t)test_config.test_id,
            (uint32_t)l1_base_read_address,
            (uint32_t)test_config.num_of_transactions,
            (uint32_t)bytes_per_transaction,
            (uint32_t)packed_subordinate_core_coordinates};

        auto requestor_kernel = CreateKernel(
            program,
            requestor_kernel_path,
            test_config.master_core_coord,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_0_default,
                .compile_args = requestor_compile_args});
    }

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    /* ================ RUNNING THE PROGRAM ================ */

    // Setup Input
    size_t element_size_bytes = bfloat16::SIZEOF;
    uint32_t num_elements = bytes_per_transaction / element_size_bytes;
    std::vector<uint32_t> packed_input = tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, num_elements, chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_golden = packed_input;

    // Write Input to Master L1
    tt_metal::detail::WriteToDeviceL1(device, test_config.master_core_coord, l1_base_write_address, packed_input);
    tt_metal::detail::WriteToDeviceL1(device, test_config.subordinate_core_coord, l1_base_read_address, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    // LAUNCH THE PROGRAM
    detail::LaunchProgram(device, program);

    // Record Output from Subordinate L1
    std::vector<uint32_t> packed_sender_output;
    std::vector<uint32_t> packed_requestor_output;
    tt_metal::detail::ReadFromDeviceL1(
        device, test_config.subordinate_core_coord, l1_base_write_address, bytes_per_transaction, packed_sender_output);
    tt_metal::detail::ReadFromDeviceL1(
        device, test_config.master_core_coord, l1_base_read_address, bytes_per_transaction, packed_requestor_output);

    // Compare output with golden vector
    bool pcc;
    pcc = is_close_packed_vectors<bfloat16, uint32_t>(
        packed_sender_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });
    pcc &= is_close_packed_vectors<bfloat16, uint32_t>(
        packed_requestor_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });

    if (!pcc) {
        log_error(tt::LogTest, "PCC Check failed");
        log_info(tt::LogTest, "Golden vector");
        print_vector<uint32_t>(packed_golden);
        log_info(tt::LogTest, "Sender output vector");
        print_vector<uint32_t>(packed_sender_output);
        log_info(tt::LogTest, "Requestor output vector");
        print_vector<uint32_t>(packed_requestor_output);
        return false;
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
    uint32_t write_vc = 0,
    bool same_kernel = false) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(arch_, devices_.at(0));

    // Adjustable Parameters
    // Ideal: Less transactions, more data per transaction
    uint32_t num_of_transactions = 256 * (2 << 10);
    uint32_t pages_per_transaction = max_transmittable_pages / 2;  // 256; max_transmittable_pages / 2;

    // Test Config
    unit_tests::dm::core_to_and_from_core::OneToAndFromOneConfig test_config = {
        .test_id = test_id,
        .master_core_coord = master_core_coord,
        .subordinate_core_coord = subordinate_core_coord,
        .num_of_transactions = num_of_transactions,
        .pages_per_transaction = pages_per_transaction,
        .bytes_per_page = bytes_per_page,
        .l1_data_format = DataFormat::Float16_b,
        .write_vc = write_vc,
        .same_kernel = same_kernel};

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

void same_vc_test(
    ARCH arch_,
    vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_id,
    CoreCoord master_core_coord = {0, 0},
    CoreCoord subordinate_core_coord = {0, 1},
    bool same_kernel = false) {
    uint32_t read_req_vc = 1;

    directed_ideal_test(
        arch_, devices_, num_devices_, test_id, master_core_coord, subordinate_core_coord, read_req_vc, same_kernel);
}

void packet_sizes_test(
    ARCH arch_,
    vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_id,
    CoreCoord master_core_coord = {0, 0},
    CoreCoord subordinate_core_coord = {1, 1},
    bool same_kernel = false) {
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
            unit_tests::dm::core_to_and_from_core::OneToAndFromOneConfig test_config = {
                .test_id = test_id,
                .master_core_coord = master_core_coord,
                .subordinate_core_coord = subordinate_core_coord,
                .num_of_transactions = num_of_transactions,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
                .same_kernel = same_kernel};

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

}  // namespace unit_tests::dm::core_to_and_from_core

TEST_F(DeviceFixture, TensixDataMovementOneToAndFromOneDirectedIdealSameKernel) {
    // Test ID (Arbitrary)
    uint32_t test_id = 140;
    bool same_kernel = true;

    unit_tests::dm::core_to_and_from_core::directed_ideal_test(
        arch_,
        devices_,
        num_devices_,
        test_id,
        {0, 0},  // master_core_coord
        {0, 1},  // subordinate_core_coord
        0,       // write_vc
        same_kernel);
}

TEST_F(DeviceFixture, TensixDataMovementOneToAndFromOneDirectedIdealDifferentKernels) {
    // Test ID (Arbitrary)
    uint32_t test_id = 141;
    bool same_kernel = false;

    unit_tests::dm::core_to_and_from_core::directed_ideal_test(
        arch_,
        devices_,
        num_devices_,
        test_id,
        {0, 0},  // master_core_coord
        {0, 1},  // subordinate_core_coord
        0,       // write_vc
        same_kernel);
}

TEST_F(DeviceFixture, TensixDataMovementOneToAndFromOneSameVCSameKernel) {
    // Test ID (Arbitrary)
    uint32_t test_id = 142;
    bool same_kernel = true;

    unit_tests::dm::core_to_and_from_core::same_vc_test(
        arch_,
        devices_,
        num_devices_,
        test_id,
        {0, 0},  // master_core_coord
        {0, 1},  // subordinate_core_coord
        same_kernel);
}

TEST_F(DeviceFixture, TensixDataMovementOneToAndFromOneSameVCDifferentKernels) {
    // Test ID (Arbitrary)
    uint32_t test_id = 143;
    bool same_kernel = false;

    unit_tests::dm::core_to_and_from_core::same_vc_test(
        arch_,
        devices_,
        num_devices_,
        test_id,
        {0, 0},  // master_core_coord
        {0, 1},  // subordinate_core_coord
        same_kernel);
}

TEST_F(DeviceFixture, TensixDataMovementOneToAndFromOneWriteVCSweepSameKernel) {
    // Test ID base
    uint32_t test_id_base = 146;
    bool same_kernel = true;

    // Sweep through write VCs 0-3
    for (uint32_t write_vc = 0; write_vc < 4; write_vc++) {
        uint32_t test_id = test_id_base + write_vc;

        unit_tests::dm::core_to_and_from_core::directed_ideal_test(
            arch_,
            devices_,
            num_devices_,
            test_id,
            {0, 0},  // master_core_coord
            {0, 1},  // subordinate_core_coord
            write_vc,
            same_kernel);
    }
}

TEST_F(DeviceFixture, TensixDataMovementOneToAndFromOneWriteVCSweepDifferentKernels) {
    // Test ID base
    uint32_t test_id_base = 150;
    bool same_kernel = false;

    // Sweep through write VCs 0-3
    for (uint32_t write_vc = 0; write_vc < 4; write_vc++) {
        uint32_t test_id = test_id_base + write_vc;

        unit_tests::dm::core_to_and_from_core::directed_ideal_test(
            arch_,
            devices_,
            num_devices_,
            test_id,
            {0, 0},  // master_core_coord
            {0, 1},  // subordinate_core_coord
            write_vc,
            same_kernel);
    }
}

TEST_F(DeviceFixture, TensixDataMovementOneToAndFromOnePacketSizesSameKernel) {
    // Test ID
    uint32_t test_id = 144;
    bool same_kernel = true;

    unit_tests::dm::core_to_and_from_core::packet_sizes_test(
        arch_,
        devices_,
        num_devices_,
        test_id,
        {0, 0},  // master_core_coord
        {1, 1},  // subordinate_core_coord
        same_kernel);
}

TEST_F(DeviceFixture, TensixDataMovementOneToAndFromOnePacketSizesDifferentKernels) {
    // Test ID
    uint32_t test_id = 145;
    bool same_kernel = false;

    unit_tests::dm::core_to_and_from_core::packet_sizes_test(
        arch_,
        devices_,
        num_devices_,
        test_id,
        {0, 0},  // master_core_coord
        {1, 1},  // subordinate_core_coord
        same_kernel);
}

}  // namespace tt::tt_metal
