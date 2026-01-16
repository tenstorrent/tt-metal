// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::core_to_and_from_core {

// Test config, i.e. test parameters
struct CoreBidirectionalConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord;
    CoreCoord subordinate_core_coord;
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
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const CoreBidirectionalConfig& test_config) {
    // Get the actual device for this single-device test
    IDevice* device = mesh_device->impl().get_device(0);
    /* ================ SETUP ================ */

    // Program
    Program program = CreateProgram();

    // Buffer Parameters
    const size_t bytes_per_transaction = test_config.pages_per_transaction * test_config.bytes_per_page;

    // Obtain L1 Address for Storing Data

    L1AddressInfo master_l1_info =
        tt::tt_metal::unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.master_core_coord);

    // Check if the L1 size is sufficient for the test configuration
    if (master_l1_info.size / 2 < bytes_per_transaction) {
        log_error(tt::LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }

    // Assigns a "safe" L1 local address for the master and subordinate cores
    uint32_t l1_base_write_address = master_l1_info.base_address;
    uint32_t l1_base_read_address = l1_base_write_address + (master_l1_info.size / 2);

    // Physical Core Coordinates
    CoreCoord physical_subordinate_core = device->worker_core_from_logical_core(test_config.subordinate_core_coord);
    uint32_t packed_subordinate_core_coordinates =
        physical_subordinate_core.x << 16 | (physical_subordinate_core.y & 0xFFFF);

    // KERNELS

    std::string kernels_dir = "tests/tt_metal/tt_metal/data_movement/core_bidirectional/kernels/";

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

        CreateKernel(
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

        CreateKernel(
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

        CreateKernel(
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
    size_t element_size_bytes = sizeof(bfloat16);
    uint32_t num_elements = bytes_per_transaction / element_size_bytes;
    std::vector<uint32_t> packed_input = tt::test_utils::generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, num_elements, chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_golden = packed_input;

    // Write Input to Master L1
    tt_metal::detail::WriteToDeviceL1(device, test_config.master_core_coord, l1_base_write_address, packed_input);
    tt_metal::detail::WriteToDeviceL1(device, test_config.subordinate_core_coord, l1_base_read_address, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices =
        distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));  // Single device at (0,0)
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // Record Output from Subordinate L1
    std::vector<uint32_t> packed_sender_output;
    std::vector<uint32_t> packed_requestor_output;
    tt_metal::detail::ReadFromDeviceL1(
        device, test_config.subordinate_core_coord, l1_base_write_address, bytes_per_transaction, packed_sender_output);
    tt_metal::detail::ReadFromDeviceL1(
        device, test_config.master_core_coord, l1_base_read_address, bytes_per_transaction, packed_requestor_output);

    // Compare output with golden vector
    bool is_equal;
    is_equal = (packed_sender_output == packed_golden);
    is_equal &= (packed_requestor_output == packed_golden);

    if (!is_equal) {
        log_error(tt::LogTest, "Equality Check failed");
        log_info(tt::LogTest, "Golden vector");
        print_vector<uint32_t>(packed_golden);
        log_info(tt::LogTest, "Sender output vector");
        print_vector<uint32_t>(packed_sender_output);
        log_info(tt::LogTest, "Requestor output vector");
        print_vector<uint32_t>(packed_requestor_output);
        return false;
    }

    return is_equal;
}

void directed_ideal_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord master_core_coord = {0, 0},
    CoreCoord subordinate_core_coord = {0, 1},
    uint32_t write_vc = 0,
    bool same_kernel = false) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    // Adjustable Parameters
    // Ideal: Less transactions, more data per transaction
    uint32_t num_of_transactions = 256 * (2 << 10);
    uint32_t pages_per_transaction = max_transmittable_pages / 2;  // 256; max_transmittable_pages / 2;

    // Test Config
    unit_tests::dm::core_to_and_from_core::CoreBidirectionalConfig test_config = {
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
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

void same_vc_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord master_core_coord = {0, 0},
    CoreCoord subordinate_core_coord = {0, 1},
    bool same_kernel = false) {
    uint32_t read_req_vc = 1;

    directed_ideal_test(mesh_device, test_id, master_core_coord, subordinate_core_coord, read_req_vc, same_kernel);
}

void packet_sizes_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord master_core_coord = {0, 0},
    CoreCoord subordinate_core_coord = {1, 1},
    bool same_kernel = false) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    IDevice* device = mesh_device->impl().get_device(0);
    uint32_t max_transactions = 256;
    uint32_t max_pages_per_transaction =
        device->arch() == tt::ARCH::BLACKHOLE ? 1024 : 2048;  // Max total transaction size == 64 KB

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_pages_per_transaction;
             pages_per_transaction *= 2) {
            // Check if the total page size is within the limits
            if (pages_per_transaction > max_transmittable_pages) {
                continue;
            }

            // Test config
            unit_tests::dm::core_to_and_from_core::CoreBidirectionalConfig test_config = {
                .test_id = test_id,
                .master_core_coord = master_core_coord,
                .subordinate_core_coord = subordinate_core_coord,
                .num_of_transactions = num_of_transactions,
                .pages_per_transaction = pages_per_transaction,
                .bytes_per_page = bytes_per_page,
                .l1_data_format = DataFormat::Float16_b,
                .same_kernel = same_kernel};

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

}  // namespace unit_tests::dm::core_to_and_from_core

// ========== Directed Ideal Tests ==========

TEST_F(GenericMeshDeviceFixture, TensixDataMovementCoreBidirectionalDirectedIdealSameKernel) {
    // Test ID (Arbitrary)
    uint32_t test_id = 140;
    bool same_kernel = true;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};
    uint32_t write_vc = 0;

    unit_tests::dm::core_to_and_from_core::directed_ideal_test(
        get_mesh_device(), test_id, master_core_coord, subordinate_core_coord, write_vc, same_kernel);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementCoreBidirectionalDirectedIdealDifferentKernels) {
    // Test ID (Arbitrary)
    uint32_t test_id = 141;
    bool same_kernel = false;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};
    uint32_t write_vc = 0;

    unit_tests::dm::core_to_and_from_core::directed_ideal_test(
        get_mesh_device(), test_id, master_core_coord, subordinate_core_coord, write_vc, same_kernel);
}

// ========== Same VC Tests ==========

TEST_F(GenericMeshDeviceFixture, TensixDataMovementCoreBidirectionalSameVCSameKernel) {
    // Test ID (Arbitrary)
    uint32_t test_id = 142;
    bool same_kernel = true;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};

    unit_tests::dm::core_to_and_from_core::same_vc_test(
        get_mesh_device(), test_id, master_core_coord, subordinate_core_coord, same_kernel);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementCoreBidirectionalSameVCDifferentKernels) {
    // Test ID (Arbitrary)
    uint32_t test_id = 143;
    bool same_kernel = false;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};

    unit_tests::dm::core_to_and_from_core::same_vc_test(
        get_mesh_device(), test_id, master_core_coord, subordinate_core_coord, same_kernel);
}

// ========== Write VC Sweep Tests ==========

TEST_F(GenericMeshDeviceFixture, TensixDataMovementCoreBidirectionalWriteVCSweepSameKernel) {
    // Test ID base
    uint32_t test_id_base = 144;
    bool same_kernel = true;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};

    // Sweep through write VCs 0-3
    for (uint32_t write_vc = 0; write_vc < 4; write_vc++) {
        uint32_t test_id = test_id_base + write_vc;

        unit_tests::dm::core_to_and_from_core::directed_ideal_test(
            get_mesh_device(), test_id, master_core_coord, subordinate_core_coord, write_vc, same_kernel);
    }
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementCoreBidirectionalWriteVCSweepDifferentKernels) {
    // Test ID base
    uint32_t test_id_base = 145;
    bool same_kernel = false;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};

    // Sweep through write VCs 0-3
    for (uint32_t write_vc = 0; write_vc < 4; write_vc++) {
        uint32_t test_id = test_id_base + write_vc;

        unit_tests::dm::core_to_and_from_core::directed_ideal_test(
            get_mesh_device(), test_id, master_core_coord, subordinate_core_coord, write_vc, same_kernel);
    }
}

// ========== Packet Sizes Tests ==========

TEST_F(GenericMeshDeviceFixture, TensixDataMovementCoreBidirectionalPacketSizesSameKernel) {
    // Test ID
    uint32_t test_id = 146;
    bool same_kernel = true;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {1, 1};

    unit_tests::dm::core_to_and_from_core::packet_sizes_test(
        get_mesh_device(), test_id, master_core_coord, subordinate_core_coord, same_kernel);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementCoreBidirectionalPacketSizesDifferentKernels) {
    // Test ID
    uint32_t test_id = 147;
    bool same_kernel = false;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {1, 1};

    unit_tests::dm::core_to_and_from_core::packet_sizes_test(
        get_mesh_device(), test_id, master_core_coord, subordinate_core_coord, same_kernel);
}

// ========== Custom Test Case ==========

TEST_F(GenericMeshDeviceFixture, TensixDataMovementCoreBidirectionalCustom) {
    // Test ID
    uint32_t test_id = 148;
    bool same_kernel = true;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 1};
    uint32_t write_vc = 0;

    unit_tests::dm::core_to_and_from_core::directed_ideal_test(
        get_mesh_device(), test_id, master_core_coord, subordinate_core_coord, write_vc, same_kernel);
}

}  // namespace tt::tt_metal
