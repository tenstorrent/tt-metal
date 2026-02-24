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
using namespace test_utils;

namespace unit_tests::dm::core_to_core {

// Test config, i.e. test parameters
struct OneToOneConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_core_coord = {0, 0};
    uint32_t num_of_transactions = 0;
    uint32_t pages_per_transaction = 0;
    uint32_t bytes_per_page = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    uint32_t num_virtual_channels = 1;  // Number of virtual channels to cycle through (must be > 1 for cycling)
    NOC noc_id = NOC::NOC_0;
    bool use_2_0_api = false;  // Use Device 2.0 API
};

/// @brief Does L1 Sender Core --> L1 Receiver Core
/// @param mesh_device - MeshDevice to run the test on
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const OneToOneConfig& test_config) {
    // Get the actual device for this single-device test
    IDevice* device = mesh_device->impl().get_device(0);

    /* ================ SETUP ================ */

    // Program
    Program program = CreateProgram();

    // Buffer Parameters
    const size_t bytes_per_transaction = test_config.pages_per_transaction * test_config.bytes_per_page;

    // (Logical) Core coordinates and ranges
    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    CoreRangeSet subordinate_core_set({CoreRange(test_config.subordinate_core_coord)});
    CoreRangeSet combined_core_set = master_core_set.merge<CoreRangeSet>(subordinate_core_set);

    // Obtain L1 Address for Storing Data
    // NOTE: We don't know if the whole block of memory is actually available.
    //       This is something that could probably be checked
    L1AddressInfo master_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.master_core_coord);
    L1AddressInfo subordinate_l1_info =
        unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.subordinate_core_coord);
    // Checks that both master and subordinate cores have the same L1 base address and size
    if (master_l1_info.base_address != subordinate_l1_info.base_address ||
        master_l1_info.size != subordinate_l1_info.size) {
        log_error(LogTest, "Mismatch in L1 address or size between master and subordinate cores");
        return false;
    }
    // Check if the L1 size is sufficient for the test configuration
    if (master_l1_info.size < bytes_per_transaction) {
        log_error(LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }
    // Validate virtual channels configuration
    if (test_config.num_virtual_channels > 4) {
        log_error(
            tt::LogTest,
            "num_virtual_channels must not be greater than 4 as there are only 4 unicast write virtual channels");
        return false;
    }
    // Assigns a "safe" L1 local address for the master and subordinate cores
    uint32_t l1_base_address = master_l1_info.base_address;

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
        (uint32_t)packed_subordinate_core_coordinates,
        (uint32_t)test_config.num_virtual_channels};

    // Kernels
    std::string kernels_dir = "tests/tt_metal/tt_metal/data_movement/one_to_one/kernels/";
    std::string sender_kernel_filename = "sender";
    if (test_config.use_2_0_api) {
        sender_kernel_filename += "_2_0";
    }
    std::string sender_kernel_path = kernels_dir + sender_kernel_filename + ".cpp";

    CreateKernel(
        program,
        sender_kernel_path,
        test_config.master_core_coord,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = test_config.noc_id,
            .compile_args = sender_compile_args});

    // Assign unique id
    log_info(LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    /* ================ RUNNING THE PROGRAM ================ */

    // Setup Input
    // NOTE: The converted vector (uint32_t -> bfloat16) preserves the number of bytes,
    // but the number of elements is bound to change
    // l1_data_format is assumed to be bfloat16
    size_t element_size_bytes = sizeof(bfloat16);
    uint32_t num_elements = bytes_per_transaction / element_size_bytes;
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, num_elements, chrono::system_clock::now().time_since_epoch().count());
    vector<uint32_t> packed_golden = packed_input;

    // Write Input to Master L1
    detail::WriteToDeviceL1(device, test_config.master_core_coord, l1_base_address, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    // LAUNCH THE PROGRAM - Use mesh workload approach
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices =
        distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));  // Single device at (0,0)
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // Record Output from Subordinate L1
    vector<uint32_t> packed_output;
    detail::ReadFromDeviceL1(
        device, test_config.subordinate_core_coord, l1_base_address, bytes_per_transaction, packed_output);

    // Compare output with golden vector
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
    uint32_t test_id,
    CoreCoord master_core_coord = {0, 0},
    CoreCoord subordinate_core_coord = {0, 1},
    uint32_t num_virtual_channels = 1) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

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
        .num_virtual_channels = num_virtual_channels};

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

void virtual_channels_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord master_core_coord = {0, 0},
    CoreCoord subordinate_core_coord = {1, 1}) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    std::uint32_t max_num_pages_per_transaction = 1 << 12;
    std::uint32_t num_of_transactions = 256;  // Constant value
    std::uint32_t max_num_virtual_channels = 4;

    // Loop through:
    // 1. NOCs (NOC_0, NOC_1)
    // 2. Size of transactions
    // 3. Numbers of virtual channels
    for (NOC noc_id : {NOC::NOC_0, NOC::NOC_1}) {
        for (uint32_t pages_per_transaction = 1; pages_per_transaction <= max_num_pages_per_transaction;
             pages_per_transaction *= 2) {
            for (uint32_t num_virtual_channels = 1; num_virtual_channels <= max_num_virtual_channels;
                 num_virtual_channels++) {
                // Check if the total page size is within the limits
                if (pages_per_transaction > max_transmittable_pages) {
                    continue;
                }

                unit_tests::dm::core_to_core::OneToOneConfig test_config = {
                    .test_id = test_id,
                    .master_core_coord = master_core_coord,
                    .subordinate_core_coord = subordinate_core_coord,
                    .num_of_transactions = num_of_transactions,
                    .pages_per_transaction = pages_per_transaction,
                    .bytes_per_page = bytes_per_page,
                    .l1_data_format = DataFormat::Float16_b,
                    .num_virtual_channels = num_virtual_channels,
                    .noc_id = noc_id};

                EXPECT_TRUE(run_dm(mesh_device, test_config));
            }
        }
    }
}

void packet_sizes_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord master_core_coord = {0, 0},
    CoreCoord subordinate_core_coord = {1, 1},
    bool use_2_0_api = false) {
    IDevice* device = mesh_device->impl().get_device(0);
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_pages_per_transaction =
        device->arch() == ARCH::BLACKHOLE ? 1024 : 2048;  // Max total transaction size == 64 KB

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
                .use_2_0_api = use_2_0_api,
            };

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

void custom_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord master_core_coord = {0, 0},
    CoreCoord subordinate_core_coord = {1, 1},
    uint32_t num_of_transactions = 256,
    uint32_t pages_per_transaction = 1,
    uint32_t num_virtual_channels = 4) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Test config
    unit_tests::dm::core_to_core::OneToOneConfig test_config = {
        .test_id = test_id,
        .master_core_coord = master_core_coord,
        .subordinate_core_coord = subordinate_core_coord,
        .num_of_transactions = num_of_transactions,
        .pages_per_transaction = pages_per_transaction,
        .bytes_per_page = bytes_per_page,
        .l1_data_format = DataFormat::Float16_b,
        .num_virtual_channels = num_virtual_channels};

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

}  // namespace unit_tests::dm::core_to_core

/* ========== TEST CASES ========== */

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToOnePacketSizes) {
    // Test ID
    uint32_t test_id = 4;

    unit_tests::dm::core_to_core::packet_sizes_test(get_mesh_device(), test_id);
}

/*
    This test case is for directed ideal data movement from one L1 to another L1.
        1. Largest/most performant transaction size
        2. Large enough number of transactions to amortize the cycles for initialization
        3. Core locations with minimal number of hops
*/

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToOneDirectedIdeal) {
    // Test ID (Arbitrary)
    uint32_t test_id = 50;

    unit_tests::dm::core_to_core::directed_ideal_test(
        get_mesh_device(),
        test_id,
        CoreCoord(0, 0),  // Master Core
        CoreCoord(0, 1)   // Subordinate Core
    );
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToOneVirtualChannels) {
    GTEST_SKIP() << "Skipping test";
    // Test ID (Arbitrary)
    uint32_t test_id = 150;

    unit_tests::dm::core_to_core::virtual_channels_test(
        get_mesh_device(),
        test_id,
        CoreCoord(0, 0),  // Master Core
        CoreCoord(0, 1)   // Subordinate Core
    );
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToOneCustom) {
    GTEST_SKIP() << "Skipping test";
    uint32_t test_id = 151;

    // Parameters
    uint32_t num_of_transactions = 256;
    uint32_t pages_per_transaction = 1;
    uint32_t num_virtual_channels = 4;

    unit_tests::dm::core_to_core::custom_test(
        get_mesh_device(),
        test_id,
        CoreCoord(0, 0),  // Master Core
        CoreCoord(0, 1),  // Subordinate Core
        num_of_transactions,
        pages_per_transaction,
        num_virtual_channels);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToOnePacketSizes2_0) {
    // Test ID
    uint32_t test_id = 158;

    unit_tests::dm::core_to_core::packet_sizes_test(get_mesh_device(), test_id, CoreCoord(0, 0), CoreCoord(1, 1), true);
}
}  // namespace tt::tt_metal
