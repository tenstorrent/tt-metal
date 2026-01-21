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

namespace unit_tests::dm::core_from_all {
// Test config, i.e. test parameters
struct OneFromAllConfig {
    uint32_t test_id = 0;

    CoreCoord master_core_coord;
    CoreCoord sub_start_core_coord;
    CoreCoord sub_grid_size;

    uint32_t num_of_transactions = 0;
    uint32_t transaction_size_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    NOC noc_id = NOC::RISCV_1_default;
    uint32_t num_virtual_channels = 1;

    // TODO: Add the following parameters
    //  1. Virtual Channel
    //  2. Which NOC to use
};

/// @brief Does Gatherer Core --> L1 Responder Cores --> L1 Gatherer Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @param fixture - DispatchFixture pointer for dispatch-aware operations
/// @return
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const OneFromAllConfig& test_config) {
    IDevice* device = mesh_device->impl().get_device(0);
    // Program
    Program program = CreateProgram();

    // Sharded L1 buffers
    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});

    // Calculate subordinate core set from start coord and grid size
    CoreCoord sub_logical_start_coord = test_config.sub_start_core_coord;
    CoreCoord sub_logical_end_coord = CoreCoord(
        sub_logical_start_coord.x + test_config.sub_grid_size.x - 1,
        sub_logical_start_coord.y + test_config.sub_grid_size.y - 1);
    CoreRangeSet subordinate_core_set({CoreRange(sub_logical_start_coord, sub_logical_end_coord)});

    size_t total_subordinate_cores = subordinate_core_set.num_cores();

    const size_t transaction_size_bytes = test_config.transaction_size_pages * test_config.page_size_bytes;

    // Obtain L1 Address for Storing Data
    // NOTE: We don't know if the whole block of memory is actually available.
    //       This is something that could probably be checked
    L1AddressInfo master_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.master_core_coord);

    auto sub_core_list = corerange_to_cores(subordinate_core_set);
    for (auto& core : sub_core_list) {
        L1AddressInfo subordinate_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, core);
        // Checks that both master and subordinate cores have the same L1 base address and size
        if (master_l1_info.base_address != subordinate_l1_info.base_address ||
            master_l1_info.size != subordinate_l1_info.size) {
            log_error(LogTest, "Mismatch in L1 address or size between master and subordinate cores");
            return false;
        }
    }
    // Check if the L1 size is sufficient for the test configuration
    if (master_l1_info.size < transaction_size_bytes) {
        log_error(LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }
    // Assigns a "safe" L1 local address for the master and subordinate cores
    uint32_t l1_base_address = master_l1_info.base_address;

    // Compile-time arguments for kernels
    vector<uint32_t> gatherer_compile_args = {
        (uint32_t)l1_base_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)transaction_size_bytes,
        (uint32_t)test_config.test_id,
        (uint32_t)total_subordinate_cores,
        (uint32_t)test_config.num_virtual_channels,
    };

    // Kernels
    auto gatherer_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/one_from_all/kernels/gatherer.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = test_config.noc_id,
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
    log_info(LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Input
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f,
        100.0f,
        transaction_size_bytes / sizeof(bfloat16),
        chrono::system_clock::now().time_since_epoch().count());

    // Golden output
    vector<uint32_t> packed_golden = packed_input;

    // Launch program and record outputs
    for (size_t i = 0; i < total_subordinate_cores; i++) {
        detail::WriteToDeviceL1(device, sub_core_list[i], l1_base_address, packed_input);
    }
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    // LAUNCH PROGRAM - Use mesh workload approach
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    vector<uint32_t> packed_output;
    detail::ReadFromDeviceL1(
        device, test_config.master_core_coord, l1_base_address, transaction_size_bytes, packed_output);

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
    uint32_t test_id,
    CoreCoord master_core_coord,
    CoreCoord subordinate_start_coord,
    CoreCoord subordinate_grid_size,
    NOC noc_id = NOC::RISCV_1_default) {
    // Physical Constraints
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    size_t total_subordinate_cores = subordinate_grid_size.x * subordinate_grid_size.y;

    // Parameters
    // Ideal: Less transactions, more data per transaction
    uint32_t num_of_transactions = 1;
    uint32_t transaction_size_pages = max_transmittable_pages / (num_of_transactions * total_subordinate_cores);

    // Test config
    OneFromAllConfig test_config = {
        .test_id = test_id,
        .master_core_coord = master_core_coord,
        .sub_start_core_coord = subordinate_start_coord,
        .sub_grid_size = subordinate_grid_size,
        .num_of_transactions = num_of_transactions,
        .transaction_size_pages = transaction_size_pages,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
        .noc_id = noc_id,
    };

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

void packet_sizes_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord master_core_coord,
    CoreCoord subordinate_start_coord,
    CoreCoord subordinate_grid_size,
    NOC noc_id = NOC::RISCV_1_default) {
    // Physical Constraints
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages = mesh_device->impl().get_device(0)->arch() == tt::ARCH::BLACKHOLE
                                              ? 1024
                                              : 2048;  // Max total transaction size == 64 KB

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            if (transaction_size_pages > max_transmittable_pages) {
                continue;
            }

            // Test config
            OneFromAllConfig test_config = {
                .test_id = test_id,
                .master_core_coord = master_core_coord,
                .sub_start_core_coord = subordinate_start_coord,
                .sub_grid_size = subordinate_grid_size,
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

void virtual_channels_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord master_core_coord,
    CoreCoord subordinate_start_coord,
    CoreCoord subordinate_grid_size) {
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
                if (pages_per_transaction > max_transmittable_pages) {
                    continue;
                }

                // Test config
                OneFromAllConfig test_config = {
                    .test_id = test_id,
                    .master_core_coord = master_core_coord,
                    .sub_start_core_coord = subordinate_start_coord,
                    .sub_grid_size = subordinate_grid_size,
                    .num_of_transactions = num_of_transactions,
                    .transaction_size_pages = pages_per_transaction,
                    .page_size_bytes = bytes_per_page,
                    .l1_data_format = DataFormat::Float16_b,
                    .noc_id = noc_id,
                    .num_virtual_channels = num_virtual_channels,
                };

                // Run
                EXPECT_TRUE(run_dm(mesh_device, test_config));
            }
        }
    }
}

void custom_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord master_core_coord,
    CoreCoord subordinate_start_coord,
    CoreCoord subordinate_grid_size,
    uint32_t num_of_transactions,
    uint32_t pages_per_transaction,
    uint32_t num_virtual_channels,
    NOC noc_id = NOC::RISCV_1_default) {
    // Physical Constraints
    auto [bytes_per_page, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(mesh_device);

    size_t total_subordinate_cores = subordinate_grid_size.x * subordinate_grid_size.y;
    if (pages_per_transaction > max_transmittable_pages / total_subordinate_cores) {
        log_trace(LogTest, "Skipping test due to page size limitations");
        return;
    }

    // Test config
    OneFromAllConfig test_config = {
        .test_id = test_id,
        .master_core_coord = master_core_coord,
        .sub_start_core_coord = subordinate_start_coord,
        .sub_grid_size = subordinate_grid_size,
        .num_of_transactions = num_of_transactions,
        .transaction_size_pages = pages_per_transaction,
        .page_size_bytes = bytes_per_page,
        .l1_data_format = DataFormat::Float16_b,
        .noc_id = noc_id,
        .num_virtual_channels = num_virtual_channels,
    };

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

}  // namespace unit_tests::dm::core_from_all

/* ========== Test case for one from all data movement; Test id = 15 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneFromAllPacketSizes) {
    uint32_t test_id = 15;
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_start_coord = {0, 0};
    CoreCoord subordinate_grid_size = {
        device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::core_from_all::packet_sizes_test(
        get_mesh_device(), test_id, master_core_coord, subordinate_start_coord, subordinate_grid_size);
}

/* ========== Test case for one from all data movement; Test id = 30 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneFromAllDirectedIdeal) {
    uint32_t test_id = 30;
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_start_coord = {0, 0};
    CoreCoord subordinate_grid_size = {
        device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::core_from_all::directed_ideal_test(
        get_mesh_device(), test_id, master_core_coord, subordinate_start_coord, subordinate_grid_size);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneFromAllVirtualChannels) {
    GTEST_SKIP() << "Skipping test";

    // Test ID (Arbitrary)
    uint32_t test_id = 156;
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_start_coord = {0, 0};
    CoreCoord subordinate_grid_size = {
        device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::core_from_all::virtual_channels_test(
        get_mesh_device(), test_id, master_core_coord, subordinate_start_coord, subordinate_grid_size);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneFromAllCustom) {
    GTEST_SKIP() << "Skipping test";

    uint32_t test_id = 157;
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    CoreCoord master_core_coord = {0, 0};
    CoreCoord subordinate_start_coord = {0, 0};
    CoreCoord subordinate_grid_size = {
        device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    // Parameters
    uint32_t num_of_transactions = 256;
    uint32_t pages_per_transaction = 1;
    uint32_t num_virtual_channels = 4;

    unit_tests::dm::core_from_all::custom_test(
        get_mesh_device(),
        test_id,
        master_core_coord,
        subordinate_start_coord,
        subordinate_grid_size,
        num_of_transactions,
        pages_per_transaction,
        num_virtual_channels);
}

}  // namespace tt::tt_metal
