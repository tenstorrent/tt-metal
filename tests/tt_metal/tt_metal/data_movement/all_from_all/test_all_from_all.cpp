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
using namespace test_utils;

namespace unit_tests::dm::all_from_all {

constexpr uint32_t START_ID = 310;

// Test Config (i.e. test parameters)
struct AllFromAllConfig {
    /* Test ID */
    uint32_t test_id = START_ID;

    /* Grid configurations */
    CoreCoord mst_logical_start_coord = {0, 0};
    CoreCoord sub_logical_start_coord = {0, 0};
    CoreCoord mst_grid_size = {0, 0};
    CoreCoord sub_grid_size = {0, 0};

    /* Transaction size configurations */
    uint32_t num_of_transactions_per_subordinate = 1;
    uint32_t pages_reservable_per_transaction = 1;
    uint32_t bytes_per_page = 32;

    /* Write configurations */
    DataFormat l1_data_format = DataFormat::Invalid;
    NOC noc_id = NOC::NOC_1;
    uint32_t num_virtual_channels = 1;

    // TODO: Add the following parameters
    //  1. Virtual Channel (only useful for unicast)
    //  2. Posted flag (posted multicast has much better performance at larger grid sizes, than non-posted due to
    //  response packets) (60, 45, 23, vs 60, 60, 60 at posted)
};

/// @brief Performs communication from L1 Sender Cores to L1 Receiver Cores.
/// @param mesh_device The mesh device on which the test is executed.
/// @param test_config Configuration of the test, defined by a specific struct.
/// @return Status of the test execution (e.g., success or failure).
bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const AllFromAllConfig& test_config) {
    // Get the actual device for this single-device test
    IDevice* device = mesh_device->impl().get_device(0);
    /* ================ SETUP ================ */

    // Program
    Program program = CreateProgram();

    // Initialize core sets //
    /*
        - CoreRange: Represents a single rectangular range of cores in a 2D grid. It is defined by a starting coordinate
        (`start_coord`) and an ending coordinate (`end_coord`). For example, a CoreRange from (0, 0) to (3, 3) would
        include all cores in that rectangular area.

        - CoreRangeSet: Represents a collection of CoreRange objects. It is used to manage multiple ranges of cores
        and provides functionality for operations like merging, subtracting, and finding intersections between ranges.
    */

    // Logical Cores

    // Master

    CoreCoord mst_logical_start_coord = test_config.mst_logical_start_coord;
    CoreCoord mst_logical_end_coord = CoreCoord(
        mst_logical_start_coord.x + test_config.mst_grid_size.x - 1,
        mst_logical_start_coord.y + test_config.mst_grid_size.y - 1);

    CoreRangeSet mst_logical_core_set({CoreRange(mst_logical_start_coord, mst_logical_end_coord)});

    // Subordinate

    CoreCoord sub_logical_start_coord = test_config.sub_logical_start_coord;
    CoreCoord sub_logical_end_coord = CoreCoord(
        sub_logical_start_coord.x + test_config.sub_grid_size.x - 1,
        sub_logical_start_coord.y + test_config.sub_grid_size.y - 1);

    CoreRangeSet sub_logical_core_set({CoreRange(sub_logical_start_coord, sub_logical_end_coord)});
    uint32_t num_subordinates = sub_logical_core_set.num_cores();

    // Subordinate Worker Coordinates

    vector<uint32_t> sub_worker_coordinates = {};
    for (auto& sub_logical_core : corerange_to_cores(sub_logical_core_set)) {
        CoreCoord sub_worker_core = device->worker_core_from_logical_core(sub_logical_core);
        sub_worker_coordinates.push_back(sub_worker_core.x);
        sub_worker_coordinates.push_back(sub_worker_core.y);
    }

    // Transaction Configurations
    const size_t bytes_per_transaction = test_config.pages_reservable_per_transaction * test_config.bytes_per_page;

    // Obtain L1 Address for Storing Data

    L1AddressInfo core_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, {0, 0});
    uint32_t sub_l1_base_address = core_l1_info.base_address;
    uint32_t mst_l1_base_address = sub_l1_base_address + bytes_per_transaction;

    // Possible To-Do: Implement checks to see that the needed space is available in all master and subordinate cores

    // Kernels

    // Compile-time arguments for kernels

    vector<uint32_t> requestor_compile_args = {
        //     0: Test ID
        (uint32_t)test_config.test_id,
        // 1 - 2: L1 Addresses
        (uint32_t)mst_l1_base_address,
        (uint32_t)sub_l1_base_address,
        // 3 - 4: Transaction parameters
        (uint32_t)test_config.num_of_transactions_per_subordinate,
        (uint32_t)bytes_per_transaction,
        //     5: Subordinate count
        (uint32_t)num_subordinates,
        //     6: Virtual channels
        (uint32_t)test_config.num_virtual_channels,
    };

    // Create kernels
    auto requestor_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/all_from_all/kernels/requestor.cpp",
        mst_logical_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = test_config.noc_id,
            .compile_args = requestor_compile_args});

    // Run-time Arguments for kernels
    vector<uint32_t> requestor_runtime_args = sub_worker_coordinates;
    for (auto& mst_logical_core : corerange_to_cores(mst_logical_core_set)) {
        SetRuntimeArgs(program, requestor_kernel, mst_logical_core, requestor_runtime_args);
    }

    // Assign unique id
    log_info(LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    /* ================ RUNNING THE PROGRAM ================ */

    // Setting up Inputs and Golden Output
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f,
        100.0f,
        bytes_per_transaction / sizeof(bfloat16),
        chrono::system_clock::now().time_since_epoch().count());

    vector<uint32_t> packed_golden = packed_input;

    for (auto& sub_logical_core : corerange_to_cores(sub_logical_core_set)) {
        detail::WriteToDeviceL1(device, sub_logical_core, sub_l1_base_address, packed_input);
        MetalContext::instance().get_cluster().l1_barrier(device->id());
    }

    // LAUNCH PROGRAM - Use mesh workload approach
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices =
        distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));  // Single device at (0,0)
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    vector<uint32_t> packed_output;

    bool is_equal = false;

    for (auto& mst_logical_core : corerange_to_cores(mst_logical_core_set)) {
        detail::ReadFromDeviceL1(device, mst_logical_core, mst_l1_base_address, bytes_per_transaction, packed_output);

        // Results comparison
        is_equal = (packed_output == packed_golden);
        if (!is_equal) {
            log_error(LogTest, "Equality Check failed");  // TO-DO: Print the failed core's coordinates here
            log_info(LogTest, "Golden vector");
            print_vector<uint32_t>(packed_golden);
            log_info(LogTest, "Output vector");
            print_vector<uint32_t>(packed_output);
            return is_equal;
        }
    }

    return is_equal;
}

void directed_ideal_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    CoreCoord mst_start_coord,
    CoreCoord sub_start_coord,
    CoreCoord mst_grid_size,
    CoreCoord sub_grid_size) {
    NOC noc_id = NOC::NOC_1;

    // Physical Constraints
    auto [bytes_per_page, max_reservable_bytes, max_reservable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);
    /* Running the Test */

    uint32_t num_of_transactions_per_subordinate = 1;
    uint32_t pages_reservable_per_transaction =
        max_reservable_pages / num_of_transactions_per_subordinate / 2;  // Half for master and subordinate

    // Test config
    unit_tests::dm::all_from_all::AllFromAllConfig test_config = {

        .test_id = test_case_id,

        .mst_logical_start_coord = mst_start_coord,
        .sub_logical_start_coord = sub_start_coord,
        .mst_grid_size = mst_grid_size,
        .sub_grid_size = sub_grid_size,

        .num_of_transactions_per_subordinate = num_of_transactions_per_subordinate,
        .pages_reservable_per_transaction = pages_reservable_per_transaction,
        .bytes_per_page = bytes_per_page,

        .l1_data_format = DataFormat::Float16_b,
        .noc_id = noc_id,
    };

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

void packet_sizes_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    CoreCoord mst_start_coord,
    CoreCoord sub_start_coord,
    CoreCoord mst_grid_size,
    CoreCoord sub_grid_size) {
    NOC noc_id = NOC::NOC_1;

    auto [bytes_per_page, max_reservable_bytes, max_reservable_pages] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    /* Running the Test */

    uint32_t max_transactions_per_subordinate = 256;
    uint32_t max_reservable_pages_per_transaction = mesh_device->impl().get_device(0)->arch() == ARCH::BLACKHOLE
                                                        ? 1024
                                                        : 2048;  // Max total transaction size == 64 KB

    for (uint32_t num_of_transactions_per_subordinate = 1;
         num_of_transactions_per_subordinate <= max_transactions_per_subordinate;
         num_of_transactions_per_subordinate *= 4) {
        for (uint32_t pages_reservable_per_transaction = 1;
             pages_reservable_per_transaction <= max_reservable_pages_per_transaction;
             pages_reservable_per_transaction *= 2) {
            // Check if the total data size is within the limits
            if (pages_reservable_per_transaction > max_reservable_pages) {
                continue;
            }

            // Test config
            unit_tests::dm::all_from_all::AllFromAllConfig test_config = {

                .test_id = test_case_id,

                .mst_logical_start_coord = mst_start_coord,
                .sub_logical_start_coord = sub_start_coord,
                .mst_grid_size = mst_grid_size,
                .sub_grid_size = sub_grid_size,

                .num_of_transactions_per_subordinate = num_of_transactions_per_subordinate,
                .pages_reservable_per_transaction = pages_reservable_per_transaction,
                .bytes_per_page = bytes_per_page,

                .l1_data_format = DataFormat::Float16_b,
                .noc_id = noc_id,
            };

            // Run
            EXPECT_TRUE(run_dm(mesh_device, test_config));
        }
    }
}

void virtual_channels_test(const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t test_case_id) {
    IDevice* device = mesh_device->impl().get_device(0);
    // Physical Constraints
    auto [bytes_per_page, max_bytes_reservable, max_pages_reservable] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters for literal all-to-all (use the full grid for both master and subordinate)
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

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
                if (pages_per_transaction > max_pages_reservable) {
                    continue;
                }

                // Test config
                AllFromAllConfig test_config = {
                    .test_id = test_case_id,
                    .mst_logical_start_coord = mst_start_coord,
                    .sub_logical_start_coord = sub_start_coord,
                    .mst_grid_size = mst_grid_size,
                    .sub_grid_size = sub_grid_size,
                    .num_of_transactions_per_subordinate = num_of_transactions,
                    .pages_reservable_per_transaction = pages_per_transaction,
                    .bytes_per_page = bytes_per_page,
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
    uint32_t test_case_id,
    CoreCoord mst_start_coord,
    CoreCoord sub_start_coord,
    CoreCoord mst_grid_size,
    CoreCoord sub_grid_size,
    uint32_t num_of_transactions_per_subordinate,
    uint32_t pages_per_transaction,
    uint32_t num_virtual_channels) {
    // Physical Constraints
    auto [bytes_per_page, max_bytes_reservable, max_pages_reservable] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    if (pages_per_transaction > max_pages_reservable) {
        log_trace(LogTest, "Skipping test due to page size limitations");
        return;
    }

    // Test config
    AllFromAllConfig test_config = {
        .test_id = test_case_id,
        .mst_logical_start_coord = mst_start_coord,
        .sub_logical_start_coord = sub_start_coord,
        .mst_grid_size = mst_grid_size,
        .sub_grid_size = sub_grid_size,
        .num_of_transactions_per_subordinate = num_of_transactions_per_subordinate,
        .pages_reservable_per_transaction = pages_per_transaction,
        .bytes_per_page = bytes_per_page,
        .l1_data_format = DataFormat::Float16_b,
        .noc_id = NOC::NOC_0,
        .num_virtual_channels = num_virtual_channels,
    };

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

}  // namespace unit_tests::dm::all_from_all

/* =============================================================  /
/  ========== TEST CASES FOR ALL-TO-ALL DATA MOVEMENT ==========  /
/  ============================================================= */

/*
TO-DO:
    - Implement a test case that shuffles through several grid sizes to test grids of different sizes
*/

/* ======== DIRECTED IDEAL ======== */

TEST_F(GenericMeshDeviceFixture, TensixDataMovementAllFromAllDirectedIdeal) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);
    uint32_t test_case_id = 310;

    /* Parameters */

    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {0, 0};

    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::all_from_all::directed_ideal_test(
        mesh_device, test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

/* ======== PACKET SIZES ======== */

TEST_F(GenericMeshDeviceFixture, TensixDataMovementAllFromAllPacketSizes) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    uint32_t test_case_id = 311;

    /* Parameters */
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {0, 0};

    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::all_from_all::packet_sizes_test(
        mesh_device, test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

/* ======== 2x2 to 1x1 ======== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementAllFromAll2x2From1x1DirectedIdeal) {
    uint32_t test_case_id = 312;

    /* Parameters */
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {4, 4};

    CoreCoord mst_grid_size = {2, 2};
    CoreCoord sub_grid_size = {1, 1};

    unit_tests::dm::all_from_all::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

/* ======== 4x4 to 1x1 ======== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementAllFromAll4x4From1x1DirectedIdeal) {
    uint32_t test_case_id = 313;

    /* Parameters */
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {0, 0};

    CoreCoord mst_grid_size = {4, 4};
    CoreCoord sub_grid_size = {1, 1};

    unit_tests::dm::all_from_all::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

/* ======== 1x1 to 2x2 ======== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementAllFromAll1x1From2x2DirectedIdeal) {
    uint32_t test_case_id = 314;

    /* Parameters */
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {4, 4};

    CoreCoord mst_grid_size = {1, 1};
    CoreCoord sub_grid_size = {2, 2};

    unit_tests::dm::all_from_all::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

/* ======== 1x1 to 4x4 ======== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementAllFromAll1x1From4x4DirectedIdeal) {
    uint32_t test_case_id = 315;

    /* Parameters */
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {0, 0};

    CoreCoord mst_grid_size = {1, 1};
    CoreCoord sub_grid_size = {4, 4};

    unit_tests::dm::all_from_all::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

/* ======== 2x2 to 2x2 ======== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementAllFromAll2x2From2x2DirectedIdeal) {
    uint32_t test_case_id = 316;

    /* Parameters */
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {0, 0};

    CoreCoord mst_grid_size = {2, 2};
    CoreCoord sub_grid_size = {2, 2};

    unit_tests::dm::all_from_all::directed_ideal_test(
        get_mesh_device(), test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

/* ======== VIRTUAL CHANNELS ======== */

TEST_F(GenericMeshDeviceFixture, TensixDataMovementAllFromAllVirtualChannels) {
    GTEST_SKIP() << "Skipping test";
    uint32_t test_case_id = 317;

    unit_tests::dm::all_from_all::virtual_channels_test(get_mesh_device(), test_case_id);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementAllFromAllCustom) {
    GTEST_SKIP() << "Skipping test";
    uint32_t test_case_id = 318;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    // Parameters
    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {0, 0};
    CoreCoord mst_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    uint32_t num_of_transactions_per_subordinate = 256;
    uint32_t pages_per_transaction = 1;
    uint32_t num_virtual_channels = 4;

    unit_tests::dm::all_from_all::custom_test(
        mesh_device,
        test_case_id,
        mst_start_coord,
        sub_start_coord,
        mst_grid_size,
        sub_grid_size,
        num_of_transactions_per_subordinate,
        pages_per_transaction,
        num_virtual_channels);
}

}  // namespace tt::tt_metal
