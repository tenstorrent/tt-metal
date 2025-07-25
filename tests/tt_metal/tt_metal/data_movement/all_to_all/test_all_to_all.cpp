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

namespace unit_tests::dm::all_to_all {

constexpr uint32_t START_ID = 60;

// Test Config (i.e. test parameters)
struct AllToAllConfig {
    /* Test ID */
    uint32_t test_id = START_ID;

    /* Grid configurations */
    CoreCoord mst_logical_start_coord = CoreCoord();
    CoreCoord sub_logical_start_coord = CoreCoord();
    CoreCoord mst_grid_size = CoreCoord();
    CoreCoord sub_grid_size = CoreCoord();

    /* Transaction size configurations */
    uint32_t num_of_transactions_per_master = 1;
    uint32_t pages_reservable_per_transaction = 1;
    uint32_t bytes_per_page = 32;

    /* Write configurations */
    DataFormat l1_data_format = DataFormat::Invalid;
    NOC noc_id = NOC::NOC_0;

    // TODO: Add the following parameters
    //  1. Virtual Channel (only useful for unicast)
    //  2. Posted flag (posted multicast has much better performance at larger grid sizes, than non-posted due to
    //  response packets) (60, 45, 23, vs 60, 60, 60 at posted)
};

/// @brief Performs communication from L1 Sender Cores to L1 Receiver Cores.
/// @param device The device on which the test is executed.
/// @param test_config Configuration of the test, defined by a specific struct.
/// @return Status of the test execution (e.g., success or failure).
bool run_dm(IDevice* device, const AllToAllConfig& test_config) {
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

    L1AddressInfo core_l1_info = unit_tests::dm::get_l1_address_and_size(device, {0, 0});
    uint32_t mst_l1_base_address = core_l1_info.base_address;
    uint32_t sub_l1_base_address = mst_l1_base_address + bytes_per_transaction;

    // Possible To-Do: Implement checks to see that the needed space is available in all master and subordinate cores

    // Kernels

    // Compile-time arguments for kernels

    vector<uint32_t> sender_compile_args = {
        //     0: Test ID
        (uint32_t)test_config.test_id,
        // 1 - 2: L1 Addresses
        (uint32_t)mst_l1_base_address,
        (uint32_t)sub_l1_base_address,
        // 3 - 4: Transaction parameters
        (uint32_t)test_config.num_of_transactions_per_master,  // num_of_transactions
        (uint32_t)bytes_per_transaction,                       // transaction_size_bytes
        //     5: Subordinate count
        (uint32_t)num_subordinates,  // num_subordinates
    };

    // Create kernels
    auto sender_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/all_to_all/kernels/sender.cpp",
        mst_logical_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = test_config.noc_id,
            .compile_args = sender_compile_args});

    // Run-time Arguments for kernels
    vector<uint32_t> sender_runtime_args = sub_worker_coordinates;
    for (auto& mst_logical_core : corerange_to_cores(mst_logical_core_set)) {
        SetRuntimeArgs(program, sender_kernel, mst_logical_core, sender_runtime_args);
    }

    // Assign unique id
    log_info(LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    /* ================ RUNNING THE PROGRAM ================ */

    // Setting up Inputs and Golden Output

    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f,
        100.0f,
        bytes_per_transaction / bfloat16::SIZEOF,
        chrono::system_clock::now().time_since_epoch().count());

    vector<uint32_t> packed_golden = packed_input;

    // Generate random input data for each master core
    for (auto& mst_logical_core : corerange_to_cores(mst_logical_core_set)) {
        detail::WriteToDeviceL1(device, mst_logical_core, mst_l1_base_address, packed_input);
        MetalContext::instance().get_cluster().l1_barrier(device->id());
    }

    // LAUNCH PROGRAM
    detail::LaunchProgram(device, program);

    vector<uint32_t> packed_output;
    packed_output.reserve(bytes_per_transaction / sizeof(uint32_t));

    bool pcc = false;

    for (auto& sub_logical_core : corerange_to_cores(sub_logical_core_set)) {
        detail::ReadFromDeviceL1(device, sub_logical_core, sub_l1_base_address, bytes_per_transaction, packed_output);

        // Results comparison
        pcc = is_close_packed_vectors<bfloat16, uint32_t>(
            packed_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });
        if (!pcc) {
            log_error(tt::LogTest, "PCC Check failed");  // TO-DO: Print the failed core's coordinates here
            log_info(tt::LogTest, "Golden vector");
            print_vector<uint32_t>(packed_golden);
            log_info(tt::LogTest, "Output vector");
            print_vector<uint32_t>(packed_output);
            return pcc;
        }
    }

    return pcc;
}

void packet_sizes_test(
    tt::ARCH arch_,
    std::vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_case_id,
    CoreCoord mst_start_coord,
    CoreCoord sub_start_coord,
    CoreCoord mst_grid_size,
    CoreCoord sub_grid_size) {
    NOC noc_id = NOC::NOC_0;

    auto [bytes_per_page, max_reservable_bytes, max_reservable_pages] =
        unit_tests::dm::compute_physical_constraints(arch_, devices_.at(0));

    /* Running the Test */

    uint32_t max_transactions_per_master = 256;
    uint32_t max_reservable_pages_per_transaction =
        arch_ == tt::ARCH::BLACKHOLE ? 1024 : 2048;  // Max total transaction size == 64 KB

    for (uint32_t num_of_transactions_per_master = 1; num_of_transactions_per_master <= max_transactions_per_master;
         num_of_transactions_per_master *= 4) {
        for (uint32_t pages_reservable_per_transaction = 1;
             pages_reservable_per_transaction <= max_reservable_pages_per_transaction;
             pages_reservable_per_transaction *= 2) {
            // Check if the total data size is within the limits
            if (pages_reservable_per_transaction > max_reservable_pages) {
                continue;
            }

            // Test config
            unit_tests::dm::all_to_all::AllToAllConfig test_config = {

                .test_id = unit_tests::dm::all_to_all::START_ID + test_case_id,

                .mst_logical_start_coord = mst_start_coord,
                .sub_logical_start_coord = sub_start_coord,
                .mst_grid_size = mst_grid_size,
                .sub_grid_size = sub_grid_size,

                .num_of_transactions_per_master = num_of_transactions_per_master,
                .pages_reservable_per_transaction = pages_reservable_per_transaction,
                .bytes_per_page = bytes_per_page,

                .l1_data_format = DataFormat::Float16_b,
                .noc_id = noc_id,
            };

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

void directed_ideal_test(
    tt::ARCH arch_,
    std::vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_case_id,
    CoreCoord mst_start_coord,
    CoreCoord sub_start_coord,
    CoreCoord mst_grid_size,
    CoreCoord sub_grid_size) {
    NOC noc_id = NOC::NOC_0;

    // Physical Constraints
    auto [bytes_per_page, max_reservable_bytes, max_reservable_pages] =
        unit_tests::dm::compute_physical_constraints(arch_, devices_.at(0));
    /* Running the Test */

    uint32_t num_of_transactions_per_master = 1;
    uint32_t pages_reservable_per_transaction =
        max_reservable_pages / num_of_transactions_per_master / 2;  // Half for master and subordinate

    // Test config
    unit_tests::dm::all_to_all::AllToAllConfig test_config = {

        .test_id = unit_tests::dm::all_to_all::START_ID + test_case_id,

        .mst_logical_start_coord = mst_start_coord,
        .sub_logical_start_coord = sub_start_coord,
        .mst_grid_size = mst_grid_size,
        .sub_grid_size = sub_grid_size,

        .num_of_transactions_per_master = num_of_transactions_per_master,
        .pages_reservable_per_transaction = pages_reservable_per_transaction,
        .bytes_per_page = bytes_per_page,

        .l1_data_format = DataFormat::Float16_b,
        .noc_id = noc_id,
    };

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

}  // namespace unit_tests::dm::all_to_all

/* =============================================================  /
/  ========== TEST CASES FOR ALL-TO-ALL DATA MOVEMENT ==========  /
/  ============================================================= */

/*
IDEAS:
    - Implement a for loop that shuffles through several coordinates to test grids of
        different locations
    - Implement a for loop that shuffles through several grid sizes to test grids of
        different sizes
*/

/* ======== PACKET SIZES ======== */

TEST_F(DeviceFixture, TensixDataMovementAllToAllPacketSizes) {
    if (arch_ == tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Skipping test on Blackhole, Issue #24584";
    }
    uint32_t test_case_id = 0;

    /* Parameters */

    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {0, 0};

    CoreCoord mst_grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};
    CoreCoord sub_grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};

    tt::tt_metal::unit_tests::dm::all_to_all::packet_sizes_test(
        arch_, devices_, num_devices_, test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

/* ======== DIRECTED IDEAL ======== */

/* ======== All to All ======== */
TEST_F(DeviceFixture, TensixDataMovementAllToAllDirectedIdeal) {
    if (arch_ == tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Skipping test on Blackhole, Issue #24584";
    }
    uint32_t test_case_id = 1;

    /* Parameters */

    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {0, 0};

    CoreCoord mst_grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};
    CoreCoord sub_grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};

    unit_tests::dm::all_to_all::directed_ideal_test(
        arch_, devices_, num_devices_, test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

/* ======== 2x2 to 1x1 ======== */
TEST_F(DeviceFixture, TensixDataMovementAllToAll2x2To1x1DirectedIdeal) {
    uint32_t test_case_id = 2;

    /* Parameters */

    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {4, 4};

    CoreCoord mst_grid_size = {2, 2};
    CoreCoord sub_grid_size = {1, 1};

    unit_tests::dm::all_to_all::directed_ideal_test(
        arch_, devices_, num_devices_, test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

/* ======== 4x4 to 1x1 ======== */
TEST_F(DeviceFixture, TensixDataMovementAllToAll4x4To1x1DirectedIdeal) {
    uint32_t test_case_id = 3;

    /* Parameters */

    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {0, 0};

    CoreCoord mst_grid_size = {4, 4};
    CoreCoord sub_grid_size = {1, 1};

    unit_tests::dm::all_to_all::directed_ideal_test(
        arch_, devices_, num_devices_, test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

/* ======== 1x1 to 2x2 ======== */
TEST_F(DeviceFixture, TensixDataMovementAllToAll1x1To2x2DirectedIdeal) {
    uint32_t test_case_id = 4;

    /* Parameters */

    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {4, 4};

    CoreCoord mst_grid_size = {1, 1};
    CoreCoord sub_grid_size = {2, 2};

    unit_tests::dm::all_to_all::directed_ideal_test(
        arch_, devices_, num_devices_, test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

/* ======== 1x1 to 4x4 ======== */
TEST_F(DeviceFixture, TensixDataMovementAllToAll1x1To4x4DirectedIdeal) {
    uint32_t test_case_id = 5;

    /* Parameters */

    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {0, 0};

    CoreCoord mst_grid_size = {1, 1};
    CoreCoord sub_grid_size = {4, 4};

    unit_tests::dm::all_to_all::directed_ideal_test(
        arch_, devices_, num_devices_, test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

/* ======== 2x2 to 2x2 ======== */
TEST_F(DeviceFixture, TensixDataMovementAllToAll2x2To2x2DirectedIdeal) {
    uint32_t test_case_id = 6;

    /* Parameters */

    CoreCoord mst_start_coord = {0, 0};
    CoreCoord sub_start_coord = {0, 0};

    CoreCoord mst_grid_size = {2, 2};
    CoreCoord sub_grid_size = {2, 2};

    unit_tests::dm::all_to_all::directed_ideal_test(
        arch_, devices_, num_devices_, test_case_id, mst_start_coord, sub_start_coord, mst_grid_size, sub_grid_size);
}

}  // namespace tt::tt_metal
