// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Note: The sender kernels in One To All write the same transaction_size_bytes amount of data to the same location
// num_of_transactions times

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include "test_one_to_all.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::core_to_all {

constexpr uint32_t START_ID = 6;

// Test config, i.e. test parameters
struct OneToAllConfig {
    uint32_t test_id = 0;

    CoreCoord mst_core_coord = CoreCoord();
    CoreCoord sub_start_core_coord = CoreCoord();
    CoreCoord sub_grid_size = CoreCoord();

    uint32_t num_of_transactions = 0;
    uint32_t pages_per_transaction = 0;
    uint32_t bytes_per_page = 0;

    DataFormat l1_data_format = DataFormat::Invalid;
    NOC noc_id = NOC::NOC_0;

    bool loopback = false;
    bool is_multicast = false;
    bool is_linked = false;

    uint32_t multicast_scheme_type = 0;

    // TODO: Add the following parameters
    //  1. Virtual Channel (only useful for unicast)
    //  2. Posted flag (posted multicast has much better performance at larger grid sizes, than non-posted due to
    //  response packets) (60, 45, 23, vs 60, 60, 60 at posted)
};

bool run_dm(IDevice* device, const OneToAllConfig& test_config) {
    /* ================ SETUP ================ */

    // Program
    Program program = CreateProgram();

    // assert(
    //    (test_config.is_multicast && test_config.loopback) ||
    //    (!test_config.is_multicast && !test_config.is_linked));

    // Parameters
    const size_t bytes_per_transaction = test_config.pages_per_transaction * test_config.bytes_per_page;

    if (test_config.loopback && (bytes_per_transaction > 1024 * 1024 / 2)) {
        log_error(LogTest, "Not enough memory for master core using loopback");
        return false;
    }

    // Initialize core sets

    // Master Logical
    CoreRangeSet mst_logical_core_set({CoreRange(test_config.mst_core_coord)});

    // Subordinate Logical
    CoreCoord sub_logical_start_coord = test_config.sub_start_core_coord;
    CoreCoord sub_logical_end_coord = CoreCoord(
        sub_logical_start_coord.x + test_config.sub_grid_size.x - 1,
        sub_logical_start_coord.y + test_config.sub_grid_size.y - 1);
    // POSSIBLE TO-DO: Add a check to ensure that the sub_logical_end_coord is within the device's logical core range
    CoreRangeSet sub_logical_core_set({CoreRange(sub_logical_start_coord, sub_logical_end_coord)});
    if (!test_config.loopback) {
        sub_logical_core_set = sub_logical_core_set.subtract(mst_logical_core_set);
    }
    uint32_t num_subordinates = sub_logical_core_set.num_cores();
    auto sub_core_list = corerange_to_cores(sub_logical_core_set);

    // Subordinate Physical (only needed for unicast)
    CoreCoord sub_worker_start_coord = device->worker_core_from_logical_core(sub_logical_start_coord);
    CoreCoord sub_worker_end_coord = device->worker_core_from_logical_core(sub_logical_end_coord);
    vector<uint32_t> sub_worker_coordinates = {};
    for (auto& sub_logical_core : sub_core_list) {
        CoreCoord sub_worker_core = device->worker_core_from_logical_core(sub_logical_core);
        uint32_t sub_worker_core_packed =
            (sub_worker_core.x << 16) | (sub_worker_core.y & 0xFFFF);  // Pack coordinates into a single uint32_t
        sub_worker_coordinates.push_back(sub_worker_core_packed);
    }

    // L1 Space Allocation

    // Obtain L1 Address for Storing Data
    L1AddressInfo mst_l1_info = unit_tests::dm::get_l1_address_and_size(device, test_config.mst_core_coord);
    // Check if the L1 size is sufficient for the test configuration
    if (mst_l1_info.size < bytes_per_transaction) {
        log_error(LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }

    // Checks that both master and all subordinate cores have the same L1 base address and size
    for (auto& sub_logical_core : sub_core_list) {
        L1AddressInfo sub_l1_info = unit_tests::dm::get_l1_address_and_size(device, sub_logical_core);
        if (mst_l1_info.base_address != sub_l1_info.base_address || mst_l1_info.size != sub_l1_info.size) {
            log_error(LogTest, "Mismatch in L1 address or size between master and subordinate cores");
            return false;
        }
    }

    // Assigns an L1 local address for the master and subordinate cores
    // An offset if needed for the subordinate L1 base address if loopback is enabled,
    // as both blocks must be distinct in that case to avoid overwriting data
    uint32_t mst_l1_base_address = mst_l1_info.base_address;
    uint32_t sub_l1_base_address =
        test_config.loopback ? mst_l1_base_address : mst_l1_base_address + (bytes_per_transaction);

    // Initialize Kernels

    // Sender Kernel
    vector<uint32_t> sender_compile_args = {
                                            (uint32_t)mst_l1_base_address,
                                            (uint32_t)sub_l1_base_address,
                                            (uint32_t)test_config.num_of_transactions,
                                            (uint32_t)test_config.pages_per_transaction,
                                            (uint32_t)test_config.bytes_per_page,
                                            (uint32_t)test_config.test_id,
                                            (uint32_t)num_subordinates};
    string sender_kernel_path = "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/";

    if (test_config.is_multicast) {  // Multicast Sender Kernel
        sender_compile_args.insert(
            sender_compile_args.end(),
            {
             (uint32_t)test_config.is_linked,
             (uint32_t)test_config.loopback,
             (uint32_t)sub_worker_start_coord.x,
             (uint32_t)sub_worker_start_coord.y,
             (uint32_t)sub_worker_end_coord.x,
             (uint32_t)sub_worker_end_coord.y,
             (uint32_t)test_config.multicast_scheme_type,
             (uint32_t)test_config.sub_grid_size.x,
             (uint32_t)test_config.sub_grid_size.y});

        sender_kernel_path += "sender_multicast.cpp";

    } else {  // Unicast Sender Kernel
        sender_kernel_path += "sender_unicast.cpp";
    }

    DataMovementProcessor data_movement_processor = DataMovementProcessor::RISCV_0;
    auto sender_kernel = CreateKernel(
        program,
        sender_kernel_path,
        mst_logical_core_set,
        DataMovementConfig{
            .processor = data_movement_processor, .noc = test_config.noc_id, .compile_args = sender_compile_args});

    // Runtime Arguments
    std::vector<uint32_t> sender_runtime_args = {};

    if (!test_config.is_multicast) {  // Unicast Sender Runtime Arguments
        sender_runtime_args.insert(
            sender_runtime_args.end(), sub_worker_coordinates.begin(), sub_worker_coordinates.end());
    }

    SetRuntimeArgs(program, sender_kernel, mst_logical_core_set, sender_runtime_args);

    // Assign unique id

    log_info(LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    /* ================ EXECUTION ================ */

    // Setup Input and Golden Output
    size_t element_size_bytes = bfloat16::SIZEOF;
    uint32_t num_elements = bytes_per_transaction / element_size_bytes;
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, num_elements, chrono::system_clock::now().time_since_epoch().count());
    vector<uint32_t> packed_golden = packed_input;

    // Write input to master L1 buffer
    detail::WriteToDeviceL1(device, test_config.mst_core_coord, mst_l1_base_address, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    // LAUNCH THE PROGRAM
    detail::LaunchProgram(device, program);

    // Read output from subordinate L1 buffers (implement a loop)
    vector<uint32_t> packed_output;

    for (auto& sub_logical_core : sub_core_list) {
        detail::ReadFromDeviceL1(device, sub_logical_core, sub_l1_base_address, bytes_per_transaction, packed_output);

        // Results comparison
        bool pcc = is_close_packed_vectors<bfloat16, uint32_t>(
            packed_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });

        if (!pcc) {
            log_error(LogTest, "PCC Check failed");
            log_info(LogTest, "Golden vector");
            print_vector<uint32_t>(packed_golden);
            log_info(LogTest, "Output vector");
            print_vector<uint32_t>(packed_output);
            return false;
        }
    }
    return true;
}

/* TEST TYPES */

void directed_ideal_test(
    ARCH arch_,
    vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_case_id,
    bool is_multicast,
    bool is_linked,
    CoreCoord mst_core_coord,
    CoreCoord sub_start_core_coord,
    CoreCoord sub_grid_size,
    bool loopback,
    NOC noc_id,
    uint32_t multicast_scheme_type) {
    // Physical Constraints
    auto [bytes_per_page, max_bytes_reservable, max_pages_reservable] =
        unit_tests::dm::compute_physical_constraints(arch_, devices_.at(0));

    if (loopback) {
        max_pages_reservable /= 2;  // Loopback uses half of the memory
    }

    // Adjustable Parameters (Ideal: Less transactions, more data per transaction)
    uint32_t pages_per_transaction = 256;
    uint32_t num_of_transactions = 256;

    unit_tests::dm::core_to_all::OneToAllConfig test_config = {
        .test_id = test_case_id,
        .mst_core_coord = mst_core_coord,
        .sub_start_core_coord = sub_start_core_coord,
        .sub_grid_size = sub_grid_size,
        .num_of_transactions = num_of_transactions,
        .pages_per_transaction = pages_per_transaction,
        .bytes_per_page = bytes_per_page,
        .l1_data_format = DataFormat::Float16_b,
        .noc_id = noc_id,
        .loopback = loopback,
        .is_multicast = is_multicast,
        .is_linked = is_linked,
        .multicast_scheme_type = multicast_scheme_type,
    };

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

void packet_sizes_test(
    ARCH arch_,
    vector<IDevice*>& devices_,
    uint32_t num_devices_,
    uint32_t test_case_id,
    bool is_multicast,
    bool is_linked,
    CoreCoord mst_core_coord,
    CoreCoord sub_start_core_coord,
    CoreCoord sub_grid_size) {
    // Parameters
    NOC noc_id = NOC::NOC_0;
    auto [bytes_per_page, max_bytes_reservable, max_pages_reservable] =
        unit_tests::dm::compute_physical_constraints(arch_, devices_.at(0));

    /* Running the Test */

    uint32_t max_transactions = 256;
    uint32_t max_pages_reservable_per_transaction =
        arch_ == ARCH::BLACKHOLE ? 1024 : 2048;  // Max total transaction size == 64 KB

    for (bool loopback : {true, false}) {
        if (loopback) {
            max_pages_reservable /= 2;
        }

        for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
            for (uint32_t pages_reservable_per_transaction = 1;
                 pages_reservable_per_transaction <= max_pages_reservable_per_transaction;
                 pages_reservable_per_transaction *= 2) {
                // Check if the total data size is within the limits
                if (pages_reservable_per_transaction > max_pages_reservable) {
                    continue;
                }

                // Test config
                unit_tests::dm::core_to_all::OneToAllConfig test_config = {
                    .test_id = test_case_id,
                    .mst_core_coord = mst_core_coord,
                    .sub_start_core_coord = sub_start_core_coord,
                    .sub_grid_size = sub_grid_size,
                    .num_of_transactions = num_of_transactions,
                    .pages_per_transaction = pages_reservable_per_transaction,
                    .bytes_per_page = bytes_per_page,
                    .l1_data_format = DataFormat::Float16_b,
                    .noc_id = noc_id,
                    .loopback = loopback,
                    .is_multicast = is_multicast,
                    .is_linked = is_linked,
                };

                // Run
                for (unsigned int id = 0; id < num_devices_; id++) {
                    EXPECT_TRUE(run_dm(devices_.at(id), test_config));
                }
            }
        }
    }
}

}  // namespace unit_tests::dm::core_to_all

/* =================================== */
/* =========== TEST CASES ============ */
/* =================================== */

/* ========== PACKET SIZES ========== */

/* ========== UNICAST ========== */

/* ========== 2x2 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllUnicast2x2PacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 0;

    bool is_multicast = false;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {2, 2};

    unit_tests::dm::core_to_all::packet_sizes_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size);
}

/* ========== 5x5 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllUnicast5x5PacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 1;

    bool is_multicast = false;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {5, 5};

    unit_tests::dm::core_to_all::packet_sizes_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size);
}

/* ========== All ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllUnicastPacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 2;

    bool is_multicast = false;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::packet_sizes_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size);
}

/* ========== MULTICAST ========== */

/* ========== 2x2 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticast2x2PacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 3;

    bool is_multicast = true;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {2, 2};

    unit_tests::dm::core_to_all::packet_sizes_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size);
}

/* ========== 5x5 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticast5x5PacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 4;

    bool is_multicast = true;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {5, 5};

    unit_tests::dm::core_to_all::packet_sizes_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size);
}

/* ========== All ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastPacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 5;

    bool is_multicast = true;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::packet_sizes_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size);
}

/* ========== MULTICAST LINKED ========== */

/* ========== 2x2 ========= */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastLinked2x2PacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 6;

    bool is_multicast = true;
    bool is_linked = true;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {2, 2};

    unit_tests::dm::core_to_all::packet_sizes_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size);
}

/* ========== 5x5 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastLinked5x5PacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 7;

    bool is_multicast = true;
    bool is_linked = true;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {5, 5};

    unit_tests::dm::core_to_all::packet_sizes_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size);
}

/* ========== 11x10 ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastLinkedPacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 8;

    bool is_multicast = true;
    bool is_linked = true;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::packet_sizes_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size);
}

/* ========== DIRECTED IDEAL ========== */

/* ========== UNICAST ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllUnicastDirectedIdeal) {
    // Parameters
    uint32_t test_case_id = 52;  // Arbitrary test id

    bool loopback = true;
    NOC noc_id = NOC::NOC_1;

    bool is_multicast = false;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::directed_ideal_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size,
        loopback,
        noc_id);
}

/* ========== MULTICAST ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastDirectedIdeal) {
    // Parameters
    uint32_t test_case_id = 53;  // Arbitrary test id

    bool loopback = true;
    NOC noc_id = NOC::NOC_1;

    bool is_multicast = true;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::directed_ideal_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size,
        loopback,
        noc_id);
}

/* ========== MULTICAST LINKED ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastLinkedDirectedIdeal) {
    // Parameters
    uint32_t test_case_id = 54;  // Arbitrary test id

    bool loopback = true;
    NOC noc_id = NOC::NOC_1;

    bool is_multicast = true;
    bool is_linked = true;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::directed_ideal_test(
        arch_,
        devices_,
        num_devices_,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size,
        loopback,
        noc_id);
}

}  // namespace tt::tt_metal
