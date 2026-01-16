// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Note: The sender kernels in One To All write the same transaction_size_bytes amount of data to the same location
// num_of_transactions times

#include "multi_device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"
#include "test_one_to_all.hpp"
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace test_utils;

namespace unit_tests::dm::core_to_all {

// Test config, i.e. test parameters
struct OneToAllConfig {
    uint32_t test_id = 0;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {0, 0};

    uint32_t num_of_transactions = 0;
    uint32_t pages_per_transaction = 0;
    uint32_t bytes_per_page = 0;

    DataFormat l1_data_format = DataFormat::Invalid;
    NOC noc_id = NOC::NOC_0;

    bool loopback = false;
    bool is_multicast = false;
    bool is_linked = false;

    uint32_t multicast_scheme_type = 0;
    uint32_t num_virtual_channels = 1;  // Number of virtual channels to cycle through (only useful for unicast)
    bool use_2_0_api = false;           // Use Device 2.0 API

    // TODO: Add the following parameters
    //  1. Posted flag (posted multicast has much better performance at larger grid sizes, than non-posted due to
    //  response packets) (60, 45, 23, vs 60, 60, 60 at posted)
};

bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const OneToAllConfig& test_config) {
    IDevice* device = mesh_device->impl().get_device(0);
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
    L1AddressInfo mst_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, test_config.mst_core_coord);
    // Check if the L1 size is sufficient for the test configuration
    if (mst_l1_info.size < bytes_per_transaction) {
        log_error(LogTest, "Insufficient L1 size for the test configuration");
        return false;
    }

    // Validate virtual channels configuration
    if (test_config.num_virtual_channels > 4) {
        log_error(
            LogTest,
            "num_virtual_channels must not be greater than 4 as there are only 4 unicast write virtual channels");
        return false;
    }

    // Checks that both master and all subordinate cores have the same L1 base address and size
    for (auto& sub_logical_core : sub_core_list) {
        L1AddressInfo sub_l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, sub_logical_core);
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
            {(uint32_t)test_config.is_linked,
             (uint32_t)test_config.loopback,
             (uint32_t)sub_worker_start_coord.x,
             (uint32_t)sub_worker_start_coord.y,
             (uint32_t)sub_worker_end_coord.x,
             (uint32_t)sub_worker_end_coord.y,
             (uint32_t)test_config.multicast_scheme_type,
             (uint32_t)test_config.sub_grid_size.x,
             (uint32_t)test_config.sub_grid_size.y});

        sender_kernel_path += "sender_multicast";

    } else {  // Unicast Sender Kernel
        sender_compile_args.push_back((uint32_t)test_config.num_virtual_channels);
        sender_kernel_path += "sender_unicast";
    }

    if (test_config.use_2_0_api) {
        sender_kernel_path += "_2_0";
    }
    sender_kernel_path += ".cpp";

    DataMovementProcessor data_movement_processor = DataMovementProcessor::RISCV_0;
    auto sender_kernel = CreateKernel(
        program,
        sender_kernel_path,
        mst_logical_core_set,
        DataMovementConfig{
            .processor = data_movement_processor, .noc = test_config.noc_id, .compile_args = sender_compile_args});

    // Runtime Arguments
    vector<uint32_t> sender_runtime_args = {};

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
    size_t element_size_bytes = sizeof(bfloat16);
    uint32_t num_elements = bytes_per_transaction / element_size_bytes;
    vector<uint32_t> packed_input = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, num_elements, chrono::system_clock::now().time_since_epoch().count());
    vector<uint32_t> packed_golden = packed_input;

    // Write input to master L1 buffer
    detail::WriteToDeviceL1(device, test_config.mst_core_coord, mst_l1_base_address, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    // LAUNCH THE PROGRAM
    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));

    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    // Read output from subordinate L1 buffers (implement a loop)
    vector<uint32_t> packed_output;

    for (auto& sub_logical_core : sub_core_list) {
        detail::ReadFromDeviceL1(device, sub_logical_core, sub_l1_base_address, bytes_per_transaction, packed_output);

        // Results comparison
        bool is_equal = (packed_output == packed_golden);

        if (!is_equal) {
            log_error(LogTest, "Equality Check failed");
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
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    bool is_multicast,
    bool is_linked,
    CoreCoord mst_core_coord,
    CoreCoord sub_start_core_coord,
    CoreCoord sub_grid_size,
    bool loopback,
    NOC noc_id,
    uint32_t multicast_scheme_type,
    bool use_2_0_api) {
    // Physical Constraints
    auto [bytes_per_page, max_bytes_reservable, max_pages_reservable] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

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
        .num_virtual_channels = 1,
        .use_2_0_api = use_2_0_api,
    };

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

void packet_sizes_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    bool is_multicast,
    bool is_linked,
    CoreCoord mst_core_coord,
    CoreCoord sub_start_core_coord,
    CoreCoord sub_grid_size,
    bool use_2_0_api = false) {
    // Parameters
    NOC noc_id = NOC::NOC_0;
    auto [bytes_per_page, max_bytes_reservable, max_pages_reservable] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    /* Running the Test */

    uint32_t max_transactions = 256;
    uint32_t max_pages_reservable_per_transaction = mesh_device->impl().get_device(0)->arch() == ARCH::BLACKHOLE
                                                        ? 1024
                                                        : 2048;  // Max total transaction size == 64 KB

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
                    .use_2_0_api = use_2_0_api};

                // Run
                EXPECT_TRUE(run_dm(mesh_device, test_config));
            }
        }
    }
}

void virtual_channels_test(
    const shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_case_id,
    bool is_multicast,
    bool is_linked,
    CoreCoord mst_core_coord,
    CoreCoord sub_start_core_coord,
    CoreCoord sub_grid_size,
    bool loopback) {
    // Virtual channels are only meaningful for unicast
    if (is_multicast) {
        log_info(LogTest, "Virtual channels test is only applicable for unicast, skipping multicast test");
        return;
    }

    // Physical Constraints
    auto [bytes_per_page, max_bytes_reservable, max_pages_reservable] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    // Parameters
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
                // Check if the total page size is within the limits (adjusted for loopback)
                if (pages_per_transaction > max_pages_reservable / (loopback ? 2 : 1)) {
                    continue;
                }

                // Test config - loopback is always true for virtual channels test
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
    bool is_multicast,
    bool is_linked,
    CoreCoord mst_core_coord,
    CoreCoord sub_start_core_coord,
    CoreCoord sub_grid_size,
    uint32_t num_of_transactions,
    uint32_t pages_per_transaction,
    uint32_t num_virtual_channels,
    bool loopback) {
    // Virtual channels are only meaningful for unicast
    if (is_multicast) {
        log_info(LogTest, "Virtual channels test is only applicable for unicast, skipping multicast test");
        return;
    }

    // Physical Constraints
    auto [bytes_per_page, max_bytes_reservable, max_pages_reservable] =
        unit_tests::dm::compute_physical_constraints(mesh_device);

    if (pages_per_transaction > max_pages_reservable / (loopback ? 2 : 1)) {
        log_trace(LogTest, "Skipping test due to page size limitations with loopback={}", loopback);
        return;
    }

    // Test config
    unit_tests::dm::core_to_all::OneToAllConfig test_config = {
        .test_id = test_case_id,
        .mst_core_coord = mst_core_coord,
        .sub_start_core_coord = sub_start_core_coord,
        .sub_grid_size = sub_grid_size,
        .num_of_transactions = num_of_transactions,
        .pages_per_transaction = pages_per_transaction,
        .bytes_per_page = bytes_per_page,
        .l1_data_format = DataFormat::Float16_b,
        .noc_id = NOC::NOC_0,
        .loopback = loopback,
        .is_multicast = is_multicast,
        .is_linked = is_linked,
        .num_virtual_channels = num_virtual_channels,
    };

    // Run
    EXPECT_TRUE(run_dm(mesh_device, test_config));
}

}  // namespace unit_tests::dm::core_to_all

/* =================================== */
/* =========== TEST CASES ============ */
/* =================================== */

/* ========== PACKET SIZES ========== */

/* ========== UNICAST ========== */

/* ========== 2x2 ========== */
TEST_F(GenericMeshDeviceFixture, NIGHTLY_TensixDataMovementOneToAllUnicast2x2PacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 0;

    bool is_multicast = false;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {2, 2};

    auto mesh_device = get_mesh_device();
    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size);
}

/* ========== 5x5 ========== */
TEST_F(GenericMeshDeviceFixture, NIGHTLY_TensixDataMovementOneToAllUnicast5x5PacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 1;

    bool is_multicast = false;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {5, 5};

    auto mesh_device = get_mesh_device();
    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size);
}

/* ========== All ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllUnicastPacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 2;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    bool is_multicast = false;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size);
}

/* ========== MULTICAST ========== */

/* ========== 2x2 ========== */
TEST_F(GenericMeshDeviceFixture, NIGHTLY_TensixDataMovementOneToAllMulticast2x2PacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 3;

    auto mesh_device = get_mesh_device();

    bool is_multicast = true;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {2, 2};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size);
}

/* ========== 5x5 ========== */
TEST_F(GenericMeshDeviceFixture, NIGHTLY_TensixDataMovementOneToAllMulticast5x5PacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 4;

    auto mesh_device = get_mesh_device();

    bool is_multicast = true;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {5, 5};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size);
}

/* ========== All ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastPacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 5;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    bool is_multicast = true;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size);
}

/* ========== MULTICAST LINKED ========== */

/* ========== 2x2 ========= */
TEST_F(GenericMeshDeviceFixture, NIGHTLY_TensixDataMovementOneToAllMulticastLinked2x2PacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 6;

    auto mesh_device = get_mesh_device();

    bool is_multicast = true;
    bool is_linked = true;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {2, 2};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size);
}

/* ========== 5x5 ========== */
TEST_F(GenericMeshDeviceFixture, NIGHTLY_TensixDataMovementOneToAllMulticastLinked5x5PacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 7;

    auto mesh_device = get_mesh_device();

    bool is_multicast = true;
    bool is_linked = true;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {5, 5};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size);
}

/* ========== 11x10 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastLinkedPacketSizes) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID + 8;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    bool is_multicast = true;
    bool is_linked = true;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size);
}

/* ========== DIRECTED IDEAL ========== */

/* ========== UNICAST ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllUnicastDirectedIdeal) {
    // Parameters
    uint32_t test_case_id = 52;  // Arbitrary test id

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    bool loopback = true;
    NOC noc_id = NOC::NOC_0;

    bool is_multicast = false;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::directed_ideal_test(
        mesh_device,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size,
        loopback,
        noc_id,
        0);  // multicast_scheme_type (not used for unicast)
}

/* ========== MULTICAST ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastDirectedIdeal) {
    // Parameters
    uint32_t test_case_id = 53;  // Arbitrary test id

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    bool loopback = true;
    NOC noc_id = NOC::NOC_0;

    bool is_multicast = true;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::directed_ideal_test(
        mesh_device,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size,
        loopback,
        noc_id,
        0);  // multicast_scheme_type (not used here)
}

/* ========== MULTICAST LINKED ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastLinkedDirectedIdeal) {
    // Parameters
    uint32_t test_case_id = 54;  // Arbitrary test id

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    bool loopback = true;
    NOC noc_id = NOC::NOC_0;

    bool is_multicast = true;
    bool is_linked = true;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::directed_ideal_test(
        mesh_device,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size,
        loopback,
        noc_id,
        0);  // multicast_scheme_type (not used here)
}

/* ========== VIRTUAL CHANNELS ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllUnicastVirtualChannels) {  // Expose loopback here?
    GTEST_SKIP() << "Skipping test";

    // Parameters
    uint32_t test_case_id = 154;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    // These should always be false
    bool is_multicast = false;
    bool is_linked = false;

    // Grid Parameters
    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    // Loopback
    bool loopback = true;

    unit_tests::dm::core_to_all::virtual_channels_test(
        mesh_device,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size,
        loopback);
}

TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllUnicastCustom) {
    GTEST_SKIP() << "Skipping test";

    // Parameters
    uint32_t test_case_id = 155;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    // These should always be false
    bool is_multicast = false;
    bool is_linked = false;

    // Grid Parameters
    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    // Custom Parameters
    bool loopback = true;
    uint32_t num_of_transactions = 256;
    uint32_t pages_per_transaction = 1;
    uint32_t num_virtual_channels = 4;

    unit_tests::dm::core_to_all::custom_test(
        mesh_device,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size,
        num_of_transactions,
        pages_per_transaction,
        num_virtual_channels,
        loopback);
}

/* ========== UNICAST 2.0 ========== */

/* ========== 2x2 ========== */
TEST_F(GenericMeshDeviceFixture, NIGHTLY_TensixDataMovementOneToAllUnicast2x2PacketSizes2_0) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID_2_0 + 0;

    bool is_multicast = false;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {2, 2};

    auto mesh_device = get_mesh_device();
    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size, true);
}

/* ========== 5x5 ========== */
TEST_F(GenericMeshDeviceFixture, NIGHTLY_TensixDataMovementOneToAllUnicast5x5PacketSizes2_0) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID_2_0 + 1;

    bool is_multicast = false;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {5, 5};

    auto mesh_device = get_mesh_device();
    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size, true);
}

/* ========== All ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllUnicastPacketSizes2_0) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID_2_0 + 2;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    bool is_multicast = false;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size, true);
}

/* ========== MULTICAST 2.0 ========== */

/* ========== 2x2 ========== */
TEST_F(GenericMeshDeviceFixture, NIGHTLY_TensixDataMovementOneToAllMulticast2x2PacketSizes2_0) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID_2_0 + 3;

    auto mesh_device = get_mesh_device();

    bool is_multicast = true;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {2, 2};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size, true);
}

/* ========== 5x5 ========== */
TEST_F(GenericMeshDeviceFixture, NIGHTLY_TensixDataMovementOneToAllMulticast5x5PacketSizes2_0) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID_2_0 + 4;

    auto mesh_device = get_mesh_device();

    bool is_multicast = true;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {5, 5};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size, true);
}

/* ========== All ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastPacketSizes2_0) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID_2_0 + 5;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    bool is_multicast = true;
    bool is_linked = false;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size, true);
}

/* ========== MULTICAST LINKED ========== */

/* ========== 2x2 ========= */
TEST_F(GenericMeshDeviceFixture, NIGHTLY_TensixDataMovementOneToAllMulticastLinked2x2PacketSizes2_0) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID_2_0 + 6;

    auto mesh_device = get_mesh_device();

    bool is_multicast = true;
    bool is_linked = true;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {2, 2};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size, true);
}

/* ========== 5x5 ========== */
TEST_F(GenericMeshDeviceFixture, NIGHTLY_TensixDataMovementOneToAllMulticastLinked5x5PacketSizes2_0) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID_2_0 + 7;

    auto mesh_device = get_mesh_device();

    bool is_multicast = true;
    bool is_linked = true;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {5, 5};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size, true);
}

/* ========== 11x10 ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastLinkedPacketSizes2_0) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID_2_0 + 8;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    bool is_multicast = true;
    bool is_linked = true;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::packet_sizes_test(
        mesh_device, test_case_id, is_multicast, is_linked, mst_core_coord, sub_start_core_coord, sub_grid_size, true);
}

/* ========== MULTICAST LINKED WITH LOOPBACK ========== */
TEST_F(GenericMeshDeviceFixture, TensixDataMovementOneToAllMulticastLinkedDirectedIdeal2_0) {
    // Parameters
    uint32_t test_case_id = unit_tests::dm::core_to_all::START_ID_2_0 + 9;

    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->impl().get_device(0);

    bool loopback = true;
    NOC noc_id = NOC::NOC_0;

    bool is_multicast = true;
    bool is_linked = true;

    CoreCoord mst_core_coord = {0, 0};
    CoreCoord sub_start_core_coord = {0, 0};
    CoreCoord sub_grid_size = {device->compute_with_storage_grid_size().x, device->compute_with_storage_grid_size().y};

    unit_tests::dm::core_to_all::directed_ideal_test(
        mesh_device,
        test_case_id,
        is_multicast,
        is_linked,
        mst_core_coord,
        sub_start_core_coord,
        sub_grid_size,
        loopback,
        noc_id,
        0,  // multicast_scheme_type (not used here)
        true);
}
}  // namespace tt::tt_metal
