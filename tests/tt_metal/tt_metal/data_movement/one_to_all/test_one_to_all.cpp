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

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::core_to_all {

constexpr uint32_t START_ID = 6;

// Test config, i.e. test parameters
struct OneToAllConfig {
    uint32_t test_id = 0;
    CoreCoord master_core_coord = CoreCoord();
    CoreCoord grid_size = CoreCoord();
    uint32_t num_of_transactions = 0;
    uint32_t transaction_size_pages = 0;
    uint32_t page_size_bytes = 0;
    DataFormat l1_data_format = DataFormat::Invalid;
    bool loopback = false;
    NOC noc_id = NOC::NOC_0;
    bool is_multicast = false;
    bool is_linked = false;

    // TODO: Add the following parameters
    //  1. Virtual Channel (only useful for unicast)
    //  2. Posted flag (posted multicast has much better performance at larger grid sizes, than non-posted due to
    //  response packets) (60, 45, 23, vs 60, 60, 60 at posted)
};

/// @brief Does L1 Sender Core --> L1 Receiver Cores
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const OneToAllConfig& test_config) {
    // Program
    Program program = CreateProgram();

    if (test_config.is_multicast) {
        assert(test_config.loopback);
    }
    if (!test_config.is_multicast) {
        assert(!test_config.is_linked);
    }

    const uint32_t transaction_size_bytes = test_config.transaction_size_pages * test_config.page_size_bytes;

    if (test_config.loopback && (transaction_size_bytes > 1024 * 1024 / 2)) {
        log_error(tt::LogTest, "Not enough memory for master core using loopback");
        return false;
    }

    CoreRangeSet master_core_set({CoreRange(test_config.master_core_coord)});
    CoreRangeSet subordinate_core_set(
        {CoreRange(CoreCoord(0, 0), CoreCoord(test_config.grid_size.x - 1, test_config.grid_size.y - 1))});
    CoreCoord start_coord = device->worker_core_from_logical_core(CoreCoord(0, 0));
    CoreCoord end_coord =
        device->worker_core_from_logical_core(CoreCoord(test_config.grid_size.x - 1, test_config.grid_size.y - 1));
    if (!test_config.loopback) {
        subordinate_core_set = subordinate_core_set.subtract(master_core_set);
    }
    uint32_t num_subordinates = subordinate_core_set.num_cores();
    uint32_t total_sender_size_bytes =
        test_config.transaction_size_pages * test_config.page_size_bytes * num_subordinates;

    // Obtain L1 Address for Storing Data
    // NOTE: We don't know if the whole block of memory is actually available.
    //       This is something that could probably be checked
    L1AddressInfo master_l1_info =
        tt::tt_metal::unit_tests::dm::get_l1_address_and_size(device, test_config.master_core_coord);

    auto sub_core_list = corerange_to_cores(subordinate_core_set);
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

    uint32_t subordinate_l1_byte_address = master_l1_info.base_address;
    // Offset the address for loopback to avoid data race when master writes to self
    uint32_t master_l1_byte_address =
        test_config.loopback ? subordinate_l1_byte_address + transaction_size_bytes : subordinate_l1_byte_address;

    // Compile-time arguments for kernels
    vector<uint32_t> sender_compile_args = {
        (uint32_t)master_l1_byte_address,
        (uint32_t)subordinate_l1_byte_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id,
        (uint32_t)num_subordinates,
        (uint32_t)total_sender_size_bytes,
        (uint32_t)test_config.is_linked};

    vector<uint32_t> receiver_compile_args = {
        (uint32_t)master_l1_byte_address,
        (uint32_t)subordinate_l1_byte_address,
        (uint32_t)test_config.num_of_transactions,
        (uint32_t)test_config.transaction_size_pages,
        (uint32_t)test_config.page_size_bytes,
        (uint32_t)test_config.test_id};

    // Kernels
    auto sender_kernel = CreateKernel(
        program,
        test_config.is_multicast ? "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/sender_multicast.cpp"
                                 : "tests/tt_metal/tt_metal/data_movement/one_to_all/kernels/sender.cpp",
        master_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = test_config.noc_id,
            .compile_args = sender_compile_args});

    // Semaphores
    CoreRangeSet sem_core_set = subordinate_core_set.merge<CoreRangeSet>(master_core_set);
    const uint32_t sem_id = CreateSemaphore(program, sem_core_set, 0);

    // Runtime Arguments
    std::vector<uint32_t> master_run_args = {sem_id, start_coord.x, start_coord.y, end_coord.x, end_coord.y};
    for (auto& core : sub_core_list) {
        if (!test_config.loopback &&
            (core.x == test_config.master_core_coord.x && core.y == test_config.master_core_coord.y)) {
            continue;
        }
        CoreCoord worker = device->worker_core_from_logical_core(core);
        master_run_args.push_back(worker.x);
        master_run_args.push_back(worker.y);
    }
    SetRuntimeArgs(program, sender_kernel, master_core_set, master_run_args);

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
    detail::WriteToDeviceL1(device, test_config.master_core_coord, master_l1_byte_address, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());
    detail::LaunchProgram(device, program);

    vector<uint32_t> packed_output;
    for (auto& core : sub_core_list) {
        detail::ReadFromDeviceL1(device, core, subordinate_l1_byte_address, transaction_size_bytes, packed_output);
        // Results comparison
        bool pcc = is_close_packed_vectors<bfloat16, uint32_t>(
            packed_output, packed_golden, [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b); });
        if (!pcc) {
            log_error(tt::LogTest, "PCC Check failed");
            log_info(tt::LogTest, "Golden vector");
            print_vector<uint32_t>(packed_golden);
            log_info(tt::LogTest, "Output vector");
            print_vector<uint32_t>(packed_output);
            return false;
        }
    }
    return true;
}
}  // namespace unit_tests::dm::core_to_all

/* ========== Test case for one to all data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAll2x2PacketSizes) {
    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    CoreCoord grid_size = {2, 2};
    NOC noc_id = NOC::NOC_0;
    for (uint32_t l = 0; l < 2; l++) {
        bool loopback = (l == 1);  // fully test loopback = false, then loopback = true
        for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
            for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
                 transaction_size_pages *= 2) {
                // Test config
                unit_tests::dm::core_to_all::OneToAllConfig test_config = {
                    .test_id = unit_tests::dm::core_to_all::START_ID + 0,
                    .master_core_coord = master_core_coord,
                    .grid_size = grid_size,
                    .num_of_transactions = num_of_transactions,
                    .transaction_size_pages = transaction_size_pages,
                    .page_size_bytes = page_size_bytes,
                    .l1_data_format = DataFormat::Float16_b,
                    .loopback = loopback,
                    .noc_id = noc_id,
                    .is_multicast = false,
                    .is_linked = false,
                };

                // Run
                for (unsigned int id = 0; id < num_devices_; id++) {
                    EXPECT_TRUE(run_dm(devices_.at(id), test_config));
                }
            }
        }
    }
}

/* ========== Test case for one to all data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAll4x4PacketSizes) {
    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    CoreCoord grid_size = {4, 4};
    NOC noc_id = NOC::NOC_0;
    for (uint32_t l = 0; l < 2; l++) {
        bool loopback = (l == 1);  // fully test loopback = false, then loopback = true
        for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
            for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
                 transaction_size_pages *= 2) {
                // Test config
                unit_tests::dm::core_to_all::OneToAllConfig test_config = {
                    .test_id = unit_tests::dm::core_to_all::START_ID + 1,
                    .master_core_coord = master_core_coord,
                    .grid_size = grid_size,
                    .num_of_transactions = num_of_transactions,
                    .transaction_size_pages = transaction_size_pages,
                    .page_size_bytes = page_size_bytes,
                    .l1_data_format = DataFormat::Float16_b,
                    .loopback = loopback,
                    .noc_id = noc_id,
                    .is_multicast = false,
                    .is_linked = false,
                };

                // Run
                for (unsigned int id = 0; id < num_devices_; id++) {
                    EXPECT_TRUE(run_dm(devices_.at(id), test_config));
                }
            }
        }
    }
}

/* ========== Test case for one to all data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAll10x10PacketSizes) {
    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    NOC noc_id = NOC::NOC_0;

    CoreCoord cs_grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};
    uint32_t same_grid_size = cs_grid_size.x > cs_grid_size.y ? cs_grid_size.y : cs_grid_size.x;
    CoreCoord grid_size = {same_grid_size, same_grid_size};

    // Limit the grid size to 100 cores because the max allowed kernel args is 256
    if (grid_size.x * grid_size.y > 100) {
        uint32_t smaller_dim = grid_size.x > grid_size.y ? grid_size.y : grid_size.x;
        grid_size = {smaller_dim, smaller_dim};
    }
    for (uint32_t l = 0; l < 2; l++) {
        bool loopback = (l == 1);  // fully test loopback = false, then loopback = true
        for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
            for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
                 transaction_size_pages *= 2) {
                // Test config
                unit_tests::dm::core_to_all::OneToAllConfig test_config = {
                    .test_id = unit_tests::dm::core_to_all::START_ID + 2,
                    .master_core_coord = master_core_coord,
                    .grid_size = grid_size,
                    .num_of_transactions = num_of_transactions,
                    .transaction_size_pages = transaction_size_pages,
                    .page_size_bytes = page_size_bytes,
                    .l1_data_format = DataFormat::Float16_b,
                    .loopback = loopback,
                    .noc_id = noc_id,
                    .is_multicast = false,
                    .is_linked = false,
                };

                // Run
                for (unsigned int id = 0; id < num_devices_; id++) {
                    EXPECT_TRUE(run_dm(devices_.at(id), test_config));
                }
            }
        }
    }
}

/* ========== Test case for one to all multicast data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticast2x2PacketSizes) {
    // Parameters
    uint32_t max_transactions = 128;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    CoreCoord grid_size = {2, 2};
    NOC noc_id = NOC::NOC_0;
    bool is_linked = false;

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            // Test config
            unit_tests::dm::core_to_all::OneToAllConfig test_config = {
                .test_id = unit_tests::dm::core_to_all::START_ID + 3,
                .master_core_coord = master_core_coord,
                .grid_size = grid_size,
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .loopback = true,
                .noc_id = noc_id,
                .is_multicast = true,
                .is_linked = is_linked,
            };

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ========== Test case for one to all multicast data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticast5x5PacketSizes) {
    // Parameters
    uint32_t max_transactions = 128;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    CoreCoord grid_size = {5, 5};
    NOC noc_id = NOC::NOC_0;
    bool is_linked = false;

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            // Test config
            unit_tests::dm::core_to_all::OneToAllConfig test_config = {
                .test_id = unit_tests::dm::core_to_all::START_ID + 4,
                .master_core_coord = master_core_coord,
                .grid_size = grid_size,
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .loopback = true,
                .noc_id = noc_id,
                .is_multicast = true,
                .is_linked = is_linked,
            };

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ========== Test case for one to all multicast data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticast11x10PacketSizes) {
    // Parameters
    uint32_t max_transactions = 128;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    CoreCoord grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};
    NOC noc_id = NOC::NOC_0;
    bool is_linked = false;

    // Limit the grid size to 100 cores because the max allowed kernel args is 256
    if (grid_size.x * grid_size.y > 100) {
        uint32_t smaller_dim = grid_size.x > grid_size.y ? grid_size.y : grid_size.x;
        grid_size = {smaller_dim, smaller_dim};
    }

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            // Test config
            unit_tests::dm::core_to_all::OneToAllConfig test_config = {
                .test_id = unit_tests::dm::core_to_all::START_ID + 5,
                .master_core_coord = master_core_coord,
                .grid_size = grid_size,
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .loopback = true,
                .noc_id = noc_id,
                .is_multicast = true,
                .is_linked = is_linked,
            };

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ========== Test case for one to all multicast data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastLinked2x2PacketSizes) {
    // Parameters
    uint32_t max_transactions = 128;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    CoreCoord grid_size = {2, 2};
    NOC noc_id = NOC::NOC_0;
    bool is_linked = true;

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            // Test config
            unit_tests::dm::core_to_all::OneToAllConfig test_config = {
                .test_id = unit_tests::dm::core_to_all::START_ID + 6,
                .master_core_coord = master_core_coord,
                .grid_size = grid_size,
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .loopback = true,
                .noc_id = noc_id,
                .is_multicast = true,
                .is_linked = is_linked,
            };

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ========== Test case for one to all multicast data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastLinked5x5PacketSizes) {
    // Parameters
    uint32_t max_transactions = 128;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    CoreCoord grid_size = {5, 5};
    NOC noc_id = NOC::NOC_0;
    bool is_linked = true;

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            // Test config
            unit_tests::dm::core_to_all::OneToAllConfig test_config = {
                .test_id = unit_tests::dm::core_to_all::START_ID + 7,
                .master_core_coord = master_core_coord,
                .grid_size = grid_size,
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .loopback = true,
                .noc_id = noc_id,
                .is_multicast = true,
                .is_linked = is_linked,
            };

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ========== Test case for one to all multicast data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticastLinked11x10PacketSizes) {
    // Parameters
    uint32_t max_transactions = 128;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    CoreCoord grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};
    NOC noc_id = NOC::NOC_0;
    bool is_linked = true;

    // Limit the grid size to 100 cores because the max allowed kernel args is 256
    if (grid_size.x * grid_size.y > 100) {
        uint32_t smaller_dim = grid_size.x > grid_size.y ? grid_size.y : grid_size.x;
        grid_size = {smaller_dim, smaller_dim};
    }

    for (uint32_t num_of_transactions = 1; num_of_transactions <= max_transactions; num_of_transactions *= 4) {
        for (uint32_t transaction_size_pages = 1; transaction_size_pages <= max_transaction_size_pages;
             transaction_size_pages *= 2) {
            // Test config
            unit_tests::dm::core_to_all::OneToAllConfig test_config = {
                .test_id = unit_tests::dm::core_to_all::START_ID + 8,
                .master_core_coord = master_core_coord,
                .grid_size = grid_size,
                .num_of_transactions = num_of_transactions,
                .transaction_size_pages = transaction_size_pages,
                .page_size_bytes = page_size_bytes,
                .l1_data_format = DataFormat::Float16_b,
                .loopback = true,
                .noc_id = noc_id,
                .is_multicast = true,
                .is_linked = is_linked,
            };

            // Run
            for (unsigned int id = 0; id < num_devices_; id++) {
                EXPECT_TRUE(run_dm(devices_.at(id), test_config));
            }
        }
    }
}

/* ========== Test case for one to all multicast data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllDirectedIdeal) {
    uint32_t test_id = 52;  // Arbitrary test id

    // Physical Constraints
    auto [page_size_bytes, max_transmittable_bytes, max_transmittable_pages] =
        tt::tt_metal::unit_tests::dm::compute_physical_constraints(arch_, devices_.at(0));

    // Parameters
    uint32_t num_of_transactions = 256;
    uint32_t transaction_size_pages = 256;

    CoreCoord master_core_coord = {0, 0};
    CoreCoord grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};
    NOC noc_id = NOC::NOC_0;
    bool is_linked = true;

    // Limit the grid size to 100 cores because the max allowed kernel args is 256
    if (grid_size.x * grid_size.y > 100) {
        uint32_t smaller_dim = grid_size.x > grid_size.y ? grid_size.y : grid_size.x;
        grid_size = {smaller_dim, smaller_dim};
    }

    unit_tests::dm::core_to_all::OneToAllConfig test_config = {
        .test_id = test_id,
        .master_core_coord = master_core_coord,
        .grid_size = grid_size,
        .num_of_transactions = num_of_transactions,
        .transaction_size_pages = transaction_size_pages,
        .page_size_bytes = page_size_bytes,
        .l1_data_format = DataFormat::Float16_b,
        .loopback = true,
        .noc_id = noc_id,
        .is_multicast = true,
        .is_linked = is_linked,
    };

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
