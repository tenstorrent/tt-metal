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

    // Sharded L1 buffers
    const uint32_t transaction_size_bytes = test_config.transaction_size_pages * test_config.page_size_bytes;

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
    uint32_t total_sender_size_pages = test_config.transaction_size_pages * num_subordinates;

    auto master_shard_parameters = ShardSpecBuffer(
        master_core_set,
        {1, transaction_size_bytes / 2},
        ShardOrientation::ROW_MAJOR,
        {1, test_config.page_size_bytes / 2},
        {1, test_config.transaction_size_pages});
    auto master_l1_buffer = CreateBuffer(ShardedBufferConfig{
        .device = device,
        .size = transaction_size_bytes,
        .page_size = test_config.page_size_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
        .shard_parameters = std::move(master_shard_parameters),
    });
    uint32_t master_l1_byte_address = master_l1_buffer->address();

    auto subordinate_shard_parameters = ShardSpecBuffer(
        subordinate_core_set,
        {1, transaction_size_bytes / 2},
        ShardOrientation::ROW_MAJOR,
        {1, test_config.page_size_bytes / 2},
        {1, total_sender_size_pages});
    auto subordinate_l1_buffer = CreateBuffer(ShardedBufferConfig{
        .device = device,
        .size = total_sender_size_bytes,
        .page_size = test_config.page_size_bytes,
        .buffer_type = BufferType::L1,
        .buffer_layout = TensorMemoryLayout::WIDTH_SHARDED,
        .shard_parameters = std::move(subordinate_shard_parameters),
    });
    uint32_t subordinate_l1_byte_address = subordinate_l1_buffer->address();

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
    for (auto& core : corerange_to_cores(subordinate_core_set)) {
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
    for (uint32_t i = 1; i < num_subordinates; i++) {
        packed_golden.insert(packed_golden.end(), packed_input.begin(), packed_input.end());
    }

    // Launch program and record outputs
    detail::WriteToBuffer(master_l1_buffer, packed_input);
    MetalContext::instance().get_cluster().l1_barrier(device->id());
    detail::LaunchProgram(device, program);

    vector<uint32_t> packed_output;
    detail::ReadFromBuffer(subordinate_l1_buffer, packed_output);
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
}  // namespace unit_tests::dm::core_to_all

/* ========== Test case for one to all data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAll2x2PacketSizes) {
    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    CoreCoord grid_size = {2, 2};
    bool loopback = true;
    NOC noc_id = NOC::NOC_0;

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

/* ========== Test case for one to all data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAll4x4PacketSizes) {
    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    CoreCoord grid_size = {4, 4};
    bool loopback = true;
    NOC noc_id = NOC::NOC_0;

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

/* ========== Test case for one to all data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAll10x10PacketSizes) {
    // Parameters
    uint32_t max_transactions = 256;
    uint32_t max_transaction_size_pages = 64;
    uint32_t page_size_bytes = arch_ == tt::ARCH::BLACKHOLE ? 64 : 32;  // =Flit size: 32 bytes for WH, 64 for BH
    CoreCoord master_core_coord = {0, 0};
    bool loopback = true;
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

/* ========== Test case for one to all multicast data movement; ========== */
TEST_F(DeviceFixture, TensixDataMovementOneToAllMulticast2x2PacketSizes) {
    // Parameters
    uint32_t max_transactions = 256;
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
    uint32_t max_transactions = 256;
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
    uint32_t max_transactions = 256;
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
    uint32_t max_transactions = 256;
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
    uint32_t max_transactions = 256;
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
    uint32_t max_transactions = 256;
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

    // Parameters
    /*
        L1 Capacity: 1.5 MB (I think, might be wrong)
        - Max transaction size
            = 4 * 32 pages
            = 128 pages * 32 (or 64) bytes/page
            = 4096 bytes for WH; 8192 bytes for BH
        - Max total transaction size
            = 128 transactions * 4096 bytes
            = 524,288 Bytes
            < 1.25 MB ~= L1 buffer capacity (.25 MB is allocated for the kernel code and other overheads)
    */
    uint32_t page_size_bytes, num_of_transactions;
    uint32_t transaction_size_pages = 4 * 32;
    if (arch_ == tt::ARCH::BLACKHOLE) {
        page_size_bytes = 64;  // (=flit size): 64 bytes for BH
        num_of_transactions = 64;
    } else {
        page_size_bytes = 32;  // (=flit size): 32 bytes for WH
        num_of_transactions = 128;
    }
    CoreCoord master_core_coord = {0, 0};
    CoreCoord grid_size = {
        devices_.at(0)->compute_with_storage_grid_size().x, devices_.at(0)->compute_with_storage_grid_size().y};
    NOC noc_id = NOC::NOC_0;
    bool is_linked = true;  // True or False?

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
