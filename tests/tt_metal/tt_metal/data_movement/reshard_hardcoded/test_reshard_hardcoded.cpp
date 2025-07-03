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

namespace unit_tests::dm::reshard_hardcoded {

constexpr uint32_t START_ID = 17;

// Test config, i.e. test parameters
struct ReshardConfig {
    uint32_t test_id = 0;
    CoreRangeSet dest_core_set;
    std::vector<uint32_t> dest_core_compile_args;
    std::vector<uint32_t> dest_core_runtime_args;
    NOC noc_id = NOC::NOC_0;
};

/// @brief Does L1 Sender Core --> L1 Receiver Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const ReshardConfig& test_config) {
    // Program
    Program program = CreateProgram();

    // Kernels
    auto receiver_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/reshard_hardcoded/kernels/reshard_reader.cpp",
        test_config.dest_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = test_config.noc_id,
            .compile_args = test_config.dest_core_compile_args});

    // Runtime Arguments
    SetRuntimeArgs(program, receiver_kernel, test_config.dest_core_set, test_config.dest_core_runtime_args);

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Launch program
    MetalContext::instance().get_cluster().l1_barrier(device->id());
    detail::LaunchProgram(device, program);

    return true;
}
}  // namespace unit_tests::dm::reshard_hardcoded

TEST_F(DeviceFixture, TensixDataMovementReshardHardcodedPacketSmallSizes) {
    if (arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Skipping test for non-BH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::reshard_hardcoded::START_ID + 0;
    NOC noc_id = NOC::NOC_0;
    std::set<CoreRange> dest_core_set = {CoreRange(CoreCoord(0, 0)), CoreRange(CoreCoord(1, 0))};
    CoreRangeSet wrapper_dest_core_set(dest_core_set);
    std::vector<uint32_t> dest_core_compile_args;
    std::vector<uint32_t> dest_core_runtime_args;

    dest_core_compile_args.push_back(1566720);  // l1_write_addr
    dest_core_compile_args.push_back(11);       // num_x_cores
    dest_core_compile_args.push_back(10);       // num_y_cores
    dest_core_compile_args.push_back(2048);     // page_size
    dest_core_compile_args.push_back(
        1 * (((66048) & 0x0ffff) >>
             8));  // num_of_transactions = num_ranges * (((stride_size_num_strides_skip)&0x0ffff) >> 8)
    dest_core_compile_args.push_back(
        (66048 >> 16) * 2048);  // transaction_size_bytes = (stride_size_num_strides_skip >> 16) * page_size
    dest_core_compile_args.push_back(test_id);  // test_id

    dest_core_runtime_args.push_back(1);        // 0
    dest_core_runtime_args.push_back(2);        // 1
    dest_core_runtime_args.push_back(3);        // 2
    dest_core_runtime_args.push_back(4);        // 3
    dest_core_runtime_args.push_back(0);        // 4
    dest_core_runtime_args.push_back(0);        // 5
    dest_core_runtime_args.push_back(0);        // 6
    dest_core_runtime_args.push_back(0);        // 7
    dest_core_runtime_args.push_back(0);        // 8
    dest_core_runtime_args.push_back(0);        // 9
    dest_core_runtime_args.push_back(0);        // 10
    dest_core_runtime_args.push_back(2);        // 11
    dest_core_runtime_args.push_back(0);        // 12
    dest_core_runtime_args.push_back(0);        // 13
    dest_core_runtime_args.push_back(0);        // 14
    dest_core_runtime_args.push_back(0);        // 15
    dest_core_runtime_args.push_back(0);        // 16
    dest_core_runtime_args.push_back(0);        // 17
    dest_core_runtime_args.push_back(0);        // 18
    dest_core_runtime_args.push_back(0);        // 19
    dest_core_runtime_args.push_back(0);        // 20
    dest_core_runtime_args.push_back(1570816);  // 21
    dest_core_runtime_args.push_back(2);        // 22
    dest_core_runtime_args.push_back(1);        // 23
    dest_core_runtime_args.push_back(0);        // 24
    dest_core_runtime_args.push_back(256);      // 25
    dest_core_runtime_args.push_back(0);        // 26
    dest_core_runtime_args.push_back(66048);    // 27
    dest_core_runtime_args.push_back(0);        // 28
    dest_core_runtime_args.push_back(0);        // 29
    dest_core_runtime_args.push_back(0);        // 30

    // Test config
    unit_tests::dm::reshard_hardcoded::ReshardConfig test_config = {
        .test_id = test_id,
        .dest_core_set = wrapper_dest_core_set,
        .dest_core_compile_args = dest_core_compile_args,
        .dest_core_runtime_args = dest_core_runtime_args,
        .noc_id = noc_id,
    };

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

TEST_F(DeviceFixture, TensixDataMovementReshardHardcodedPacketMedSizes) {
    if (arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Skipping test for non-BH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::reshard_hardcoded::START_ID + 1;
    NOC noc_id = NOC::NOC_0;
    std::set<CoreRange> dest_core_set = {CoreRange(CoreCoord(0, 0)), CoreRange(CoreCoord(1, 0))};
    CoreRangeSet wrapper_dest_core_set(dest_core_set);
    std::vector<uint32_t> dest_core_compile_args;
    std::vector<uint32_t> dest_core_runtime_args;

    dest_core_compile_args.push_back(1480704);  // l1_write_addr
    dest_core_compile_args.push_back(11);       // num_x_cores
    dest_core_compile_args.push_back(10);       // num_y_cores
    dest_core_compile_args.push_back(2048);     // page_size
    dest_core_compile_args.push_back(
        6 * (((131584) & 0x0ffff) >>
             8));  // num_of_transactions = num_ranges * (((stride_size_num_strides_skip)&0x0ffff) >> 8)
    dest_core_compile_args.push_back(
        (131584 >> 16) * 2048);  // transaction_size_bytes = (stride_size_num_strides_skip >> 16) * page_size
    dest_core_compile_args.push_back(test_id);  // test_id

    dest_core_runtime_args.push_back(1);         // 0
    dest_core_runtime_args.push_back(2);         // 1
    dest_core_runtime_args.push_back(0);         // 2
    dest_core_runtime_args.push_back(0);         // 3
    dest_core_runtime_args.push_back(5);         // 4
    dest_core_runtime_args.push_back(6);         // 5
    dest_core_runtime_args.push_back(0);         // 6
    dest_core_runtime_args.push_back(0);         // 7
    dest_core_runtime_args.push_back(0);         // 8
    dest_core_runtime_args.push_back(0);         // 9
    dest_core_runtime_args.push_back(0);         // 10
    dest_core_runtime_args.push_back(2);         // 11
    dest_core_runtime_args.push_back(0);         // 12
    dest_core_runtime_args.push_back(0);         // 13
    dest_core_runtime_args.push_back(0);         // 14
    dest_core_runtime_args.push_back(0);         // 15
    dest_core_runtime_args.push_back(0);         // 16
    dest_core_runtime_args.push_back(0);         // 17
    dest_core_runtime_args.push_back(0);         // 18
    dest_core_runtime_args.push_back(0);         // 19
    dest_core_runtime_args.push_back(0);         // 20
    dest_core_runtime_args.push_back(1554432);   // 21
    dest_core_runtime_args.push_back(18);        // 22
    dest_core_runtime_args.push_back(6);         // 23
    dest_core_runtime_args.push_back(0);         // 24
    dest_core_runtime_args.push_back(67108864);  // 25
    dest_core_runtime_args.push_back(0);         // 26
    dest_core_runtime_args.push_back(131584);    // 27
    dest_core_runtime_args.push_back(67108864);  // 28
    dest_core_runtime_args.push_back(4);         // 29
    dest_core_runtime_args.push_back(131584);    // 30
    dest_core_runtime_args.push_back(67108864);  // 31
    dest_core_runtime_args.push_back(8);         // 32
    dest_core_runtime_args.push_back(65792);     // 33
    dest_core_runtime_args.push_back(16777216);  // 34
    dest_core_runtime_args.push_back(0);         // 35
    dest_core_runtime_args.push_back(131584);    // 36
    dest_core_runtime_args.push_back(16777216);  // 37
    dest_core_runtime_args.push_back(4);         // 38
    dest_core_runtime_args.push_back(131584);    // 39
    dest_core_runtime_args.push_back(16777216);  // 40
    dest_core_runtime_args.push_back(8);         // 41
    dest_core_runtime_args.push_back(65792);     // 42
    dest_core_runtime_args.push_back(0);         // 43
    dest_core_runtime_args.push_back(0);         // 44
    dest_core_runtime_args.push_back(0);         // 45
    dest_core_runtime_args.push_back(0);         // 46
    dest_core_runtime_args.push_back(0);         // 47
    dest_core_runtime_args.push_back(0);         // 48
    dest_core_runtime_args.push_back(0);         // 49
    dest_core_runtime_args.push_back(0);         // 50

    // Test config
    unit_tests::dm::reshard_hardcoded::ReshardConfig test_config = {
        .test_id = test_id,
        .dest_core_set = wrapper_dest_core_set,
        .dest_core_compile_args = dest_core_compile_args,
        .dest_core_runtime_args = dest_core_runtime_args,
        .noc_id = noc_id,
    };

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

TEST_F(DeviceFixture, TensixDataMovementReshardHardcodedPacketManyCoresSizes) {
    if (arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Skipping test for non-BH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::reshard_hardcoded::START_ID + 2;
    NOC noc_id = NOC::NOC_0;
    std::set<CoreRange> dest_core_set = {
        CoreRange(CoreCoord(0, 0)),
        CoreRange(CoreCoord(1, 0)),
        CoreRange(CoreCoord(2, 0)),
        CoreRange(CoreCoord(3, 0)),
        CoreRange(CoreCoord(4, 0)),
        CoreRange(CoreCoord(5, 0)),
        CoreRange(CoreCoord(6, 0)),
        CoreRange(CoreCoord(7, 0))};
    CoreRangeSet wrapper_dest_core_set(dest_core_set);
    std::vector<uint32_t> dest_core_compile_args;
    std::vector<uint32_t> dest_core_runtime_args;

    dest_core_compile_args.push_back(1499136);  // l1_write_addr
    dest_core_compile_args.push_back(11);       // num_x_cores
    dest_core_compile_args.push_back(10);       // num_y_cores
    dest_core_compile_args.push_back(2048);     // page_size
    dest_core_compile_args.push_back(
        4 * (((131584) & 0x0ffff) >>
             8));  // num_of_transactions = num_ranges * (((stride_size_num_strides_skip)&0x0ffff) >> 8)
    dest_core_compile_args.push_back(
        (131584 >> 16) * 2048);  // transaction_size_bytes = (stride_size_num_strides_skip >> 16) * page_size
    dest_core_compile_args.push_back(test_id);  // test_id

    dest_core_runtime_args.push_back(1);         // 0
    dest_core_runtime_args.push_back(2);         // 1
    dest_core_runtime_args.push_back(3);         // 2
    dest_core_runtime_args.push_back(4);         // 3
    dest_core_runtime_args.push_back(0);         // 4
    dest_core_runtime_args.push_back(0);         // 5
    dest_core_runtime_args.push_back(0);         // 6
    dest_core_runtime_args.push_back(0);         // 7
    dest_core_runtime_args.push_back(0);         // 8
    dest_core_runtime_args.push_back(0);         // 9
    dest_core_runtime_args.push_back(0);         // 10
    dest_core_runtime_args.push_back(2);         // 11
    dest_core_runtime_args.push_back(3);         // 12
    dest_core_runtime_args.push_back(4);         // 13
    dest_core_runtime_args.push_back(5);         // 14
    dest_core_runtime_args.push_back(6);         // 15
    dest_core_runtime_args.push_back(7);         // 16
    dest_core_runtime_args.push_back(8);         // 17
    dest_core_runtime_args.push_back(9);         // 18
    dest_core_runtime_args.push_back(0);         // 19
    dest_core_runtime_args.push_back(0);         // 20
    dest_core_runtime_args.push_back(1564672);   // 21
    dest_core_runtime_args.push_back(16);        // 22
    dest_core_runtime_args.push_back(4);         // 23
    dest_core_runtime_args.push_back(0);         // 24
    dest_core_runtime_args.push_back(65536);     // 25
    dest_core_runtime_args.push_back(0);         // 26
    dest_core_runtime_args.push_back(131584);    // 27
    dest_core_runtime_args.push_back(16777216);  // 28
    dest_core_runtime_args.push_back(0);         // 29
    dest_core_runtime_args.push_back(131584);    // 30
    dest_core_runtime_args.push_back(33554432);  // 31
    dest_core_runtime_args.push_back(0);         // 32
    dest_core_runtime_args.push_back(131584);    // 33
    dest_core_runtime_args.push_back(50331648);  // 34
    dest_core_runtime_args.push_back(0);         // 35
    dest_core_runtime_args.push_back(131584);    // 36
    dest_core_runtime_args.push_back(0);         // 37
    dest_core_runtime_args.push_back(0);         // 38
    dest_core_runtime_args.push_back(0);         // 39
    dest_core_runtime_args.push_back(0);         // 40

    // Test config
    unit_tests::dm::reshard_hardcoded::ReshardConfig test_config = {
        .test_id = test_id,
        .dest_core_set = wrapper_dest_core_set,
        .dest_core_compile_args = dest_core_compile_args,
        .dest_core_runtime_args = dest_core_runtime_args,
        .noc_id = noc_id,
    };

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

TEST_F(DeviceFixture, TensixDataMovementReshardHardcodedPacketSmallCoresToManyCoresSizes) {
    if (arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Skipping test for non-BH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::reshard_hardcoded::START_ID + 3;
    NOC noc_id = NOC::NOC_0;
    std::set<CoreRange> dest_core_set = {
        CoreRange(CoreCoord(0, 0)),
        CoreRange(CoreCoord(1, 0)),
        CoreRange(CoreCoord(2, 0)),
        CoreRange(CoreCoord(3, 0)),
        CoreRange(CoreCoord(4, 0))};
    CoreRangeSet wrapper_dest_core_set(dest_core_set);
    std::vector<uint32_t> dest_core_compile_args;
    std::vector<uint32_t> dest_core_runtime_args;

    dest_core_compile_args.push_back(1558528);  // l1_write_addr
    dest_core_compile_args.push_back(11);       // num_x_cores
    dest_core_compile_args.push_back(10);       // num_y_cores
    dest_core_compile_args.push_back(2048);     // page_size
    dest_core_compile_args.push_back(
        1 * (((131328) & 0x0ffff) >>
             8));  // num_of_transactions = num_ranges * (((stride_size_num_strides_skip)&0x0ffff) >> 8)
    dest_core_compile_args.push_back(
        (131328 >> 16) * 2048);  // transaction_size_bytes = (stride_size_num_strides_skip >> 16) * page_size
    dest_core_compile_args.push_back(test_id);  // test_id

    dest_core_runtime_args.push_back(1);        // 0
    dest_core_runtime_args.push_back(2);        // 1
    dest_core_runtime_args.push_back(0);        // 2
    dest_core_runtime_args.push_back(0);        // 3
    dest_core_runtime_args.push_back(0);        // 4
    dest_core_runtime_args.push_back(0);        // 5
    dest_core_runtime_args.push_back(0);        // 6
    dest_core_runtime_args.push_back(0);        // 7
    dest_core_runtime_args.push_back(0);        // 8
    dest_core_runtime_args.push_back(0);        // 9
    dest_core_runtime_args.push_back(0);        // 10
    dest_core_runtime_args.push_back(2);        // 11
    dest_core_runtime_args.push_back(0);        // 12
    dest_core_runtime_args.push_back(0);        // 13
    dest_core_runtime_args.push_back(0);        // 14
    dest_core_runtime_args.push_back(0);        // 15
    dest_core_runtime_args.push_back(0);        // 16
    dest_core_runtime_args.push_back(0);        // 17
    dest_core_runtime_args.push_back(0);        // 18
    dest_core_runtime_args.push_back(0);        // 19
    dest_core_runtime_args.push_back(0);        // 20
    dest_core_runtime_args.push_back(1562624);  // 21
    dest_core_runtime_args.push_back(2);        // 22
    dest_core_runtime_args.push_back(1);        // 23
    dest_core_runtime_args.push_back(0);        // 24
    dest_core_runtime_args.push_back(0);        // 25
    dest_core_runtime_args.push_back(2);        // 26
    dest_core_runtime_args.push_back(131328);   // 27
    dest_core_runtime_args.push_back(0);        // 28
    dest_core_runtime_args.push_back(0);        // 29
    dest_core_runtime_args.push_back(0);        // 30
    dest_core_runtime_args.push_back(0);        // 31
    dest_core_runtime_args.push_back(0);        // 32
    dest_core_runtime_args.push_back(0);        // 33
    dest_core_runtime_args.push_back(0);        // 34
    dest_core_runtime_args.push_back(0);        // 35
    dest_core_runtime_args.push_back(0);        // 36
    dest_core_runtime_args.push_back(0);        // 37
    dest_core_runtime_args.push_back(0);        // 38
    dest_core_runtime_args.push_back(0);        // 39
    dest_core_runtime_args.push_back(0);        // 40

    // Test config
    unit_tests::dm::reshard_hardcoded::ReshardConfig test_config = {
        .test_id = test_id,
        .dest_core_set = wrapper_dest_core_set,
        .dest_core_compile_args = dest_core_compile_args,
        .dest_core_runtime_args = dest_core_runtime_args,
        .noc_id = noc_id,
    };

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
