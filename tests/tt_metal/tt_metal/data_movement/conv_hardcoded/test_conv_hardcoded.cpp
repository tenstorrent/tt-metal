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

namespace unit_tests::dm::conv_hardcoded {

constexpr uint32_t START_ID = 21;

// Test config, i.e. test parameters
struct ConvConfig {
    uint32_t test_id = 0;
    CoreRangeSet dest_core_set;
    std::vector<uint32_t> dest_core_compile_args;
    std::vector<uint32_t> dest_core_runtime_args;
    NOC noc_id = NOC::NOC_0;
    std::string kernel_name = "";
};

/// @brief Does L1 Sender Core --> L1 Receiver Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const ConvConfig& test_config) {
    // Program
    Program program = CreateProgram();

    // Kernels
    auto receiver_kernel = CreateKernel(
        program,
        test_config.kernel_name,
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
}  // namespace unit_tests::dm::conv_hardcoded

TEST_F(DeviceFixture, TensixDataMovementConvActHalo3x3) {
    if (arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Skipping test for non-BH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::conv_hardcoded::START_ID + 0;
    NOC noc_id = NOC::NOC_0;
    std::set<CoreRange> dest_core_set = {CoreRange(CoreCoord(0, 0)), CoreRange(CoreCoord(1, 0))};
    CoreRangeSet wrapper_dest_core_set(dest_core_set);
    std::vector<uint32_t> dest_core_compile_args;
    std::vector<uint32_t> dest_core_runtime_args;

    dest_core_compile_args.push_back(1);                // dilation_h
    dest_core_compile_args.push_back(1);                // dilation_w
    dest_core_compile_args.push_back(64);               // conv_act_c_read_bytes
    dest_core_compile_args.push_back(3);                // window_outer
    dest_core_compile_args.push_back(3);                // window_inner
    dest_core_compile_args.push_back(512);              // act_block_h_datums
    dest_core_compile_args.push_back(48);               // act_block_num_tiles
    dest_core_compile_args.push_back(0);                // 0
    dest_core_compile_args.push_back(3);                // weight_size_w
    dest_core_compile_args.push_back(162);              // conv_act_size_w_padded
    dest_core_compile_args.push_back(0);                // act_block_w_extra_align_bytes
    dest_core_compile_args.push_back(6);                // act_num_blocks_h
    dest_core_compile_args.push_back(0);                // 0
    dest_core_compile_args.push_back(0);                // 0
    dest_core_compile_args.push_back(0);                // 0
    dest_core_compile_args.push_back(0);                // 0
    dest_core_compile_args.push_back(0);                // 0
    dest_core_compile_args.push_back(0);                // 0
    dest_core_compile_args.push_back(0);                // 0
    dest_core_compile_args.push_back(0);                // 0
    dest_core_compile_args.push_back(512);              // act_block_h_datums_last_block
    dest_core_compile_args.push_back(0);                // act_block_h_datums_second_reader
    dest_core_compile_args.push_back(0);                // needs_act_block_zero_out
    dest_core_compile_args.push_back(3);                // cb_id_act
    dest_core_compile_args.push_back(2);                // cb_id_sharded_act
    dest_core_compile_args.push_back(7);                // cb_reader_indices
    dest_core_compile_args.push_back(1565696);          // packed_reader_indices_ptr
    dest_core_compile_args.push_back(1336576);          // act_l1_read_addr
    dest_core_compile_args.push_back(192);              // coalesced_read_bytes
    dest_core_compile_args.push_back(98304);            // l1_write_addr_act
    dest_core_compile_args.push_back(256);              // act_block_h_datums_read_curr
    dest_core_compile_args.push_back(6 * 3 * 256 * 2);  // num_of_transactions = act_num_blocks_h *
                                                        //    window_outer *
                                                        //    act_block_h_datums_read_curr *
                                                        //    (dilation_w != 1 ? 2*weight_size_w : 2)
    dest_core_compile_args.push_back(192);              // transaction_size_bytes = coalesced_read_bytes
    dest_core_compile_args.push_back(test_id);          // test_id

    dest_core_runtime_args.push_back(0);  // 0

    // Test config
    unit_tests::dm::conv_hardcoded::ConvConfig test_config = {
        .test_id = test_id,
        .dest_core_set = wrapper_dest_core_set,
        .dest_core_compile_args = dest_core_compile_args,
        .dest_core_runtime_args = dest_core_runtime_args,
        .noc_id = noc_id,
        .kernel_name =
            "tests/tt_metal/tt_metal/data_movement/conv_hardcoded/kernels/"
            "reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp",
    };

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

TEST_F(DeviceFixture, TensixDataMovementConvActHalo3x3Smaller) {
    if (arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Skipping test for non-BH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::conv_hardcoded::START_ID + 1;
    NOC noc_id = NOC::NOC_0;
    std::set<CoreRange> dest_core_set = {CoreRange(CoreCoord(0, 0)), CoreRange(CoreCoord(1, 0))};
    CoreRangeSet wrapper_dest_core_set(dest_core_set);
    std::vector<uint32_t> dest_core_compile_args;
    std::vector<uint32_t> dest_core_runtime_args;

    dest_core_compile_args.push_back(2);                     // dilation_h
    dest_core_compile_args.push_back(2);                     // dilation_w
    dest_core_compile_args.push_back(16);                    // conv_act_c_read_bytes
    dest_core_compile_args.push_back(4);                     // window_outer
    dest_core_compile_args.push_back(4);                     // window_inner
    dest_core_compile_args.push_back(256);                   // act_block_h_datums
    dest_core_compile_args.push_back(8);                     // act_block_num_tiles
    dest_core_compile_args.push_back(0);                     // 0
    dest_core_compile_args.push_back(4);                     // weight_size_w
    dest_core_compile_args.push_back(115);                   // conv_act_size_w_padded
    dest_core_compile_args.push_back(0);                     // act_block_w_extra_align_bytes
    dest_core_compile_args.push_back(12);                    // act_num_blocks_h
    dest_core_compile_args.push_back(0);                     // 0
    dest_core_compile_args.push_back(0);                     // 0
    dest_core_compile_args.push_back(0);                     // 0
    dest_core_compile_args.push_back(0);                     // 0
    dest_core_compile_args.push_back(0);                     // 0
    dest_core_compile_args.push_back(0);                     // 0
    dest_core_compile_args.push_back(0);                     // 0
    dest_core_compile_args.push_back(0);                     // 0
    dest_core_compile_args.push_back(256);                   // act_block_h_datums_last_block
    dest_core_compile_args.push_back(0);                     // act_block_h_datums_second_reader
    dest_core_compile_args.push_back(0);                     // needs_act_block_zero_out
    dest_core_compile_args.push_back(3);                     // cb_id_act
    dest_core_compile_args.push_back(2);                     // cb_id_sharded_act
    dest_core_compile_args.push_back(7);                     // cb_reader_indices
    dest_core_compile_args.push_back(1566528);               // packed_reader_indices_ptr
    dest_core_compile_args.push_back(1482368);               // act_l1_read_addr
    dest_core_compile_args.push_back(16);                    // coalesced_read_bytes
    dest_core_compile_args.push_back(98304);                 // l1_write_addr_act
    dest_core_compile_args.push_back(128);                   // act_block_h_datums_read_curr
    dest_core_compile_args.push_back(12 * 4 * 128 * 2 * 4);  // num_of_transactions = act_num_blocks_h *
                                                             //    window_outer *
                                                             //    act_block_h_datums_read_curr *
                                                             //    (dilation_w != 1 ? 2*weight_size_w : 2)
    dest_core_compile_args.push_back(16);                    // transaction_size_bytes = coalesced_read_bytes
    dest_core_compile_args.push_back(test_id);               // test_id

    dest_core_runtime_args.push_back(0);  // 0

    // Test config
    unit_tests::dm::conv_hardcoded::ConvConfig test_config = {
        .test_id = test_id,
        .dest_core_set = wrapper_dest_core_set,
        .dest_core_compile_args = dest_core_compile_args,
        .dest_core_runtime_args = dest_core_runtime_args,
        .noc_id = noc_id,
        .kernel_name =
            "tests/tt_metal/tt_metal/data_movement/conv_hardcoded/kernels/"
            "reader_conv_activations_padded_with_halo_3x3_weights_v2.cpp",
    };

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

TEST_F(DeviceFixture, TensixDataMovementConvHaloGather) {
    if (arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Skipping test for non-BH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::conv_hardcoded::START_ID + 2;
    NOC noc_id = NOC::NOC_0;
    std::set<CoreRange> dest_core_set = {CoreRange(CoreCoord(0, 0))};
    CoreRangeSet wrapper_dest_core_set(dest_core_set);
    std::vector<uint32_t> dest_core_compile_args;
    std::vector<uint32_t> dest_core_runtime_args;

    dest_core_compile_args.push_back(0);        // padding_config_cb_id
    dest_core_compile_args.push_back(7);        // gather_config_cb_id
    dest_core_compile_args.push_back(0);        // src_cb_id
    dest_core_compile_args.push_back(4);        // in_cb_id
    dest_core_compile_args.push_back(1);        // out_cb_id
    dest_core_compile_args.push_back(2);        // pad_cb_id
    dest_core_compile_args.push_back(0);        // pad_val_u32
    dest_core_compile_args.push_back(96);       // in_nsticks
    dest_core_compile_args.push_back(64);       // stick_nbytes
    dest_core_compile_args.push_back(0);        // is_block_sharded
    dest_core_compile_args.push_back(0);        // remote_read
    dest_core_compile_args.push_back(0);        // is_col_major
    dest_core_compile_args.push_back(0);        // is_width_sharded
    dest_core_compile_args.push_back(64);       // input_aligned_page_size
    dest_core_compile_args.push_back(0);        // skip_untilize
    dest_core_compile_args.push_back(32);       // block_size_height
    dest_core_compile_args.push_back(1);        // block_size_width_tiles
    dest_core_compile_args.push_back(1);        // block_start_offset
    dest_core_compile_args.push_back(2);        // block_stride
    dest_core_compile_args.push_back(102464);   // in_base_l1_addr
    dest_core_compile_args.push_back(1232128);  // out_base_l1_addr
    dest_core_compile_args.push_back(98304);    // padding_l1_addr
    dest_core_compile_args.push_back(1232128);  // dst_base_addr
    dest_core_compile_args.push_back(11 * 1);   // num_of_transactions
    dest_core_compile_args.push_back(2048);     // transaction_size_bytes
    dest_core_compile_args.push_back(test_id);  // test_id

    dest_core_runtime_args.push_back(11);   // 0
    dest_core_runtime_args.push_back(1);    // 1
    dest_core_runtime_args.push_back(2);    // 2
    dest_core_runtime_args.push_back(1);    // 3
    dest_core_runtime_args.push_back(32);   // 4
    dest_core_runtime_args.push_back(195);  // 5
    dest_core_runtime_args.push_back(32);   // 6

    // Test config
    unit_tests::dm::conv_hardcoded::ConvConfig test_config = {
        .test_id = test_id,
        .dest_core_set = wrapper_dest_core_set,
        .dest_core_compile_args = dest_core_compile_args,
        .dest_core_runtime_args = dest_core_runtime_args,
        .noc_id = noc_id,
        .kernel_name = "tests/tt_metal/tt_metal/data_movement/conv_hardcoded/kernels/halo_gather.cpp",
    };

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
