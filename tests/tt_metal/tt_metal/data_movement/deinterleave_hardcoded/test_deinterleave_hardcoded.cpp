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
using namespace test_utils;

namespace unit_tests::dm::deinterleave_hardcoded {

constexpr uint32_t START_ID = 200;

// Test config, i.e. test parameters
struct DeinterleaveConfig {
    uint32_t test_id = 0;
    vector<CoreRangeSet> dest_core_set;
    vector<vector<uint32_t>> dest_core_compile_args;
    vector<vector<uint32_t>> dest_core_runtime_args;
    NOC noc_id = NOC::NOC_0;
};

/// @brief Does L1 Sender Core --> L1 Receiver Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const DeinterleaveConfig& test_config) {
    // Program
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program = CreateProgram();
    auto& cq = mesh_device->mesh_command_queue();
    auto device = mesh_device->get_devices()[0];

    for (int k = 0; k < test_config.dest_core_set.size(); k++) {
        // Kernels
        auto receiver_kernel = CreateKernel(
            program,
            "tests/tt_metal/tt_metal/data_movement/deinterleave_hardcoded/kernels/deinterleave_kernel_rm.cpp",
            test_config.dest_core_set[k],
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = test_config.noc_id,
                .compile_args = test_config.dest_core_compile_args[k]});

        // Runtime Arguments
        SetRuntimeArgs(program, receiver_kernel, test_config.dest_core_set[k], test_config.dest_core_runtime_args[k]);
    }

    // Assign unique id
    log_info(LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Launch program using slow dispatch
    MetalContext::instance().get_cluster().l1_barrier(device->id());
    distributed::AddProgramToMeshWorkload(workload, std::move(program), device_range);
    distributed::EnqueueMeshWorkload(cq, workload, true);

    return true;
}
}  // namespace unit_tests::dm::deinterleave_hardcoded

TEST_F(MeshDeviceFixture, TensixDataMovementDeinterleaveSingleCore) {
    auto mesh_device = devices_.at(0);
    auto arch_ = mesh_device->arch();

    if (arch_ != ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for non-WH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::deinterleave_hardcoded::START_ID + 0;
    NOC noc_id = NOC::NOC_0;
    vector<CoreRangeSet> send_dest_core_set;
    vector<vector<uint32_t>> send_dest_core_compile_args;
    vector<vector<uint32_t>> send_dest_core_runtime_args;

    {
        set<CoreRange> dest_core_set = {CoreRange(CoreCoord(0, 0))};
        CoreRangeSet wrapper_dest_core_set(dest_core_set);
        vector<uint32_t> dest_core_compile_args;
        vector<uint32_t> dest_core_runtime_args;

        dest_core_compile_args.push_back(0);        // src_cb_id
        dest_core_compile_args.push_back(1);        // dst_cb_id
        dest_core_compile_args.push_back(128);      // width
        dest_core_compile_args.push_back(16);       // height
        dest_core_compile_args.push_back(64);       // stick_size_bytes
        dest_core_compile_args.push_back(2);        // stride_h
        dest_core_compile_args.push_back(2);        // stride_w
        dest_core_compile_args.push_back(0);        // barrier_threshold
        dest_core_compile_args.push_back(test_id);  // test_id

        dest_core_runtime_args.push_back(0);        // start_x
        dest_core_runtime_args.push_back(4);        // end_x
        dest_core_runtime_args.push_back(0);        // start_y
        dest_core_runtime_args.push_back(1);        // end_y
        dest_core_runtime_args.push_back(128);      // src_width_stride
        dest_core_runtime_args.push_back(8192);     // src_height_offset_to_next
        dest_core_runtime_args.push_back(0);        // src_offset
        dest_core_runtime_args.push_back(32768);    // dst_size_bytes
        dest_core_runtime_args.push_back(32768);    // dst_offset
        dest_core_runtime_args.push_back(1);        // offset_x
        dest_core_runtime_args.push_back(0);        // offset_y
        dest_core_runtime_args.push_back(4);        // num_src_cores
        dest_core_runtime_args.push_back(32768);    // dst_rollover_offset
        dest_core_runtime_args.push_back(1236992);  // dst_address_get_write_ptr
        dest_core_runtime_args.push_back(1368064);  // src_address_get_read_ptr

        send_dest_core_set.push_back(wrapper_dest_core_set);
        send_dest_core_compile_args.push_back(dest_core_compile_args);
        send_dest_core_runtime_args.push_back(dest_core_runtime_args);
    }

    for (uint32_t barrier_threshold = 0; barrier_threshold <= 256; barrier_threshold += 16) {
        for (auto& dest_core_compile_args : send_dest_core_compile_args) {
            // Set barrier threshold
            dest_core_compile_args[7] = barrier_threshold == 256 ? 65535 : barrier_threshold;
        }

        // Test config
        unit_tests::dm::deinterleave_hardcoded::DeinterleaveConfig test_config = {
            .test_id = test_id,
            .dest_core_set = send_dest_core_set,
            .dest_core_compile_args = send_dest_core_compile_args,
            .dest_core_runtime_args = send_dest_core_runtime_args,
            .noc_id = noc_id};

        // Run
        EXPECT_TRUE(run_dm(mesh_device, test_config));
    }
}

TEST_F(MeshDeviceFixture, TensixDataMovementDeinterleaveMultiCore) {
    auto mesh_device = devices_.at(0);
    auto arch_ = mesh_device->arch();

    if (arch_ != ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for non-WH architecture";
    }

    // Parameters
    uint32_t test_id = unit_tests::dm::deinterleave_hardcoded::START_ID + 1;
    NOC noc_id = NOC::NOC_0;
    vector<CoreRangeSet> send_dest_core_set;
    vector<vector<uint32_t>> send_dest_core_compile_args;
    vector<vector<uint32_t>> send_dest_core_runtime_args;

    uint32_t offset_y_part_count = 0;
    uint32_t offset_y_part2_count = 0;

    for (uint32_t x = 0; x < 8; x++) {
        for (uint32_t y = 0; y < 8; y++) {
            set<CoreRange> dest_core_set = {CoreRange(CoreCoord(x, y))};
            CoreRangeSet wrapper_dest_core_set(dest_core_set);
            vector<uint32_t> dest_core_compile_args;
            vector<uint32_t> dest_core_runtime_args;

            if (x >= 4) {
                if (y == 3) {
                    offset_y_part2_count++;
                } else if (y == 7) {
                    offset_y_part2_count--;
                }
            } else {
                if (y == 4) {
                    offset_y_part_count++;
                }
            }

            dest_core_compile_args.push_back(0);        // src_cb_id
            dest_core_compile_args.push_back(1);        // dst_cb_id
            dest_core_compile_args.push_back(128);      // width
            dest_core_compile_args.push_back(16);       // height
            dest_core_compile_args.push_back(128);      // stick_size_bytes
            dest_core_compile_args.push_back(4);        // stride_h
            dest_core_compile_args.push_back(4);        // stride_w
            dest_core_compile_args.push_back(32);       // barrier_threshold
            dest_core_compile_args.push_back(test_id);  // test_id

            dest_core_runtime_args.push_back(0);                               // start_x
            dest_core_runtime_args.push_back(8);                               // end_x
            dest_core_runtime_args.push_back(2 * (x % 4) + 0);                 // start_y
            dest_core_runtime_args.push_back(2 * (x % 4) + 2);                 // end_y
            dest_core_runtime_args.push_back(384);                             // src_width_stride
            dest_core_runtime_args.push_back(36864);                           // src_height_offset_to_next
            dest_core_runtime_args.push_back(192 * y);                         // src_offset
            dest_core_runtime_args.push_back(12288);                           // dst_size_bytes
            dest_core_runtime_args.push_back(12288 * y);                       // dst_offset
            dest_core_runtime_args.push_back(2 * (y % 4) + (x >= 4 ? 0 : 1));  // offset_x
            dest_core_runtime_args.push_back(x >= 4 ? offset_y_part2_count : offset_y_part_count);  // offset_y
            dest_core_runtime_args.push_back(16);                                                   // num_src_cores
            dest_core_runtime_args.push_back(x >= 4 ? 0 : 12288);  // dst_rollover_offset
            dest_core_runtime_args.push_back(1105920);             // dst_address_get_write_ptr
            dest_core_runtime_args.push_back(1302528);             // src_address_get_read_ptr

            send_dest_core_set.push_back(wrapper_dest_core_set);
            send_dest_core_compile_args.push_back(dest_core_compile_args);
            send_dest_core_runtime_args.push_back(dest_core_runtime_args);
        }

        if (x >= 4) {
            offset_y_part2_count += 2;
        } else {
            offset_y_part_count++;
        }
    }

    for (uint32_t barrier_threshold = 0; barrier_threshold <= 256; barrier_threshold += 16) {
        for (auto& dest_core_compile_args : send_dest_core_compile_args) {
            // Set barrier threshold
            dest_core_compile_args[7] = barrier_threshold == 256 ? 65535 : barrier_threshold;
            dest_core_compile_args[8] = test_id;
        }

        // Test config
        unit_tests::dm::deinterleave_hardcoded::DeinterleaveConfig test_config = {
            .test_id = test_id,
            .dest_core_set = send_dest_core_set,
            .dest_core_compile_args = send_dest_core_compile_args,
            .dest_core_runtime_args = send_dest_core_runtime_args,
            .noc_id = noc_id};

        // Run
        EXPECT_TRUE(run_dm(mesh_device, test_config));
    }
}

}  // namespace tt::tt_metal
